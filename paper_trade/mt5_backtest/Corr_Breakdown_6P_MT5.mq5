//+------------------------------------------------------------------+
//|                               Corr_Breakdown_6P_MT5.mq5          |
//|  #8 Correlation Breakdown 6-Pair Universe MT5 Strategy Tester    |
//+------------------------------------------------------------------+
#property copyright "Proxima Trading"
#property version   "1.00"
#property strict

input double   LEADER_THRESH_PCT   = 0.0015;  // Leader 15m return (>= 0.15%)
input double   LAGGARD_MAX_PCT     = 0.0003;  // Laggard max return (<= 0.03%)
input int      HOLD_BARS           = 6;       // 30m hold time
input double   BASE_LOT            = 0.15;    // Lot Size for $6k account
input ulong    MAGIC_BASE          = 202699;  // Magic Base
input double   SLIPPAGE_PIPS       = 1.0;     // Slippage Pips

#define N_PAIRS 4
string LEADERS[N_PAIRS]  = {"AUDUSD", "EURJPY", "EURAUD", "EURNZD"};
string LAGGARDS[N_PAIRS] = {"NZDUSD", "GBPJPY", "GBPAUD", "GBPNZD"};

struct PairPos {
   bool     active;
   ulong    ticket;
   double   entry;
   int      held;
   int      dir;
};

PairPos g_pos[N_PAIRS];
datetime g_last_bar = 0;

int OnInit() {
   Print("=== #8 Correlation Breakdown 6-Pair Universe Init ===");
   for(int i=0; i<N_PAIRS; i++) { g_pos[i].active=false; g_pos[i].ticket=0; g_pos[i].held=0; g_pos[i].dir=0; }
   return INIT_SUCCEEDED;
}

void OnDeinit(const int) {}

bool OpenLag(int idx, string side, double lot) {
   string s = LAGGARDS[idx];
   MqlTick tk;
   if(!SymbolInfoTick(s, tk)) return false;
   double pr = (side == "BUY") ? tk.ask : tk.bid;
   ENUM_ORDER_TYPE ot = (side == "BUY") ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
   MqlTradeRequest req = {}; MqlTradeResult res = {};
   req.action = TRADE_ACTION_DEAL; req.symbol = s; req.volume = lot;
   req.type = ot; req.price = pr;
   req.deviation = (ulong)(SLIPPAGE_PIPS * 10);
   req.magic = MAGIC_BASE + idx; req.comment = "Corr6P_entry";
   if(!OrderSend(req, res) || res.retcode != TRADE_RETCODE_DONE) return false;
   g_pos[idx].active = true; g_pos[idx].ticket = res.order; g_pos[idx].entry = res.price;
   g_pos[idx].held = 0; g_pos[idx].dir = (side == "BUY") ? 1 : -1;
   Print("  ENTRY ", s, " ", side, " v=", lot, " at=", res.price);
   return true;
}

bool CloseLag(int idx, string why) {
   if(!g_pos[idx].active) return false;
   string s = LAGGARDS[idx];
   if(!PositionSelect(s)) { g_pos[idx].active = false; g_pos[idx].ticket = 0; return false; }
   double vol = NormalizeDouble(PositionGetDouble(POSITION_VOLUME), 2);
   ulong tix = (ulong)PositionGetInteger(POSITION_TICKET);
   ENUM_POSITION_TYPE pos_type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
   MqlTick tk;
   if(!SymbolInfoTick(s, tk)) return false;
   
   MqlTradeRequest req = {}; MqlTradeResult res = {};
   req.action = TRADE_ACTION_DEAL; req.symbol = s; req.volume = vol; req.position = tix;
   if(pos_type == POSITION_TYPE_BUY) {
      req.type = ORDER_TYPE_SELL; req.price = tk.bid;
   } else {
      req.type = ORDER_TYPE_BUY; req.price = tk.ask;
   }
   req.deviation = (ulong)(SLIPPAGE_PIPS * 10);
   req.magic = MAGIC_BASE + idx; req.comment = "Corr6P_exit";
   if(!OrderSend(req, res) || res.retcode != TRADE_RETCODE_DONE) return false;
   double pnl = PositionGetDouble(POSITION_PROFIT);
   Print("  CLOSE ", s, " ", why, " pnl=", pnl);
   g_pos[idx].active = false; g_pos[idx].ticket = 0;
   return true;
}

void CheckExits() {
   for(int i=0; i<N_PAIRS; i++) {
      if(!g_pos[i].active) continue;
      g_pos[i].held++;
      if(g_pos[i].held >= HOLD_BARS) CloseLag(i, "expiry");
   }
}

void CheckEntry() {
   for(int i=0; i<N_PAIRS; i++) {
      if(g_pos[i].active) continue;
      string s_lead = LEADERS[i];
      string s_lag = LAGGARDS[i];

      double c_lead_now = iClose(s_lead, PERIOD_M5, 0);
      double c_lead_prev = iClose(s_lead, PERIOD_M5, 3);
      double c_lag_now = iClose(s_lag, PERIOD_M5, 0);
      double c_lag_prev = iClose(s_lag, PERIOD_M5, 3);

      if(c_lead_now <= 0 || c_lead_prev <= 0 || c_lag_now <= 0 || c_lag_prev <= 0) continue;

      double ret_lead = (c_lead_now - c_lead_prev) / c_lead_prev;
      double ret_lag = (c_lag_now - c_lag_prev) / c_lag_prev;

      if(ret_lead >= LEADER_THRESH_PCT && ret_lag <= LAGGARD_MAX_PCT) {
         OpenLag(i, "BUY", BASE_LOT);
      } else if(ret_lead <= -LEADER_THRESH_PCT && ret_lag >= -LAGGARD_MAX_PCT) {
         OpenLag(i, "SELL", BASE_LOT);
      }
   }
}

void OnTick() {
   datetime t[1];
   if(CopyTime(_Symbol, PERIOD_M5, 0, 1, t) != 1) return;
   if(t[0] == g_last_bar) return;
   g_last_bar = t[0];
   CheckExits();
   CheckEntry();
}
