//+------------------------------------------------------------------+
//|                                  Corr_Breakdown_MT5.mq5          |
//|    #8 Correlation Breakdown Laggard Catch-Up Pairs Trading EA    |
//+------------------------------------------------------------------+
#property copyright "Proxima Trading"
#property version   "1.00"
#property strict

input double   LEADER_THRESH_PCT   = 0.0015;  // Leader 15m return (>= 0.15%)
input double   LAGGARD_MAX_PCT     = 0.0003;  // Laggard max return (<= 0.03%)
input int      HOLD_BARS           = 6;       // 30m hold time
input double   BASE_LOT            = 0.15;    // Lot Size for $6k account
input ulong    MAGIC_BASE          = 202688;  // Magic Base
input double   SLIPPAGE_PIPS       = 1.0;     // Slippage Pips

string LEADER_PAIR  = "AUDUSD";
string LAGGARD_PAIR = "NZDUSD";

bool   g_active = false;
ulong  g_ticket = 0;
int    g_held = 0;
int    g_dir = 0;
datetime g_last_bar = 0;

int OnInit() {
   Print("=== #8 Correlation Breakdown Laggard Catch-Up EA Init ===");
   return INIT_SUCCEEDED;
}

void OnDeinit(const int) {}

bool OpenLaggard(string side, double lot) {
   MqlTick tk;
   if(!SymbolInfoTick(LAGGARD_PAIR, tk)) return false;
   double pr = (side == "BUY") ? tk.ask : tk.bid;
   ENUM_ORDER_TYPE ot = (side == "BUY") ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
   MqlTradeRequest req = {}; MqlTradeResult res = {};
   req.action = TRADE_ACTION_DEAL; req.symbol = LAGGARD_PAIR; req.volume = lot;
   req.type = ot; req.price = pr;
   req.deviation = (ulong)(SLIPPAGE_PIPS * 10);
   req.magic = MAGIC_BASE; req.comment = "Corr_Laggard_entry";
   if(!OrderSend(req, res) || res.retcode != TRADE_RETCODE_DONE) return false;
   g_active = true; g_ticket = res.order; g_held = 0; g_dir = (side == "BUY") ? 1 : -1;
   Print("  ENTRY ", LAGGARD_PAIR, " ", side, " v=", lot, " at=", res.price);
   return true;
}

bool CloseLaggard(string why) {
   if(!g_active) return false;
   if(!PositionSelect(LAGGARD_PAIR)) { g_active = false; g_ticket = 0; return false; }
   double vol = NormalizeDouble(PositionGetDouble(POSITION_VOLUME), 2);
   ulong tix = (ulong)PositionGetInteger(POSITION_TICKET);
   ENUM_POSITION_TYPE pos_type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
   MqlTick tk;
   if(!SymbolInfoTick(LAGGARD_PAIR, tk)) return false;
   
   MqlTradeRequest req = {}; MqlTradeResult res = {};
   req.action = TRADE_ACTION_DEAL; req.symbol = LAGGARD_PAIR; req.volume = vol; req.position = tix;
   if(pos_type == POSITION_TYPE_BUY) {
      req.type = ORDER_TYPE_SELL; req.price = tk.bid;
   } else {
      req.type = ORDER_TYPE_BUY; req.price = tk.ask;
   }
   req.deviation = (ulong)(SLIPPAGE_PIPS * 10);
   req.magic = MAGIC_BASE; req.comment = "Corr_Laggard_exit";
   if(!OrderSend(req, res) || res.retcode != TRADE_RETCODE_DONE) return false;
   double pnl = PositionGetDouble(POSITION_PROFIT);
   Print("  CLOSE ", LAGGARD_PAIR, " ", why, " pnl=", pnl);
   g_active = false; g_ticket = 0;
   return true;
}

void CheckExits() {
   if(!g_active) return;
   g_held++;
   if(g_held >= HOLD_BARS) CloseLaggard("expiry");
}

void CheckEntry() {
   if(g_active) return;

   double c_lead_now = iClose(LEADER_PAIR, PERIOD_M5, 0);
   double c_lead_prev = iClose(LEADER_PAIR, PERIOD_M5, 3);
   double c_lag_now = iClose(LAGGARD_PAIR, PERIOD_M5, 0);
   double c_lag_prev = iClose(LAGGARD_PAIR, PERIOD_M5, 3);

   if(c_lead_now <= 0 || c_lead_prev <= 0 || c_lag_now <= 0 || c_lag_prev <= 0) return;

   double ret_lead = (c_lead_now - c_lead_prev) / c_lead_prev;
   double ret_lag = (c_lag_now - c_lag_prev) / c_lag_prev;

   // Correlation breakdown trigger: Leader moved >= +0.15% while Laggard moved <= +0.03%
   if(ret_lead >= LEADER_THRESH_PCT && ret_lag <= LAGGARD_MAX_PCT) {
      OpenLaggard("BUY", BASE_LOT);
   } else if(ret_lead <= -LEADER_THRESH_PCT && ret_lag >= -LAGGARD_MAX_PCT) {
      OpenLaggard("SELL", BASE_LOT);
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
