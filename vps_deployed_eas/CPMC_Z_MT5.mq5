//+------------------------------------------------------------------+
//|                                     CPMC_Z_MT5.mq5               |
//|      Cross-Pair Momentum Shock Continuation (Strategy #6)       |
//+------------------------------------------------------------------+
#property copyright "Proxima Trading"
#property version   "1.00"
#property strict

input double   Z_THRESH            = 4.5;     // Z-score threshold (>= 4.5)
input int      LOOKBACK_BARS       = 200;     // Rolling window bars
input int      HOLD_BARS           = 9;       // 45m hold time
input double   BASE_LOT            = 0.15;    // Base Lot Size for $6k account
input ulong    MAGIC_BASE          = 202626;  // Magic Base
input double   SLIPPAGE_PIPS       = 1.0;     // Slippage Pips

#define N_UNIV 3
string UNIV[N_UNIV] = {"EURAUD", "GBPAUD", "GBPNZD"};

struct PositionInfo {
   bool     active;
   ulong    ticket;
   double   entry;
   int      held;
   int      dir; // 1 = BUY, -1 = SELL
};

PositionInfo g_t[N_UNIV];
datetime     g_last_bar = 0;
int          g_bar = 0;

int OnInit() {
   Print("=== CPMC Momentum Continuation v1.00 z=", Z_THRESH, " hold=", HOLD_BARS, " ===");
   for(int i=0; i<N_UNIV; i++) { g_t[i].active=false; g_t[i].ticket=0; g_t[i].held=0; g_t[i].dir=0; }
   return INIT_SUCCEEDED;
}

void OnDeinit(const int) {
   for(int i=0; i<N_UNIV; i++) { g_t[i].active=false; g_t[i].ticket=0; }
}

bool Open(int p, string side, double lot) {
   string s = UNIV[p];
   MqlTick tk;
   if(!SymbolInfoTick(s, tk)) return false;
   double pr = (side == "BUY") ? tk.ask : tk.bid;
   ENUM_ORDER_TYPE ot = (side == "BUY") ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
   MqlTradeRequest req = {}; MqlTradeResult res = {};
   req.action = TRADE_ACTION_DEAL; req.symbol = s; req.volume = lot;
   req.type = ot; req.price = pr;
   req.deviation = (ulong)(SLIPPAGE_PIPS * 10);
   req.magic = MAGIC_BASE + p; req.comment = "CPMC_Mom_entry";
   if(!OrderSend(req, res) || res.retcode != TRADE_RETCODE_DONE) return false;
   g_t[p].active = true; g_t[p].ticket = res.order; g_t[p].entry = res.price;
   g_t[p].held = 0; g_t[p].dir = (side == "BUY") ? 1 : -1;
   Print("  ENTRY ", s, " ", side, " v=", lot, " at=", res.price);
   return true;
}

bool Close(int p, string why) {
   if(!g_t[p].active) return false;
   string s = UNIV[p];
   if(!PositionSelect(s)) { g_t[p].active = false; g_t[p].ticket = 0; return false; }
   double vol = (float)PositionGetDouble(POSITION_VOLUME);
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
   req.magic = MAGIC_BASE + p; req.comment = "CPMC_Mom_exit";
   if(!OrderSend(req, res) || res.retcode != TRADE_RETCODE_DONE) return false;
   double pnl = PositionGetDouble(POSITION_PROFIT);
   Print("  CLOSE ", s, " ", why, " pnl=", pnl);
   g_t[p].active = false; g_t[p].ticket = 0;
   return true;
}

void CheckExits() {
   for(int i=0; i<N_UNIV; i++) {
      if(!g_t[i].active) continue;
      g_t[i].held++;
      if(g_t[i].held >= HOLD_BARS) Close(i, "expiry");
   }
}

void CheckEntry() {
   for(int i=0; i<N_UNIV; i++) {
      if(g_t[i].active) continue;
      string s = UNIV[i];
      
      // Calculate 3-bar (15-min) return Z-score over 200-bar rolling window
      double c_now = iClose(s, PERIOD_M5, 0);
      double c_prev = iClose(s, PERIOD_M5, 3);
      if(c_now <= 0 || c_prev <= 0) continue;
      double cur_ret = MathLog(c_now / c_prev);

      double sum_ret = 0.0;
      double rets[200];
      int n_rets = 0;
      for(int k=1; k<=200; k++) {
         double p1 = iClose(s, PERIOD_M5, k);
         double p2 = iClose(s, PERIOD_M5, k+3);
         if(p1 > 0 && p2 > 0) {
            rets[n_rets] = MathLog(p1 / p2);
            sum_ret += rets[n_rets];
            n_rets++;
         }
      }
      if(n_rets < 100) continue;
      double mean = sum_ret / n_rets;
      double sq_diff = 0.0;
      for(int k=0; k<n_rets; k++) sq_diff += (rets[k] - mean) * (rets[k] - mean);
      double std_dev = MathSqrt(sq_diff / n_rets);
      if(std_dev <= 0) continue;

      double z = (cur_ret - mean) / std_dev;

      // Enter Momentum Continuation Trade (BUY if Z >= 4.5, SELL if Z <= -4.5)
      if(z >= Z_THRESH) {
         Open(i, "BUY", BASE_LOT);
      } else if(z <= -Z_THRESH) {
         Open(i, "SELL", BASE_LOT);
      }
   }
}

void OnTick() {
   datetime t[1];
   if(CopyTime(_Symbol, PERIOD_M5, 0, 1, t) != 1) return;
   if(t[0] == g_last_bar) return;
   g_last_bar = t[0]; g_bar++;
   CheckExits();
   CheckEntry();
}
