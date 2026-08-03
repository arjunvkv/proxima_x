//+------------------------------------------------------------------+
//|                                                   CPPF_Z_MT5.mq5 |
//|          Cross-Pair Volatility Dislocation Strategy (Z >= 5.0)   |
//+------------------------------------------------------------------+
#property copyright "Proxima Trading"
#property version   "1.00"
#property strict

input double   Z_THRESH            = 6.0;     // Z-score threshold (6-Sigma shock)
input int      LOOKBACK_BARS       = 200;     // 200 M5 bars rolling window
input int      RET_PERIOD          = 3;       // 3-bar (15-min) return
input int      HOLD_BARS           = 18;      // 90m hold time
input double   BASE_LOT            = 2.5;     // Base Lot Size
input ulong    MAGIC_BASE          = 202623;  // Magic Base
input double   SLIPPAGE_PIPS       = 1.0;     // Slippage Pips

#define N_CPPF 5
string CPPF_PAIRS[N_CPPF] = {"EURNZD", "AUDNZD", "GBPNZD", "GBPAUD", "EURAUD"};

struct PositionInfo {
   bool     active;
   ulong    ticket;
   double   entry;
   int      held;
   string   side;
};

PositionInfo g_t[N_CPPF];
datetime     g_last_bar = 0;
int          g_bar = 0;

int OnInit() {
   Print("=== CPPF Z v1.00  z=", Z_THRESH, " lookback=", LOOKBACK_BARS, " hold=", HOLD_BARS, " ===");
   for(int i=0; i<N_CPPF; i++) { g_t[i].active=false; g_t[i].ticket=0; g_t[i].held=0; }
   return INIT_SUCCEEDED;
}

void OnDeinit(const int) {
   for(int i=0; i<N_CPPF; i++) { g_t[i].active=false; g_t[i].ticket=0; }
}

bool Open(int p, string side, double lot) {
   string s = CPPF_PAIRS[p];
   MqlTick tk;
   if(!SymbolInfoTick(s, tk)) return false;
   double pr = (side == "BUY") ? tk.ask : tk.bid;
   ENUM_ORDER_TYPE ot = (side == "BUY") ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
   MqlTradeRequest req = {}; MqlTradeResult res = {};
   req.action = TRADE_ACTION_DEAL; req.symbol = s; req.volume = lot;
   req.type = ot; req.price = pr;
   req.deviation = (ulong)(SLIPPAGE_PIPS * 10);
   req.magic = MAGIC_BASE + p; req.comment = "CPPF_Z_entry";
   if(!OrderSend(req, res) || res.retcode != TRADE_RETCODE_DONE) return false;
   g_t[p].active = true; g_t[p].ticket = res.order; g_t[p].entry = res.price;
   g_t[p].held = 0; g_t[p].side = side;
   Print("  ENTRY ", s, " ", side, " v=", lot, " at=", res.price);
   return true;
}

bool Close(int p, string why) {
   if(!g_t[p].active) return false;
   string s = CPPF_PAIRS[p];
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
   req.magic = MAGIC_BASE + p; req.comment = "CPPF_Z_exit";
   if(!OrderSend(req, res) || res.retcode != TRADE_RETCODE_DONE) return false;
   double pnl = PositionGetDouble(POSITION_PROFIT);
   Print("  CLOSE ", s, " ", why, " pnl=", pnl);
   g_t[p].active = false; g_t[p].ticket = 0;
   return true;
}

void CheckExits() {
   for(int i=0; i<N_CPPF; i++) {
      if(!g_t[i].active) continue;
      g_t[i].held++;
      if(g_t[i].held >= HOLD_BARS) Close(i, "expiry");
   }
}

void CheckEntry() {
   for(int i=0; i<N_CPPF; i++) {
      if(g_t[i].active) continue;
      string s = CPPF_PAIRS[i];
      
      // Calculate 3-bar return history over LOOKBACK_BARS window
      double returns[];
      ArrayResize(returns, LOOKBACK_BARS);
      double sum = 0.0;
      
      for(int k=0; k<LOOKBACK_BARS; k++) {
         double c_now = iClose(s, PERIOD_M5, k);
         double c_prev = iClose(s, PERIOD_M5, k + RET_PERIOD);
         if(c_now <= 0 || c_prev <= 0) continue;
         returns[k] = MathLog(c_now / c_prev);
         sum += returns[k];
      }
      
      double mean = sum / LOOKBACK_BARS;
      double sq_sum = 0.0;
      for(int k=0; k<LOOKBACK_BARS; k++) {
         sq_sum += (returns[k] - mean) * (returns[k] - mean);
      }
      double std_dev = MathSqrt(sq_sum / LOOKBACK_BARS);
      if(std_dev <= 0) continue;
      
      double cur_ret = returns[0];
      double z_score = (cur_ret - mean) / std_dev;
      
      if(z_score <= -Z_THRESH) {
         Open(i, "BUY", BASE_LOT);
      } else if(z_score >= Z_THRESH) {
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
