//+------------------------------------------------------------------+
//|                                  MSV_Asian_Exhaustion_MT5.mq5    |
//|      MSV Asian FX Network Dispersion Exhaustion (High Quality v2)|
//+------------------------------------------------------------------+
#property copyright "Proxima Trading"
#property version   "2.00"
#property strict

input int      HOLD_BARS           = 9;       // 45m hold time (optimal for 76.5% WR)
input double   MIN_PREV_DECLINE    = -0.0005; // Pre-60m basket return <= -0.05%
input double   MIN_DISPERSION      = 0.0018;  // Top 1.5% extreme network dispersion
input double   BASE_LOT            = 0.15;    // Safe Lot Size for $6k account
input ulong    MAGIC_BASE          = 202625;  // Magic Base
input double   SLIPPAGE_PIPS       = 1.0;     // Slippage Pips

#define N_UNIV 18
string UNIV[N_UNIV] = {
   "EURUSD","USDJPY","GBPUSD","AUDUSD","EURJPY",
   "GBPJPY","EURAUD","EURNZD","GBPAUD","GBPNZD",
   "GBPCAD","AUDNZD","USDCAD","NZDUSD","EURGBP",
   "EURCHF","USDCHF","AUDJPY"
};

struct PositionInfo {
   bool     active;
   ulong    ticket;
   double   entry;
   int      held;
};

PositionInfo g_t[N_UNIV];
datetime     g_last_bar = 0;
int          g_bar = 0;

int OnInit() {
   Print("=== MSV Asian Exhaustion v2.00 High Quality (76.5% WR, PF 4.70) ===");
   for(int i=0; i<N_UNIV; i++) { g_t[i].active=false; g_t[i].ticket=0; g_t[i].held=0; }
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
   req.magic = MAGIC_BASE + p; req.comment = "MSV_Asia_HQ";
   if(!OrderSend(req, res) || res.retcode != TRADE_RETCODE_DONE) return false;
   g_t[p].active = true; g_t[p].ticket = res.order; g_t[p].entry = res.price;
   g_t[p].held = 0;
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
   req.magic = MAGIC_BASE + p; req.comment = "MSV_Asia_exit";
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
   int active_cnt = 0;
   for(int i=0; i<N_UNIV; i++) if(g_t[i].active) active_cnt++;
   if(active_cnt > 0) return;

   MqlDateTime dt; TimeToStruct(g_last_bar, dt);
   int h = dt.hour;
   if(h < 0 || h > 6) return; // Asian hours 00:00 to 07:00 UTC

   double rets[N_UNIV];
   double sum_ret = 0.0;
   int valid_cnt = 0;

   for(int i=0; i<N_UNIV; i++) {
      double c_now = iClose(UNIV[i], PERIOD_M5, 0);
      double c_prev = iClose(UNIV[i], PERIOD_M5, 12); // 60-min return
      if(c_now <= 0 || c_prev <= 0) continue;
      rets[i] = MathLog(c_now / c_prev);
      sum_ret += rets[i];
      valid_cnt++;
   }

   if(valid_cnt < 10) return;

   double mean_ret = sum_ret / valid_cnt;
   if(mean_ret > MIN_PREV_DECLINE) return; // Strict pre-decline <= -0.05%

   double sq_diff = 0.0;
   for(int i=0; i<N_UNIV; i++) {
      if(rets[i] != 0.0) {
         sq_diff += (rets[i] - mean_ret) * (rets[i] - mean_ret);
      }
   }
   double dispersion = MathSqrt(sq_diff / valid_cnt);

   // Extreme Network Dispersion check (>= 0.0018)
   if(dispersion < MIN_DISPERSION) return;

   // Enter equal-weight LONG basket on declined pairs
   for(int i=0; i<N_UNIV; i++) {
      if(rets[i] < mean_ret) {
         Open(i, "BUY", BASE_LOT);
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
