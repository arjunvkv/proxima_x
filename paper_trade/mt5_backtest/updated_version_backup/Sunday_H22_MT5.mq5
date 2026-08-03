#include <Trade\Trade.mqh>
CTrade trade;
//+------------------------------------------------------------------+
//|                                                Sunday_H22_MT5.mq5 |
//|                   Sunday Interbank Gap Reversion Strategy (H22)  |
//+------------------------------------------------------------------+
#property copyright "Proxima Trading"
#property version   "1.00"
#property strict

input int      MIN_GAP_PIPS        = 10;      // Min Gap Pips
input int      HOLD_BARS           = 18;      // 90m max hold time
input int      TOP_N               = 5;       // Top 5 largest gap pairs
input double   BASE_LOT            = 2.5;     // Base Lot Size
input ulong    MAGIC_BASE          = 202622;  // Magic Base
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
   double   target_close;
   int      held;
   int      pair_idx;
};

PositionInfo g_t[N_UNIV];
datetime     g_last_bar = 0;
int          g_bar = 0;

int OnInit() {
   Print("=== Sunday H22 v1.00  min_gap=", MIN_GAP_PIPS, " hold=", HOLD_BARS, " n=", TOP_N, " ===");
   for(int i=0; i<N_UNIV; i++) { g_t[i].active=false; g_t[i].ticket=0; g_t[i].held=0; }
   return INIT_SUCCEEDED;
}

void OnDeinit(const int) {
   for(int i=0; i<N_UNIV; i++) { g_t[i].active=false; g_t[i].ticket=0; }
}

bool Open(int p, string side, double lot, double target_pr) {
   string s = UNIV[p];
   MqlTick tk;
   if(!SymbolInfoTick(s, tk)) return false;
   double pr = (side == "BUY") ? tk.ask : tk.bid;
   ENUM_ORDER_TYPE ot = (side == "BUY") ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
   MqlTradeRequest req = {}; MqlTradeResult res = {};
   req.action = TRADE_ACTION_DEAL; req.symbol = s; req.volume = lot;
   req.type = ot; req.price = pr;
   double sl_d = (StringFind(s, "JPY") >= 0) ? 0.40 : 0.0040; // 40p Outer Emergency SL
   double tp_d = (StringFind(s, "JPY") >= 0) ? 0.50 : 0.0050; // 50p Outer Safety TP
   req.sl = (side == "BUY") ? pr - sl_d : pr + sl_d;
   req.tp = (side == "BUY") ? pr + tp_d : pr - tp_d;
   req.deviation = (ulong)(SLIPPAGE_PIPS * 10);
   req.magic = MAGIC_BASE + p; req.comment = "SundayH22_entry";
   if(!OrderSend(req, res) || res.retcode != TRADE_RETCODE_DONE) return false;
   g_t[p].active = true; g_t[p].ticket = res.order; g_t[p].entry = res.price;
   g_t[p].target_close = target_pr; g_t[p].held = 0; g_t[p].pair_idx = p;
   Print("  ENTRY ", s, " ", side, " v=", lot, " at=", res.price, " target=", target_pr);
   return true;
}

bool Close(int p, string why) {
   if(!g_t[p].active) return false;
   string s = UNIV[p];
   if(!PositionSelect(s)) { g_t[p].active = false; g_t[p].ticket = 0; return false; }
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
   req.magic = MAGIC_BASE + p; req.comment = "SundayH22_exit";
   if(!OrderSend(req, res) || res.retcode != TRADE_RETCODE_DONE) return false;
   double pnl = PositionGetDouble(POSITION_PROFIT);
   Print("  CLOSE ", s, " ", why, " pnl=", pnl);
   g_t[p].active = false; g_t[p].ticket = 0;
   return true;
}

void CheckExits() {
   for(int i=0; i<N_UNIV; i++) {
      if(!g_t[i].active) continue;
      string s = UNIV[i];
      MqlTick tk;
      if(SymbolInfoTick(s, tk)) {
         // Gap fill check
         if(g_t[i].target_close > 0) {
            bool filled = false;
            if(tk.bid >= g_t[i].target_close && g_t[i].entry < g_t[i].target_close) filled = true; // Buy fill
            if(tk.bid <= g_t[i].target_close && g_t[i].entry > g_t[i].target_close) filled = true; // Sell fill
            if(filled) { Close(i, "gap_fill"); continue; }
         }
      }
      g_t[i].held++;
      if(g_t[i].held >= HOLD_BARS) Close(i, "expiry");
   }
}

void CheckEntry() {
   int active_cnt = 0;
   for(int i=0; i<N_UNIV; i++) if(g_t[i].active) active_cnt++;
   if(active_cnt > 0) return;

   MqlDateTime dt; TimeToStruct(g_last_bar, dt);
   // Sunday open check (Day of week == 0 and hour == 22 or first bar after weekend gap > 2 hours)
   if(dt.day_of_week != 0 && dt.day_of_week != 1) return;
   if(dt.hour != 22 && dt.hour != 23 && dt.hour != 0) return;

   struct GapInfo {
      int pair_idx;
      double gap_pips;
      double fri_close;
      double sun_open;
      string side;
   };
   GapInfo gaps[N_UNIV];
   int gap_cnt = 0;

   for(int i=0; i<N_UNIV; i++) {
      double cur_open = iOpen(UNIV[i], PERIOD_M5, 0);
      double prv_close = iClose(UNIV[i], PERIOD_M5, 1);
      if(cur_open <= 0 || prv_close <= 0) continue;
      double pt = SymbolInfoDouble(UNIV[i], SYMBOL_POINT);
      if(pt <= 0) continue;

      double diff_pips = (cur_open - prv_close) / (pt * 10.0);
      if(MathAbs(diff_pips) >= MIN_GAP_PIPS) {
         gaps[gap_cnt].pair_idx = i;
         gaps[gap_cnt].gap_pips = MathAbs(diff_pips);
         gaps[gap_cnt].fri_close = prv_close;
         gaps[gap_cnt].sun_open = cur_open;
         gaps[gap_cnt].side = (diff_pips > 0) ? "SELL" : "BUY";
         gap_cnt++;
      }
   }

   if(gap_cnt == 0) return;

   // Sort by largest absolute gap pips
   for(int i=0; i<gap_cnt-1; i++) {
      for(int j=i+1; j<gap_cnt; j++) {
         if(gaps[j].gap_pips > gaps[i].gap_pips) {
            GapInfo tmp = gaps[i]; gaps[i] = gaps[j]; gaps[j] = tmp;
         }
      }
   }

   int n_enter = MathMin(TOP_N, gap_cnt);
   for(int k=0; k<n_enter; k++) {
      Open(gaps[k].pair_idx, gaps[k].side, BASE_LOT, gaps[k].fri_close);
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