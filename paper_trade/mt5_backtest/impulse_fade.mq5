#property copyright "Proxima AI"
#property link      ""
#property version   "1.00"
#property description "ImpulseFade: 5p/20s det, 30s hold, 10p stop, 14-19 UTC, 2.0 lot"

input double   InpLotSize        = 2.0;
input int      InpDetectPips     = 5;
input int      InpWindowSec      = 20;
input int      InpHoldSec        = 30;
input int      InpStopPips       = 10;
input int      InpStartHour      = 14;
input int      InpEndHour        = 19;
input int      InpMagic          = 202608;

#define RING_SZ 4000

struct STick { datetime time; double mid; };
STick  g_t[RING_SZ];
int    g_total = 0;

double g_minv[RING_SZ]; int g_mini[RING_SZ]; int g_minh = 0, g_mint = 0;
double g_maxv[RING_SZ]; int g_maxi[RING_SZ]; int g_maxh = 0, g_maxt = 0;

int    g_ws = 0;
int    g_last_ws = -1;

double g_pip;
double g_thresh;

int    g_ticket = -1;
double g_entry_price = 0;
datetime g_entry_time = 0;
int    g_dir = 0;

int    g_trades = 0;
int    g_wins = 0;
double g_total_pnl = 0;

int OnInit() {
   g_pip = SymbolInfoDouble(_Symbol, SYMBOL_POINT) * 10;
   g_thresh = InpDetectPips * g_pip;
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason) {
   Print("=== FINAL ===");
   Print("Trades: ", g_trades, " Wins: ", g_wins, " PnL: ", DoubleToString(g_total_pnl, 1), " pips");
   if (g_trades > 0)
      Print("WR: ", DoubleToString(100.0 * g_wins / g_trades, 1), "%");
}

void OnTick() {
   MqlTick tick;
   if (!SymbolInfoTick(_Symbol, tick)) return;

   double mid = (tick.bid + tick.ask) / 2.0;
   datetime ts = tick.time;

   // --- position mgmt (check before session filter) ---
   if (g_ticket >= 0) {
      double hold = (double)(ts - g_entry_time);
      if (hold >= InpHoldSec) {
         double exit_price = (g_dir == 1) ? tick.bid : tick.ask;
         MqlTradeRequest req = {};
         MqlTradeResult res = {};
         req.action = TRADE_ACTION_DEAL;
         req.symbol = _Symbol;
         req.volume = InpLotSize;
         req.type = (g_dir == 1) ? ORDER_TYPE_SELL : ORDER_TYPE_BUY;
         req.position = g_ticket;
         req.price = NormalizeDouble(exit_price, 5);
         req.deviation = 5;
         req.magic = InpMagic;
         if (OrderSend(req, res)) {
            double pnl = (exit_price - g_entry_price) / g_pip;
            if (g_dir == -1) pnl = -pnl;
            g_trades++;
            g_total_pnl += pnl;
            if (pnl > 0) g_wins++;
            g_ticket = -1;
         }
      }
      return;
   }

   // --- session filter ---
   MqlDateTime dt;
   TimeToStruct(TimeGMT(), dt);
   if (dt.hour < InpStartHour || dt.hour > InpEndHour) return;

   // --- store tick ---
   int idx = g_total % RING_SZ;
   g_t[idx].time = ts;
   g_t[idx].mid = mid;
   g_total++;

   // --- compress deques ---
   if (g_mint >= RING_SZ - 10) { int n = g_mint - g_minh; ArrayCopy(g_minv, g_minv, 0, g_minh, n); ArrayCopy(g_mini, g_mini, 0, g_minh, n); g_mint = n; g_minh = 0; }
   if (g_maxt >= RING_SZ - 10) { int n = g_maxt - g_maxh; ArrayCopy(g_maxv, g_maxv, 0, g_maxh, n); ArrayCopy(g_maxi, g_maxi, 0, g_maxh, n); g_maxt = n; g_maxh = 0; }

   // --- update monotonic deques ---
   while (g_mint > g_minh && g_minv[g_mint - 1] >= mid) g_mint--;
   g_minv[g_mint] = mid; g_mini[g_mint] = g_total - 1; g_mint++;

   while (g_maxt > g_maxh && g_maxv[g_maxt - 1] <= mid) g_maxt--;
   g_maxv[g_maxt] = mid; g_maxi[g_maxt] = g_total - 1; g_maxt++;

   // --- slide window ---
   while (g_total - 1 - g_ws >= 2 && ts - g_t[g_ws % RING_SZ].time > InpWindowSec) {
      if (g_minh < g_mint && g_mini[g_minh] == g_ws) g_minh++;
      if (g_maxh < g_maxt && g_maxi[g_maxh] == g_ws) g_maxh++;
      g_ws++;
   }

   if (g_total - 1 <= g_ws) return;
   if (ts - g_t[g_ws % RING_SZ].time > InpWindowSec) return;

   // ensure deques have data
   if (g_minh >= g_mint) { g_minv[0]=mid; g_mini[0]=g_total-1; g_minh=0; g_mint=1; }
   if (g_maxh >= g_maxt) { g_maxv[0]=mid; g_maxi[0]=g_total-1; g_maxh=0; g_maxt=1; }

   double wp = g_t[g_ws % RING_SZ].mid;
   double hp = g_maxv[g_maxh] - wp;
   double lp = wp - g_minv[g_minh];

   if (hp < g_thresh && lp < g_thresh) return;
   if (g_ws <= g_last_ws) return;
   g_last_ws = g_ws;

   int ext_dir;
   if (hp >= g_thresh && lp >= g_thresh) {
      if (hp >= lp) ext_dir = 1;
      else ext_dir = -1;
   } else if (hp >= g_thresh) {
      ext_dir = 1;
   } else {
      ext_dir = -1;
   }

   int fade_dir = -ext_dir;

   double price = (fade_dir == 1) ? tick.ask : tick.bid;
   price = NormalizeDouble(price, 5);
   double sl = (fade_dir == 1) ? price - InpStopPips * g_pip : price + InpStopPips * g_pip;
   sl = NormalizeDouble(sl, 5);

   MqlTradeRequest req = {};
   MqlTradeResult res = {};
   req.action = TRADE_ACTION_DEAL;
   req.symbol = _Symbol;
   req.volume = InpLotSize;
   req.type = (fade_dir == 1) ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
   req.price = price;
   req.sl = sl;
   req.deviation = 5;
   req.magic = InpMagic;
   req.comment = "ImpulseFade";

   if (OrderSend(req, res)) {
      g_ticket = (int)res.order;
      g_entry_price = price;
      g_entry_time = ts;
      g_dir = fade_dir;
   }
}

double OnTester() {
   double score = 0;
   if (g_trades > 0) {
      double wr = 100.0 * g_wins / g_trades;
      score = wr + g_total_pnl * 0.1;
   }
   return score;
}
