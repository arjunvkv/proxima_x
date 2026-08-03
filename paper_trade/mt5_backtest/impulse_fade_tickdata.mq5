#property version   "1.00"
#property description "ImpulseFade — processes real ticks from impulse_ticks.bin"

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
STick g_t[RING_SZ];
int   g_total = 0;

double g_minv[RING_SZ]; int g_mini[RING_SZ]; int g_minh = 0, g_mint = 0;
double g_maxv[RING_SZ]; int g_maxi[RING_SZ]; int g_maxh = 0, g_maxt = 0;

int g_ws = 0;
int g_last_ws = -1;

int g_trades = 0, g_wins = 0;
double g_total_pnl = 0;

int OnInit() {
   double pip = SymbolInfoDouble(_Symbol, SYMBOL_POINT) * 10;
   double thresh = InpDetectPips * pip;

   int h = FileOpen("impulse_ticks.bin", FILE_BIN | FILE_READ | FILE_COMMON);
   if (h == INVALID_HANDLE) { Print("ERROR: Cannot open impulse_ticks.bin"); return INIT_FAILED; }

   long fsize = FileSize(h);
   int nticks = (int)(fsize / 24);
   Print("Processing " + IntegerToString(nticks) + " ticks");

   int in_pos = 0, fade_dir = 0;
   double entry_price = 0, sl_price = 0;
   datetime entry_time = 0;

   for (int i = 0; i < nticks && !IsStopped(); i++) {
      datetime ts = (datetime)FileReadLong(h);
      double bid = FileReadDouble(h);
      double ask = FileReadDouble(h);
      double mid = (bid + ask) / 2.0;

      MqlDateTime dt;
      TimeToStruct(ts, dt);
      int hr = dt.hour;
      if (hr < InpStartHour || hr > InpEndHour) {
         if (in_pos) {
            double hold = (double)(ts - entry_time);
            if (hold >= InpHoldSec || (fade_dir == 1 && bid <= sl_price) || (fade_dir == -1 && ask >= sl_price)) {
               double exit_price = (fade_dir == 1) ? bid : ask;
               double pnl = (exit_price - entry_price) / pip;
               if (fade_dir == -1) pnl = -pnl;
               if (pnl > 0) g_wins++;
               g_trades++; g_total_pnl += pnl;
               in_pos = 0; g_total = 0; g_ws = 0; g_last_ws = -1;
               g_minh = 0; g_mint = 0; g_maxh = 0; g_maxt = 0;
            }
         }
         continue;
      }

      if (in_pos) {
         double hold = (double)(ts - entry_time);
         if (hold >= InpHoldSec || (fade_dir == 1 && bid <= sl_price) || (fade_dir == -1 && ask >= sl_price)) {
            double exit_price = (fade_dir == 1) ? bid : ask;
            double pnl = (exit_price - entry_price) / pip;
            if (fade_dir == -1) pnl = -pnl;
            if (pnl > 0) g_wins++;
            g_trades++; g_total_pnl += pnl;
            in_pos = 0; g_total = 0; g_ws = 0; g_last_ws = -1;
            g_minh = 0; g_mint = 0; g_maxh = 0; g_maxt = 0;
         }
         continue;
      }

      // --- detection ---
      int idx = g_total % RING_SZ;
      g_t[idx].time = ts;
      g_t[idx].mid = mid;
      g_total++;

      // compress deques before they hit array bounds
      if (g_mint >= RING_SZ - 10) { int n = g_mint - g_minh; ArrayCopy(g_minv, g_minv, 0, g_minh, n); ArrayCopy(g_mini, g_mini, 0, g_minh, n); g_mint = n; g_minh = 0; }
      if (g_maxt >= RING_SZ - 10) { int n = g_maxt - g_maxh; ArrayCopy(g_maxv, g_maxv, 0, g_maxh, n); ArrayCopy(g_maxi, g_maxi, 0, g_maxh, n); g_maxt = n; g_maxh = 0; }

      while (g_mint > g_minh && g_minv[g_mint - 1] >= mid) g_mint--;
      g_minv[g_mint] = mid; g_mini[g_mint] = g_total - 1; g_mint++;

      while (g_maxt > g_maxh && g_maxv[g_maxt - 1] <= mid) g_maxt--;
      g_maxv[g_maxt] = mid; g_maxi[g_maxt] = g_total - 1; g_maxt++;

      // slide window: remove ticks older than InpWindowSec
      while (g_total - g_ws >= 2 && ts - g_t[g_ws % RING_SZ].time > InpWindowSec) {
         if (g_minh < g_mint && g_mini[g_minh] == g_ws) g_minh++;
         if (g_maxh < g_maxt && g_maxi[g_maxh] == g_ws) g_maxh++;
         g_ws++;
      }

      if (g_total - 1 <= g_ws) continue;
      if (ts - g_t[g_ws % RING_SZ].time > InpWindowSec) continue;

      // ensure deques have data
      if (g_minh >= g_mint) { g_minv[0]=mid; g_mini[0]=g_total-1; g_minh=0; g_mint=1; }
      if (g_maxh >= g_maxt) { g_maxv[0]=mid; g_maxi[0]=g_total-1; g_maxh=0; g_maxt=1; }

      double wp = g_t[g_ws % RING_SZ].mid;
      double hp = g_maxv[g_maxh] - wp;
      double lp = wp - g_minv[g_minh];

      if (hp < thresh && lp < thresh) continue;
      if (g_ws <= g_last_ws) continue;
      g_last_ws = g_ws;

      int ext_dir;
      if (hp >= thresh && lp >= thresh) ext_dir = (hp >= lp) ? 1 : -1;
      else if (hp >= thresh) ext_dir = 1;
      else ext_dir = -1;
      fade_dir = -ext_dir;

      entry_price = (fade_dir == 1) ? ask : bid;
      sl_price = (fade_dir == 1) ? entry_price - InpStopPips * pip : entry_price + InpStopPips * pip;
      entry_time = ts;
      in_pos = 1;

      // open position — clean state for next detection
      g_total = 0; g_ws = 0; g_last_ws = -1;
      g_minh = 0; g_mint = 0; g_maxh = 0; g_maxt = 0;
   }

   FileClose(h);
   Print("=== FINAL ===");
   Print("Trades: " + IntegerToString(g_trades) + " Wins: " + IntegerToString(g_wins) + " PnL: " + DoubleToString(g_total_pnl, 1) + " pips");
   if (g_trades > 0) Print("WR: " + DoubleToString(100.0 * g_wins / g_trades, 1) + "%");
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason) {}
void OnTick() { ExpertRemove(); }

double OnTester() {
   if (g_trades > 0) return 100.0 * g_wins / g_trades + g_total_pnl * 0.1;
   return 0;
}
