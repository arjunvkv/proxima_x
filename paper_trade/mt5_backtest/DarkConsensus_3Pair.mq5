#property version "1.00"
#property description "Dark Consensus - 3-pair P95 magnitude"

input double   MAG_THRESHOLD   = 0.00018741;
input int      HOLD_BARS       = 3;
input double   BASE_LOT        = 0.3;
input double   SLIPPAGE_PIPS   = 1.0;
input int      MAGIC_NUMBER    = 95002;
input string   CSV_DIR         = "";

string TRADE_PAIRS[3] = {"EURJPY", "EURUSD", "GBPJPY"};
string CSV_FILES[3]   = {"FN_EURJPY.csv", "FN_EURUSD.csv", "FN_GBPJPY.csv"};

// In-memory copy of the 3-pair CSV data
double g_opens_eurjpy[], g_opens_eurusd[], g_opens_gbpjpy[];
double g_closes_eurjpy[], g_closes_eurusd[], g_closes_gbpjpy[];
double g_highs_eurjpy[], g_highs_eurusd[], g_highs_gbpjpy[];
double g_lows_eurjpy[], g_lows_eurusd[], g_lows_gbpjpy[];
datetime g_times[];
int g_nbars = 0;

datetime g_last_bar = 0;
datetime g_entry_time = 0;
int g_entry_idx = -1;
int g_trade_pair = -1;
int g_trade_dir = 0;
int g_hold_count = 0;
int g_trade_count = 0;

bool LoadCSVData() {
   // First pass: count bars in the shortest file
   int bar_count = 0;
   for(int i = 0; i < 3; i++) {
      string path = CSV_DIR + CSV_FILES[i];
      int fh = FileOpen(path, FILE_READ | FILE_CSV | FILE_ANSI | FILE_COMMON, ",");
      if(fh == INVALID_HANDLE) return false;
      int cnt = 0;
      while(!FileIsEnding(fh)) {
         FileReadString(fh); FileReadString(fh); FileReadString(fh);
         FileReadString(fh); FileReadString(fh); FileReadString(fh);
         FileReadString(fh); cnt++;
      }
      FileClose(fh);
      if(i == 0 || cnt < bar_count) bar_count = cnt;
      Print(CSV_FILES[i] + ": " + IntegerToString(cnt) + " bars");
   }

   // Allocate global arrays
   g_nbars = bar_count;
   ArrayResize(g_times, bar_count);
   ArrayResize(g_opens_eurjpy, bar_count);   ArrayResize(g_closes_eurjpy, bar_count);
   ArrayResize(g_highs_eurjpy, bar_count);   ArrayResize(g_lows_eurjpy, bar_count);
   ArrayResize(g_opens_eurusd, bar_count);   ArrayResize(g_closes_eurusd, bar_count);
   ArrayResize(g_highs_eurusd, bar_count);   ArrayResize(g_lows_eurusd, bar_count);
   ArrayResize(g_opens_gbpjpy, bar_count);   ArrayResize(g_closes_gbpjpy, bar_count);
   ArrayResize(g_highs_gbpjpy, bar_count);   ArrayResize(g_lows_gbpjpy, bar_count);

   // Second pass: fill arrays directly (no intermediate MqlRates)
   for(int p = 0; p < 3; p++) {
      string path = CSV_DIR + CSV_FILES[p];
      int fh = FileOpen(path, FILE_READ | FILE_CSV | FILE_ANSI | FILE_COMMON, ",");
      if(fh == INVALID_HANDLE) return false;
      for(int i = 0; i < bar_count && !FileIsEnding(fh); i++) {
          string ds = FileReadString(fh);
          string ts = FileReadString(fh);
          string os = FileReadString(fh);
          string hs = FileReadString(fh);
          string ls = FileReadString(fh);
          string cs = FileReadString(fh);
          FileReadString(fh); // skip volume field
          string dt_str = ds + " " + ts;
          StringReplace(dt_str, "\"", ""); // strip quotes from quoted CSV field
          datetime t = StringToTime(dt_str) + TimeGMTOffset();
         double o = StringToDouble(os), h = StringToDouble(hs);
         double l = StringToDouble(ls), c = StringToDouble(cs);
         if(p == 0) {
            g_times[i] = t;
            g_opens_eurjpy[i] = o; g_highs_eurjpy[i] = h;
            g_lows_eurjpy[i]  = l; g_closes_eurjpy[i]  = c;
         } else if(p == 1) {
            g_opens_eurusd[i] = o; g_highs_eurusd[i] = h;
            g_lows_eurusd[i]  = l; g_closes_eurusd[i]  = c;
         } else {
            g_opens_gbpjpy[i] = o; g_highs_gbpjpy[i] = h;
            g_lows_gbpjpy[i]  = l; g_closes_gbpjpy[i]  = c;
         }
      }
      FileClose(fh);
   }
   Print("Aligned: " + IntegerToString(bar_count) + " bars");
   return true;
}

int FindBar(datetime t) {
   if(g_nbars == 0) return -1;
   int lo = 0, hi = g_nbars - 1;
   if(t <= g_times[lo]) return lo;
   if(t >= g_times[hi]) return hi;
   while(lo < hi - 1) {
      int mid = (lo + hi) / 2;
      if(g_times[mid] <= t) lo = mid;
      else hi = mid;
   }
   return lo;
}

void CloseTrade() {
   if(g_trade_pair < 0) return;
   string sym = TRADE_PAIRS[g_trade_pair];
   if(!PositionSelect(sym)) { g_entry_time = 0; g_entry_idx = -1; return; }
   ulong ticket = PositionGetInteger(POSITION_TICKET);
   double vol = PositionGetDouble(POSITION_VOLUME);
   long pt = PositionGetInteger(POSITION_TYPE);
   MqlTradeRequest req = {};
   req.action = TRADE_ACTION_DEAL;
   req.position = ticket;
   req.symbol = sym;
   req.volume = vol;
   req.type = (pt == POSITION_TYPE_BUY) ? ORDER_TYPE_SELL : ORDER_TYPE_BUY;
   req.price = (pt == POSITION_TYPE_BUY) ? SymbolInfoDouble(sym, SYMBOL_BID)
                                          : SymbolInfoDouble(sym, SYMBOL_ASK);
   req.deviation = 5;
   req.magic = MAGIC_NUMBER;
   MqlTradeResult res;
   OrderSend(req, res);
   g_entry_time = 0; g_entry_idx = -1; g_trade_pair = -1; g_trade_dir = 0;
   g_hold_count = 0;
}

int OnInit() {
   if(!LoadCSVData()) return INIT_FAILED;
   if(g_nbars < 1440) { Print("Not enough bars: " + IntegerToString(g_nbars)); return INIT_FAILED; }

   // Log first/last bar timestamp and TimeGMTOffset
   int toff = TimeGMTOffset();
   Print("Init: " + IntegerToString(g_nbars) + " bars, toff=" + IntegerToString(toff)
         + " first=" + IntegerToString(g_times[0]) + " last=" + IntegerToString(g_times[g_nbars-1]));
   // Enable all 3 trade symbols in Market Watch (for multi-symbol tester)
   for(int i = 0; i < 3; i++) {
      SymbolSelect(TRADE_PAIRS[i], true);
      iClose(TRADE_PAIRS[i], PERIOD_M1, 0);
   }
   return INIT_SUCCEEDED;
}

void OnTick() {
   datetime bt[1];
   if(CopyTime(_Symbol, PERIOD_M1, 0, 1, bt) != 1) return;
   bool new_bar = (bt[0] != g_last_bar);
   g_last_bar = bt[0];

   // Handle open position hold (measured in CSV-bar indices)
   if(g_entry_idx >= 0 && g_trade_pair >= 0) {
      if(new_bar) {
         int cur_idx = FindBar(bt[0]);
         if(cur_idx >= g_entry_idx + HOLD_BARS) CloseTrade();
      }
      return;
   }
   if(!new_bar) return;

   // Find our position in the loaded data
   int idx = FindBar(bt[0]);
   if(idx < 1440 || idx >= g_nbars - HOLD_BARS - 1) {
      static bool dbg_skip = true;
      if(dbg_skip) {
         string s = idx>=0 ? (g_times[idx]==bt[0]?"match":"nomatch") : "noidx";
         Print("SKIP idx=" + IntegerToString(idx) + " bt=" + IntegerToString(bt[0])
               + " g_t=" + (idx>=0?IntegerToString(g_times[idx]):"NA") + " " + s);
         dbg_skip = false;
      }
      return;
   }
   // Only act on CSV-aligned timestamps (matches Python's per-bar evaluation)
   if(g_times[idx] != bt[0]) {
      static bool dbg_nm = true;
      if(dbg_nm) {
         Print("NOMATCH idx=" + IntegerToString(idx) + " bt=" + IntegerToString(bt[0])
               + " g_t=" + IntegerToString(g_times[idx])
               + " diff=" + IntegerToString(bt[0] - g_times[idx]));
         dbg_nm = false;
      }
      return;
   }
   // Compute 1-minute log returns from the JUST-COMPLETED bar (idx-1)
   // Equivalent to Python's close[t]/close[t-1] evaluated at bar t close,
   // then entering at bar t+1 open — no lookahead.
   double prev[3], curr[3];
   double rets[3], mags[3];
   double total_mag = 0;
   bool all_pos = true, all_neg = true;
   prev[0]=g_closes_eurjpy[idx-2]; curr[0]=g_closes_eurjpy[idx-1];
   prev[1]=g_closes_eurusd[idx-2]; curr[1]=g_closes_eurusd[idx-1];
   prev[2]=g_closes_gbpjpy[idx-2]; curr[2]=g_closes_gbpjpy[idx-1];
   for(int i = 0; i < 3; i++) {
      rets[i] = MathLog(curr[i] / prev[i]);
      mags[i] = MathAbs(rets[i]);
      total_mag += mags[i];
      if(rets[i] > 0) all_neg = false;
      else all_pos = false;
   }

   // 3-pair consensus + magnitude + hour filter (matches Python)
   if(!all_pos && !all_neg) return;

   // Compute UTC hour directly from Unix timestamp (matches Python research)
   int utc_hour = (int)((g_times[idx] % 86400) / 3600);
   if(utc_hour < 7 || utc_hour > 21) return;

   double avg_mag = total_mag / 3.0;
   if(avg_mag <= MAG_THRESHOLD) return;

   // Best pair: largest |return|
   int best = (mags[1] > mags[0]) ? 1 : 0;
   if(mags[2] > mags[best]) best = 2;

   int dir = all_pos ? 1 : -1;
   string sym = TRADE_PAIRS[best];
   ENUM_ORDER_TYPE otype = (dir == 1) ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;

   double price = (otype == ORDER_TYPE_BUY) ? SymbolInfoDouble(sym, SYMBOL_ASK)
                                             : SymbolInfoDouble(sym, SYMBOL_BID);
   MqlTradeRequest req = {};
   req.action = TRADE_ACTION_DEAL;
   req.symbol = sym;
   req.volume = BASE_LOT;
   req.type = otype;
   req.price = price;
   req.deviation = 5;
   req.magic = MAGIC_NUMBER;
   MqlTradeResult res;
   if(OrderSend(req, res)) {
      g_entry_time = g_times[idx];
      g_entry_idx = idx;
      g_trade_pair = best;
      g_trade_dir = dir;
      g_hold_count = 0;
      g_trade_count++;
      Print("TRADE " + sym + " " + (dir>0?"BUY":"SELL") + " idx=" + IntegerToString(idx)
            + " mag=" + DoubleToString(avg_mag,8) + " n=" + IntegerToString(g_trade_count));
   }
}

double OnTester() {
   return g_trade_count;
}
