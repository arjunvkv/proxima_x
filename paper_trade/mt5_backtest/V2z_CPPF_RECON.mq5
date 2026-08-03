#property version "1.00"
#property description "V2z CPPF RECON - reconstructed from Jul 25 report evidence"

input double   Z_THRESHOLD       = 2.5;
input double   STOP_A            = 4.0;
input double   TRIG_A            = 1.5;
input double   GAP_A             = 0.08;
input int      MAX_HOLD_BARS     = 54;
input int      ATR_PERIOD        = 20;
input int      Z_WINDOW          = 50;
input double   BASE_LOT          = 1.0;
input double   MAX_DAILY_LOSS    = 1250.0;
input double   MAX_SPREAD_PIPS   = 5.0;
input int      MAGIC_NUMBER      = 202411;
input int      MAX_TRADES_DAY    = 100;
input double   LIMIT_ENTRY_ATR   = 0.0;
input double   ATR_GATE_PCT      = 0.0;
input int      TRADE_START_HOUR  = 0;
input int      TRADE_END_HOUR    = 7;
input double   SPREAD_MULT_MAX   = 0.0;
input double   LOT_SCALE_MIN_Z   = 0.0;
input double   LOT_SCALE_MAX     = 2.0;
input double   MIN_GAP_PIPS      = 0.5;
input double   TAKE_PROFIT_ATR   = 0.0;
input int      TRADE_DIRECTION   = 0;  // 0=both, 1=LONG-only, 2=SHORT-only
input bool     ENABLE_STABILITY_GATE = false;
input double   STABILITY_THRESHOLD  = 0.50;
input double   Z_CUM_MIN            = 5.0;
input double   SPREAD_Z_MAX         = 1.5;
input double   VOL_Z_MAX            = 2.0;
input int      Z_CUM_BARS           = 10;

double g_close_buf[];
int    g_close_count = 0;
double g_atr_buf[];
int    g_atr_head = 0;
int    g_atr_count = 0;
int    g_atr_handle = INVALID_HANDLE;
datetime g_last_bar_time = 0;
int    g_daily_trades = 0;
string g_daily_date = "";
double g_entry_price = 0;
datetime g_entry_time = 0;
double g_best_price = 0;
double g_current_stop = 0;
int    g_bars_held = 0;
int    g_last_bar_number = -1;
int    g_direction = 0;

double g_z_hist[30];
int    g_z_head = 0;
int    g_z_count = 0;
double g_spread_hist[30];
int    g_spread_head = 0;
int    g_spread_count = 0;
double g_vol_hist[20];
int    g_vol_head = 0;
int    g_vol_count = 0;

int OnInit() {
   if(!SymbolInfoInteger(_Symbol, SYMBOL_TRADE_MODE)) {
      Print("Symbol ", _Symbol, " not tradeable");
      return INIT_FAILED;
   }
   g_atr_handle = iATR(_Symbol, PERIOD_M1, ATR_PERIOD);
   if(g_atr_handle == INVALID_HANDLE) {
      Print("Failed to create ATR handle"); return INIT_FAILED;
   }
   ArrayResize(g_close_buf, Z_WINDOW + 2);
   ArrayResize(g_atr_buf, ATR_PERIOD);
   ClearPosition();
    WarmUpBuffers();
    WarmUpZHistory();
    WarmUpSpreadHistory();
    WarmUpVolumeHistory();
    RecoverPosition();
   if(g_direction != 0)
      Print("Recovered dir=", g_direction, " entry=", g_entry_price,
            " sl=", g_current_stop, " held=", g_bars_held);
   Print("V2z_CPPF_RECON on ", _Symbol, " magic=", MAGIC_NUMBER);
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason) {
   if(g_atr_handle != INVALID_HANDLE) IndicatorRelease(g_atr_handle);
}

double g_entry_z = 0;
double g_entry_atr = 0;
double g_entry_spread = 0;

void ClearPosition() {
   g_entry_price = 0; g_entry_time = 0;
   g_best_price = 0; g_current_stop = 0; g_bars_held = 0; g_direction = 0;
   g_entry_z = 0; g_entry_atr = 0; g_entry_spread = 0;
}

void WarmUpBuffers() {
   double closes[];
   int got = CopyClose(_Symbol, PERIOD_M1, 1, Z_WINDOW + 2, closes);
   if(got > 0) {
      int bs = Z_WINDOW + 2;
      int start = (got >= bs) ? (got - bs) : 0;
      int count = (got >= bs) ? bs : got;
      for(int i = 0; i < count; i++)
         g_close_buf[i] = closes[start + i];
      g_close_count = count;
   }
   double highs[], lows[];
   int hg = CopyHigh(_Symbol, PERIOD_M1, 1, ATR_PERIOD, highs);
   int lw = CopyLow(_Symbol, PERIOD_M1, 1, ATR_PERIOD, lows);
   if(hg > 0 && lw > 0) {
      int cnt = (hg < lw) ? hg : lw;
      cnt = (cnt < ATR_PERIOD) ? cnt : ATR_PERIOD;
      for(int i = 0; i < cnt; i++) {
         g_atr_buf[i] = highs[hg - cnt + i] - lows[lw - cnt + i];
         g_atr_head = (i + 1) % ATR_PERIOD;
      }
      g_atr_count = cnt;
   }
}

void WarmUpZHistory() {
   g_z_count = 0; g_z_head = 0;
   int n = (g_close_count >= Z_WINDOW + 2) ? (g_close_count - 1) : 0;
   for(int i = 0; i < n && g_z_count < 30; i++) {
      double rets[51];
      int start = (i < Z_WINDOW) ? 0 : (i - Z_WINDOW);
      int nrets = (i < Z_WINDOW) ? (i + 1) : (Z_WINDOW + 1);
      for(int j = 0; j < nrets; j++)
         rets[j] = g_close_buf[start + j + 1] - g_close_buf[start + j];
      double cur_ret = rets[nrets - 1];
      double sum = 0;
      for(int j = 0; j < nrets - 1; j++) sum += rets[j];
      double mean = sum / (nrets - 1);
      double var = 0;
      for(int j = 0; j < nrets - 1; j++) { double d = rets[j] - mean; var += d * d; }
      var /= (nrets - 2);
      double zval = (var < 1e-14) ? 0 : (cur_ret - mean) / sqrt(var);
      g_z_hist[g_z_head] = zval;
      g_z_head = (g_z_head + 1) % 30;
      g_z_count++;
   }
}

void WarmUpSpreadHistory() {
   g_spread_count = 0; g_spread_head = 0;
   int sp_rates[];
   int got = CopySpread(_Symbol, PERIOD_M1, 1, 30, sp_rates);
   if(got > 0) {
      int n = (got < 30) ? got : 30;
      for(int i = 0; i < n; i++) {
         g_spread_hist[g_spread_head] = (double)sp_rates[got - n + i];
         g_spread_head = (g_spread_head + 1) % 30;
         g_spread_count++;
      }
   }
}

void WarmUpVolumeHistory() {
   g_vol_count = 0; g_vol_head = 0;
   long tv_rates[];
   int got = CopyTickVolume(_Symbol, PERIOD_M1, 1, 20, tv_rates);
   if(got > 0) {
      int n = (got < 20) ? got : 20;
      for(int i = 0; i < n; i++) {
         g_vol_hist[g_vol_head] = (double)tv_rates[got - n + i];
         g_vol_head = (g_vol_head + 1) % 20;
         g_vol_count++;
      }
   }
}

void RecoverPosition() {
   if(!PositionSelect(_Symbol)) return;
   if((int)PositionGetInteger(POSITION_MAGIC) != MAGIC_NUMBER) return;
   g_direction = (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY) ? 1 : -1;
   g_entry_price = PositionGetDouble(POSITION_PRICE_OPEN);
   g_current_stop = PositionGetDouble(POSITION_SL);
   g_entry_time = (datetime)PositionGetInteger(POSITION_TIME);
   g_entry_atr = GetATRValue();
   g_entry_spread = GetSpreadPips();
   g_entry_z = ComputeZScore();
   g_bars_held = (int)((TimeCurrent() - g_entry_time) / 60);
   MqlTick tick;
   if(SymbolInfoTick(_Symbol, tick))
      g_best_price = (g_direction > 0) ? tick.bid : tick.ask;
   UpdateDailyCounters();
}

bool IsNewBar() {
   datetime times[1];
   if(CopyTime(_Symbol, PERIOD_M1, 0, 1, times) != 1) return false;
   if(times[0] == g_last_bar_time) return false;
   int bar = (int)((times[0] - 0) / 60);
   if(bar == g_last_bar_number) return false;
   g_last_bar_number = bar;
   g_last_bar_time = times[0];
   return true;
}

void UpdateDailyCounters() {
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   string today = IntegerToString(dt.year)
      + StringFormat("%02d%02d", dt.mon, dt.day);
   if(today != g_daily_date) {
      g_daily_date = today; g_daily_trades = 0;
   }
}

void UpdateCloseBuffer() {
   double closes[1];
   if(CopyClose(_Symbol, PERIOD_M1, 1, 1, closes) != 1) return;
   int bs = Z_WINDOW + 2;
   for(int i = 0; i < bs - 1; i++)
      g_close_buf[i] = g_close_buf[i + 1];
   g_close_buf[bs - 1] = closes[0];
   if(g_close_count < bs) g_close_count++;
}

void UpdateATRBuffer() {
   double hl[1], ll[1];
   if(CopyHigh(_Symbol, PERIOD_M1, 1, 1, hl) != 1) return;
   if(CopyLow(_Symbol, PERIOD_M1, 1, 1, ll) != 1) return;
   g_atr_buf[g_atr_head] = hl[0] - ll[0];
   g_atr_head = (g_atr_head + 1) % ATR_PERIOD;
   if(g_atr_count < ATR_PERIOD) g_atr_count++;
}

void UpdateZHistory() {
   double zval = ComputeZScore();
   g_z_hist[g_z_head] = zval;
   g_z_head = (g_z_head + 1) % 30;
   if(g_z_count < 30) g_z_count++;
}

void UpdateSpreadHistory() {
   int sp_rates[1];
   if(CopySpread(_Symbol, PERIOD_M1, 1, 1, sp_rates) != 1) return;
   g_spread_hist[g_spread_head] = (double)sp_rates[0];
   g_spread_head = (g_spread_head + 1) % 30;
   if(g_spread_count < 30) g_spread_count++;
}

void UpdateVolumeHistory() {
   long tv_rates[1];
   if(CopyTickVolume(_Symbol, PERIOD_M1, 1, 1, tv_rates) != 1) return;
   g_vol_hist[g_vol_head] = (double)tv_rates[0];
   g_vol_head = (g_vol_head + 1) % 20;
   if(g_vol_count < 20) g_vol_count++;
}

double ComputeCumulativeZ(int bars) {
   if(g_z_count < bars || bars <= 0) return 0;
   double sum = 0;
   for(int i = 0; i < bars; i++) {
      int idx = (g_z_head - 1 - i + 30) % 30;
      sum += g_z_hist[idx];
   }
   return sum;
}

double ComputeSpreadZ() {
   if(g_spread_count < 30) return 0;
   double sum = 0;
   for(int i = 0; i < 30; i++) sum += g_spread_hist[i];
   double mean = sum / 30.0;
   double var = 0;
   for(int i = 0; i < 30; i++) { double d = g_spread_hist[i] - mean; var += d * d; }
   double std = sqrt(var / 29.0);
   if(std < 0.01) return 0;
   long live_spread = SymbolInfoInteger(_Symbol, SYMBOL_SPREAD);
   double pip = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   double live_sp = (double)live_spread * pip / _Point;
   return (live_sp - mean) / std;
}

double ComputeVolumeZ() {
   if(g_vol_count < 20) return 0;
   double sum = 0;
   for(int i = 0; i < 20; i++) sum += g_vol_hist[i];
   double mean = sum / 20.0;
   double var = 0;
   for(int i = 0; i < 20; i++) { double d = g_vol_hist[i] - mean; var += d * d; }
   double std = sqrt(var / 19.0);
   if(std < 0.01) return 0;
   double latest_vol = g_vol_hist[(g_vol_head - 1 + 20) % 20];
   return (latest_vol - mean) / std;
}

double ComputeStabilityIndex() {
   double z_cur = g_z_hist[(g_z_head - 1 + 30) % 30];
   double z_cum = ComputeCumulativeZ(Z_CUM_BARS);
   double spd_z = ComputeSpreadZ();
   double vol_z = ComputeVolumeZ();
   double z_norm = fmin(fabs(z_cur) / 8.0, 1.0);
   double z_cum_norm = fmin(fabs(z_cum) / 15.0, 1.0);
   double spd_norm = fmax(1.0 - spd_z / 3.0, 0.0);
   double vol_norm = fmax(1.0 - vol_z / 3.0, 0.0);
   return 0.3 * z_norm + 0.3 * z_cum_norm + 0.2 * spd_norm + 0.2 * vol_norm;
}

double ComputeZScore() {
   if(g_close_count < Z_WINDOW + 2) return 0;
   double rets[51];
   for(int i = 0; i <= Z_WINDOW; i++)
      rets[i] = g_close_buf[i + 1] - g_close_buf[i];
   double cur_ret = rets[Z_WINDOW];
   double sum = 0;
   for(int i = 0; i < Z_WINDOW; i++) sum += rets[i];
   double mean = sum / Z_WINDOW;
   double var = 0;
   for(int i = 0; i < Z_WINDOW; i++) {
      double d = rets[i] - mean; var += d * d;
   }
   var /= (Z_WINDOW - 1);
   if(var < 1e-14) return 0;
   return (cur_ret - mean) / sqrt(var);
}

double GetATRValue() {
   if(g_atr_count < ATR_PERIOD) return 0;
   double s = 0;
   for(int i = 0; i < ATR_PERIOD; i++) s += g_atr_buf[i];
   return s / ATR_PERIOD;
}

double GetSpreadPips() {
   long spread = SymbolInfoInteger(_Symbol, SYMBOL_SPREAD);
   double pip = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   return (double)spread * pip / _Point;
}

double GetRealizedPL() {
   if(!PositionSelect(_Symbol)) return 0;
   return PositionGetDouble(POSITION_PROFIT);
}

bool OpenPosition(int direction, double atr_v) {
   double s = STOP_A * atr_v;
   MqlTick tick;
   if(!SymbolInfoTick(_Symbol, tick)) return false;
   double price = (direction > 0) ? tick.ask : tick.bid;
   double sl = (direction > 0) ? (price - s) : (price + s);
   int order_type = (direction > 0) ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;

   MqlTradeRequest req = {};
   MqlTradeResult res = {};
   req.action = TRADE_ACTION_DEAL;
   req.symbol = _Symbol;
   req.volume = BASE_LOT;
   req.type = order_type;
   req.price = price;
   req.deviation = 10;
   req.magic = MAGIC_NUMBER;
   req.comment = "V2z_RECON";

   if(!OrderSend(req, res) || res.retcode != TRADE_RETCODE_DONE) {
      Print("OrderSend failed: ", res.retcode, " ", GetLastError());
      return false;
   }

   if(direction > 0) { g_entry_price = price; }
   else { g_entry_price = price; }
   g_direction = direction;
   g_entry_time = TimeCurrent();
   g_best_price = g_entry_price;
   g_current_stop = sl;
   g_bars_held = 0;
   g_daily_trades++;
   g_entry_z = ComputeZScore();
   g_entry_atr = atr_v;
   g_entry_spread = GetSpreadPips();
   Print("OPEN ", _Symbol, " dir=", direction, " entry=", g_entry_price,
         " sl=", sl, " atr=", atr_v, " z=", g_entry_z, " sprd=", g_entry_spread);
   return true;
}

bool ClosePosition(string reason) {
   if(g_direction == 0) return false;

   double pl = GetRealizedPL();

   MqlTick tick;
   if(!SymbolInfoTick(_Symbol, tick)) return false;
   int order_type = (g_direction > 0) ? ORDER_TYPE_SELL : ORDER_TYPE_BUY;
   double price = (g_direction > 0) ? tick.bid : tick.ask;

   MqlTradeRequest req = {};
   MqlTradeResult res = {};
   req.action = TRADE_ACTION_DEAL;
   req.symbol = _Symbol;
   req.volume = BASE_LOT;
   req.type = order_type;
   req.price = price;
   req.deviation = 10;
   req.magic = MAGIC_NUMBER;
   req.comment = "V2z_close";

   PositionSelect(_Symbol);
   ulong ticket = (ulong)PositionGetInteger(POSITION_TICKET);
   if(ticket == 0) {
      Print("No position for ", _Symbol); ClearPosition(); return false;
   }
   req.position = ticket;

   if(!OrderSend(req, res) || res.retcode != TRADE_RETCODE_DONE) {
      Print("Close failed: ", GetLastError()); return false;
   }

   Print("CLOSE ", _Symbol, " rsn=", reason, " pnl=", pl,
         " z=", g_entry_z, " atr=", g_entry_atr, " sprd=", g_entry_spread,
         " held=", g_bars_held, " bars");
   ClearPosition();
   return true;
}

void ManagePosition() {
   if(g_direction == 0) return;
   if(!PositionSelect(_Symbol)) {
      double cs = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_CONTRACT_SIZE);
      double pl = (g_direction > 0) ?
         (g_current_stop - g_entry_price) * BASE_LOT * cs :
         (g_entry_price - g_current_stop) * BASE_LOT * cs;
      Print("LOST ", _Symbol, " dir=", g_direction, " entry=", g_entry_price,
            " stop=", g_current_stop, " pnl=", pl,
            " atr=", g_entry_atr, " z=", g_entry_z, " sprd=", g_entry_spread,
            " held=", g_bars_held, " bars");
      ClearPosition(); return;
   }
   MqlTick tick;
   if(!SymbolInfoTick(_Symbol, tick)) return;

   double atr_v = GetATRValue();
   if(atr_v <= 0) return;
   double tg = TRIG_A * atr_v;
   double gp = GAP_A * atr_v;

   double min_gap = MIN_GAP_PIPS * _Point * 10;
   if(gp < min_gap) gp = min_gap;

   if(g_direction > 0) {
      if(tick.bid > g_best_price) {
         g_best_price = tick.bid;
         if(g_best_price - g_entry_price > tg) {
            double ns = g_best_price - gp;
            if(ns > g_current_stop) { g_current_stop = ns; ModifyStopLoss(ns); }
         }
      }
   } else {
      if(tick.ask < g_best_price) {
         g_best_price = tick.ask;
         if(g_entry_price - g_best_price > tg) {
            double ns = g_best_price + gp;
            if(ns < g_current_stop) { g_current_stop = ns; ModifyStopLoss(ns); }
         }
      }
   }

   if(g_direction > 0) {
      if(tick.bid <= g_current_stop) { ClosePosition("stop"); return; }
      if(TAKE_PROFIT_ATR > 0.0 && tick.bid >= g_entry_price + TAKE_PROFIT_ATR * atr_v) { ClosePosition("tp"); return; }
   } else {
      if(tick.ask >= g_current_stop) { ClosePosition("stop"); return; }
      if(TAKE_PROFIT_ATR > 0.0 && tick.ask <= g_entry_price - TAKE_PROFIT_ATR * atr_v) { ClosePosition("tp"); return; }
   }

   if(g_bars_held >= MAX_HOLD_BARS) ClosePosition("expiry");
}

bool ModifyStopLoss(double sl_price) {
   PositionSelect(_Symbol);
   ulong ticket = (ulong)PositionGetInteger(POSITION_TICKET);
   if(ticket == 0) return false;
   MqlTradeRequest req = {};
   MqlTradeResult res = {};
   req.action = TRADE_ACTION_SLTP;
   req.symbol = _Symbol;
   req.position = ticket;
   req.sl = sl_price;
   req.magic = MAGIC_NUMBER;
   return OrderSend(req, res) && res.retcode == TRADE_RETCODE_DONE;
}

void CheckEntry() {
   if(g_direction != 0) return;
   MqlDateTime dt; TimeToStruct(TimeCurrent(), dt);
   int hour = dt.hour;
   if(TRADE_START_HOUR < TRADE_END_HOUR) { if(hour < TRADE_START_HOUR || hour >= TRADE_END_HOUR) return; }
   else { if(hour < TRADE_START_HOUR && hour >= TRADE_END_HOUR) return; }

   double spread = GetSpreadPips();
   if(spread > MAX_SPREAD_PIPS) return;

   UpdateDailyCounters();
   if(g_daily_trades >= MAX_TRADES_DAY) return;

   double z = ComputeZScore();
   double av = GetATRValue();
   if(av <= 0) return;
   if(LIMIT_ENTRY_ATR > 0 && av < LIMIT_ENTRY_ATR) return;

   if(MathAbs(z) >= Z_THRESHOLD) {
      int dir = (z > 0) ? -1 : 1;
      if(TRADE_DIRECTION == 1 && dir == -1) return;
      if(TRADE_DIRECTION == 2 && dir == 1) return;

      if(ENABLE_STABILITY_GATE) {
         if(g_z_count < Z_CUM_BARS) return;
         double z_cum = ComputeCumulativeZ(Z_CUM_BARS);
         double spd_z = ComputeSpreadZ();
         double vol_z = ComputeVolumeZ();
         double stability = ComputeStabilityIndex();
         if(Z_CUM_MIN > 0 && fabs(z_cum) < Z_CUM_MIN) return;
         if(SPREAD_Z_MAX > 0 && spd_z > SPREAD_Z_MAX) return;
         if(VOL_Z_MAX > 0 && vol_z > VOL_Z_MAX) return;
         if(STABILITY_THRESHOLD > 0 && stability < STABILITY_THRESHOLD) return;
         Print("STABLE entry z=", z, " z_cum=", z_cum, " spd_z=", spd_z, " vol_z=", vol_z, " stab=", stability);
      }

      OpenPosition(dir, av);
   }
}

void OnTick() {
   if(g_direction != 0) ManagePosition();
   if(IsNewBar()) {
      UpdateCloseBuffer();
      UpdateATRBuffer();
      UpdateZHistory();
      UpdateSpreadHistory();
      UpdateVolumeHistory();
      if(g_direction != 0) g_bars_held++;
      if(g_direction == 0) CheckEntry();
   }
}
