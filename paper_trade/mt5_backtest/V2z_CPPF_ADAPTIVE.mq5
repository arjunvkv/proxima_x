#property version "1.00"
#property description "V2z CPPF ADAPTIVE - ATR-regime-adaptive z-threshold, trailing, spread/ATR filter"

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
input int      MAGIC_NUMBER      = 202412;
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
input int      ATR_LONG_PERIOD   = 50;
input double   SPREAD_ATR_LIMIT  = 0.15;
input int      ADAPT_MODE        = 1; // 0=off, 1=z+trailing, 2=+spread/ATR filter

double g_close_buf[];
int    g_close_count = 0;
double g_atr_buf[];
int    g_atr_head = 0;
int    g_atr_count = 0;
double g_atr_long = 0;
int    g_atr_long_count = 0;
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

double g_entry_z = 0;
double g_entry_atr = 0;
double g_entry_spread = 0;

double g_used_z_threshold = 0;
double g_used_trig_a = 0;
double g_used_gap_a = 0;

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
   Print("V2z_CPPF_ADAPTIVE on ", _Symbol, " magic=", MAGIC_NUMBER);
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason) {
   if(g_atr_handle != INVALID_HANDLE) IndicatorRelease(g_atr_handle);
}

void ClearPosition() {
   g_entry_price = 0; g_entry_time = 0;
   g_best_price = 0; g_current_stop = 0; g_bars_held = 0; g_direction = 0;
   g_entry_z = 0; g_entry_atr = 0; g_entry_spread = 0;
   g_used_z_threshold = 0; g_used_trig_a = 0; g_used_gap_a = 0;
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

void UpdateATRLong(double atr_val) {
   if(atr_val <= 0) return;
   if(g_atr_long_count == 0) {
      g_atr_long = atr_val;
      g_atr_long_count = 1;
   } else {
      double alpha = 2.0 / (ATR_LONG_PERIOD + 1);
      g_atr_long = g_atr_long + alpha * (atr_val - g_atr_long);
      if(g_atr_long_count < ATR_LONG_PERIOD) g_atr_long_count++;
   }
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

double GetSpreadATRRatio(double atr_val) {
   if(atr_val <= 0) return 999.0;
   double sprd = GetSpreadPips();
   double spread_price = sprd * _Point;
   return spread_price / atr_val;
}

void ComputeAdaptiveParams(double atr_val, double &z_out, double &trig_out, double &gap_out) {
   z_out = Z_THRESHOLD;
   trig_out = TRIG_A;
   gap_out = GAP_A;
   if(ADAPT_MODE == 0) return;
   if(atr_val <= 0 || g_atr_long <= 0) return;
   double ratio = g_atr_long / atr_val;
   if(ratio < 0.5) ratio = 0.5;
   if(ratio > 2.0) ratio = 2.0;
   double sqrt_r = MathSqrt(ratio);
   z_out = Z_THRESHOLD * sqrt_r;
   trig_out = TRIG_A * sqrt_r;
   gap_out = GAP_A * sqrt_r;
}

double GetRealizedPL() {
   if(!PositionSelect(_Symbol)) return 0;
   return PositionGetDouble(POSITION_PROFIT);
}

bool OpenPosition(int direction, double atr_val) {
   if(ADAPT_MODE >= 2) {
      double sar = GetSpreadATRRatio(atr_val);
      if(sar > SPREAD_ATR_LIMIT) return false;
   }
   double z_used, trig_used, gap_used;
   ComputeAdaptiveParams(atr_val, z_used, trig_used, gap_used);
   double s = STOP_A * atr_val;
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
   req.comment = "V2z_ADAPT";

   if(!OrderSend(req, res) || res.retcode != TRADE_RETCODE_DONE) {
      Print("OrderSend failed: ", res.retcode, " ", GetLastError());
      return false;
   }

   g_entry_price = price;
   g_direction = direction;
   g_entry_time = TimeCurrent();
   g_best_price = g_entry_price;
   g_current_stop = sl;
   g_bars_held = 0;
   g_daily_trades++;
   g_entry_z = ComputeZScore();
   g_entry_atr = atr_val;
   g_entry_spread = GetSpreadPips();
   g_used_z_threshold = z_used;
   g_used_trig_a = trig_used;
   g_used_gap_a = gap_used;
   Print("OPEN ", _Symbol, " dir=", direction, " entry=", g_entry_price,
         " sl=", sl, " atr=", atr_val, " z=", g_entry_z,
         " z_thr=", z_used, " trig=", trig_used, " gap=", gap_used,
         " sprd=", g_entry_spread);
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
   double atr_v = GetATRValue();
   if(atr_v <= 0) return;
   double z_used, trig_used, gap_used;
   ComputeAdaptiveParams(atr_v, z_used, trig_used, gap_used);

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

   double tg = trig_used * atr_v;
   double gp = gap_used * atr_v;

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

   if(g_direction > 0) { if(tick.bid <= g_current_stop) { ClosePosition("stop"); return; } }
   else                { if(tick.ask >= g_current_stop) { ClosePosition("stop"); return; } }

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
   if(GetSpreadPips() > MAX_SPREAD_PIPS) return;
   UpdateDailyCounters();
   if(g_daily_trades >= MAX_TRADES_DAY) return;

   double z = ComputeZScore();
   double av = GetATRValue();
   if(av <= 0) return;

   double z_used, trig_used, gap_used;
   ComputeAdaptiveParams(av, z_used, trig_used, gap_used);

   if(MathAbs(z) >= z_used) {
      int dir = (z > 0) ? -1 : 1;
      OpenPosition(dir, av);
   }
}

void OnTick() {
   if(g_direction != 0) ManagePosition();
   if(IsNewBar()) {
      UpdateCloseBuffer();
      UpdateATRBuffer();
      double atr_val = GetATRValue();
      if(atr_val > 0) UpdateATRLong(atr_val);
      if(g_direction != 0) g_bars_held++;
      if(g_direction == 0) CheckEntry();
   }
}
