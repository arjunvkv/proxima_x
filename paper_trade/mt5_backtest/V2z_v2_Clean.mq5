#property version "1.01"
#property description "V2z CPPF - Fade z-score extremes (vol-adaptive + trend filter)"

input double   Z_THRESHOLD        = 3.5;
input double   STOP_A             = 3.0;
input double   TRIG_A             = 1.0;
input double   GAP_A              = 0.05;
input int      MAX_HOLD_SECONDS   = 3240;
input int      ATR_PERIOD         = 20;
input int      Z_WINDOW           = 50;
input double   BASE_LOT           = 0.75;
input double   MAX_DAILY_LOSS     = 1250.0;
input double   MAX_SPREAD_PIPS    = 5.0;
input int      MAGIC_NUMBER       = 202411;
input int      MAX_TRADES_DAY     = 100;
input double   SLIPPAGE_PIPS      = 1.0;
input double   COMMISSION_PER_LOT = 5.0;
input int      TRADE_START_HOUR   = 0;
input int      TRADE_END_HOUR     = 7;

input bool     ENABLE_TREND_FILTER      = false;
input int      TREND_EMA_PERIOD         = 50;
input bool     ENABLE_VOL_ADAPTIVE_Z    = false;
input double   VOL_REF_PCT              = 10.0;

input bool     ENABLE_MICRO_FILTER      = false;
input double   MICRO_MIN_BODY_PIPS      = 4.0;
input double   MICRO_MIN_SMOOTHNESS     = 0.85;
input double   MICRO_MIN_TICK_VOL       = 0;

input bool     ENABLE_VOL_REGIME_FILTER = false;
input double   VOL_REGIME_ATR_MIN_PIPS  = 2.0;

double   g_close_buf[];
int      g_close_count = 0;
double   g_atr_buf[];
int      g_atr_head = 0;
int      g_atr_count = 0;
int      g_atr_handle = INVALID_HANDLE;
datetime g_last_bar_time = 0;
int      g_daily_trades = 0;
string   g_daily_date = "";
double   g_entry_price = 0;
datetime g_entry_time = 0;
double   g_best_price = 0;
double   g_current_stop = 0;
int      g_direction = 0;

double   g_ema_value = 0;
double   g_ema_prev  = 0;
bool     g_ema_ready = false;

double   g_vol_var  = 0;
int      g_vol_bars = 0;

int      g_log_file = INVALID_HANDLE;
string   g_log_header_written = "";

double SlipPrice() {
   return SLIPPAGE_PIPS * _Point * 10.0;
}

void InitTrendEMA() {
   double alpha = 2.0 / (TREND_EMA_PERIOD + 1);
   if(g_close_count > 0) {
      g_ema_value = g_close_buf[0];
      for(int i = 0; i < g_close_count; i++) {
         g_ema_prev = g_ema_value;
         g_ema_value = alpha * g_close_buf[i] + (1.0 - alpha) * g_ema_value;
      }
      g_ema_ready = true;
   }
}

void UpdateTrendEMA() {
   if(!g_ema_ready) {
      InitTrendEMA();
      return;
   }
   int last = Z_WINDOW + 1;
   if(g_close_count <= last) return;
   double alpha = 2.0 / (TREND_EMA_PERIOD + 1);
   g_ema_prev = g_ema_value;
   g_ema_value = alpha * g_close_buf[last] + (1.0 - alpha) * g_ema_value;
}

void UpdateVolEWMA() {
   if(g_close_count < 3) return;
   int last = Z_WINDOW + 1;
   int prev = Z_WINDOW;
   double ret = (g_close_buf[last] - g_close_buf[prev]) / g_close_buf[prev];
   double lam = 0.94;
   if(g_vol_bars == 0) {
      g_vol_var = ret * ret;
      g_vol_bars = 1;
   } else if(g_vol_bars < 20) {
      g_vol_var = (g_vol_var * g_vol_bars + ret * ret) / (g_vol_bars + 1);
      g_vol_bars++;
   } else {
      g_vol_var = lam * g_vol_var + (1.0 - lam) * ret * ret;
      g_vol_bars++;
   }
}

double GetEffectiveZThreshold() {
   if(!ENABLE_VOL_ADAPTIVE_Z || g_vol_bars < 20) return Z_THRESHOLD;
   double ann_vol_pct = MathSqrt(g_vol_var * 24.0 * 252.0) * 100.0;
   if(ann_vol_pct < 0.1) return Z_THRESHOLD;
    double adapted = Z_THRESHOLD * (ann_vol_pct / VOL_REF_PCT);
   return MathMax(2.0, MathMin(15.0, adapted));
}

bool IsTrendFilterAllowed(int signal_dir) {
   if(!ENABLE_TREND_FILTER || !g_ema_ready) return true;
   if(StringFind(_Symbol, "NZD") == -1) return true;
   bool uptrend = (g_ema_value > g_ema_prev);
   if(uptrend && signal_dir > 0) return true;
   if(!uptrend && signal_dir < 0) return true;
   return false;
}

int OnInit() {
   if(!SymbolInfoInteger(_Symbol, SYMBOL_TRADE_MODE)) {
      Print("Symbol ", _Symbol, " not tradeable");
      return INIT_FAILED;
   }
   g_atr_handle = iATR(_Symbol, _Period, ATR_PERIOD);
   if(g_atr_handle == INVALID_HANDLE) {
      Print("Failed to create ATR handle"); return INIT_FAILED;
   }
   ArrayResize(g_close_buf, Z_WINDOW + 2);
   ArrayResize(g_atr_buf, ATR_PERIOD);
   ClearPosition();
   Print("V2z_CPPF_v2 on ", _Symbol, " magic=", MAGIC_NUMBER,
         " trend=", ENABLE_TREND_FILTER, " vol_adapt=", ENABLE_VOL_ADAPTIVE_Z);
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason) {
   if(g_atr_handle != INVALID_HANDLE) IndicatorRelease(g_atr_handle);
   if(g_log_file != INVALID_HANDLE) FileClose(g_log_file);
}

void ClearPosition() {
   g_entry_price = 0; g_entry_time = 0;
   g_best_price = 0; g_current_stop = 0; g_direction = 0;
}

bool IsNewBar() {
   datetime times[1];
   if(CopyTime(_Symbol, _Period, 0, 1, times) != 1) return false;
   if(times[0] == g_last_bar_time) return false;
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
   if(CopyClose(_Symbol, _Period, 1, 1, closes) != 1) return;
   int bs = Z_WINDOW + 2;
   for(int i = 0; i < bs - 1; i++)
      g_close_buf[i] = g_close_buf[i + 1];
   g_close_buf[bs - 1] = closes[0];
   if(g_close_count < bs) g_close_count++;
}

void UpdateATRBuffer() {
   double hl[1], ll[1];
   if(CopyHigh(_Symbol, _Period, 1, 1, hl) != 1) return;
   if(CopyLow(_Symbol, _Period, 1, 1, ll) != 1) return;
   double bar_range = hl[0] - ll[0];
   g_atr_buf[g_atr_head] = bar_range;
   g_atr_head = (g_atr_head + 1) % ATR_PERIOD;
   if(g_atr_count < ATR_PERIOD) g_atr_count++;
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

bool OpenPosition(int direction, double atr_v, double z_val, double av) {
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
   req.sl = sl;
   req.deviation = (ulong)(SLIPPAGE_PIPS * 10);
   req.magic = MAGIC_NUMBER;
   req.comment = "V2z_v2";

   if(!OrderSend(req, res) || res.retcode != TRADE_RETCODE_DONE) {
      Print("OrderSend failed: ", res.retcode, " ", GetLastError());
      return false;
   }
   g_direction = direction;
   g_entry_price = res.price;
   double slip = SlipPrice();
   if(g_direction > 0) g_entry_price += slip;
   else g_entry_price -= slip;
   g_entry_time = TimeCurrent();
   g_best_price = g_entry_price;
   g_current_stop = sl;
   g_daily_trades++;
   Print("OPEN ", _Symbol, " dir=", direction, " entry=", g_entry_price,
          " sl=", sl, " atr=", atr_v, " slip=", slip);
   LogEntry(direction, g_entry_price, sl, z_val, av);
   return true;
}

double g_log_entry_z = 0, g_log_entry_atr = 0, g_log_entry_tv = 0, g_log_entry_sp = 0;
double g_log_entry_open = 0, g_log_entry_high = 0, g_log_entry_low = 0, g_log_entry_close = 0;

void LogEntry(int direction, double entry_price, double sl, double z_val, double av) {
   MqlTick tick;
   SymbolInfoTick(_Symbol, tick);
   g_log_entry_z = z_val;
   g_log_entry_atr = av;
   // Read the COMPLETED bar (index 1) which triggered the trade
   double o_buf[1], h_buf[1], l_buf[1], c_buf[1];
   if(CopyOpen(_Symbol, _Period, 1, 1, o_buf) == 1) g_log_entry_open = o_buf[0];
   if(CopyHigh(_Symbol, _Period, 1, 1, h_buf) == 1) g_log_entry_high = h_buf[0];
   if(CopyLow(_Symbol, _Period, 1, 1, l_buf) == 1) g_log_entry_low = l_buf[0];
   if(CopyClose(_Symbol, _Period, 1, 1, c_buf) == 1) g_log_entry_close = c_buf[0];
   long tv[1];
   if(CopyTickVolume(_Symbol, _Period, 1, 1, tv) == 1) g_log_entry_tv = (double)tv[0];
   g_log_entry_sp = (double)SymbolInfoInteger(_Symbol, SYMBOL_SPREAD);
   Print("ENTRYBAR ", _Symbol, " z=", z_val, " open=", g_log_entry_open,
         " high=", g_log_entry_high, " low=", g_log_entry_low, " close=", g_log_entry_close,
         " tv=", g_log_entry_tv, " sp=", g_log_entry_sp);
}

void LogClose(string reason, double exit_price, double pnl) {
   if(g_log_file == INVALID_HANDLE) {
      g_log_file = FileOpen("V2z_v2_trades.csv", FILE_TXT|FILE_READ|FILE_WRITE|FILE_CSV|FILE_ANSI|FILE_COMMON, ",");
      if(g_log_file != INVALID_HANDLE)
         FileWrite(g_log_file, "time", "dir", "entry", "sl", "z", "atr",
                   "tick_vol", "spread", "bar_open", "bar_high", "bar_low", "bar_close",
                   "exit_reason", "exit_price", "pnl", "hold_sec");
   }
   if(g_log_file == INVALID_HANDLE) return;
   double hold = (double)(TimeCurrent() - g_entry_time);
   FileWrite(g_log_file, TimeCurrent(), g_direction, g_entry_price, g_current_stop,
             g_log_entry_z, g_log_entry_atr, g_log_entry_tv, g_log_entry_sp,
             g_log_entry_open, g_log_entry_high, g_log_entry_low, g_log_entry_close,
             reason, exit_price, pnl, hold);
   FileFlush(g_log_file);
}

bool ClosePosition(string reason) {
   if(g_direction == 0) return false;
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
   req.deviation = (ulong)(SLIPPAGE_PIPS * 10);
   req.magic = MAGIC_NUMBER;
   req.comment = "V2z_v2_close";

   PositionSelect(_Symbol);
   ulong ticket = (ulong)PositionGetInteger(POSITION_TICKET);
   if(ticket == 0) {
      Print("No position for ", _Symbol); ClearPosition(); return false;
   }
   req.position = ticket;

   if(!OrderSend(req, res) || res.retcode != TRADE_RETCODE_DONE) {
      Print("Close failed: ", GetLastError()); return false;
   }
   double exit_price = res.price;
   double slip = SlipPrice();
   if(g_direction > 0) exit_price -= slip;
   else exit_price += slip;
   double cs = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_CONTRACT_SIZE);
   double raw_pnl = (g_direction > 0) ?
      (exit_price - g_entry_price) * BASE_LOT * cs :
      (g_entry_price - exit_price) * BASE_LOT * cs;
   double commission = BASE_LOT * COMMISSION_PER_LOT;
   double pnl = raw_pnl - commission;
   Print("CLOSE ", _Symbol, " rsn=", reason, " exit=", res.price,
           " raw=", raw_pnl, " comm=", commission, " pnl=", pnl);
   LogClose(reason, res.price, pnl);
   Print("CLOSEDONE ", _Symbol, " pnl=", pnl);
   ClearPosition();
   return true;
}

ulong LogCloseFromHistory() {
   if(!HistorySelect(0, TimeCurrent())) return 0;
   int total = HistoryDealsTotal();
   for(int i = total-1; i >= 0; i--) {
      ulong dtkt = HistoryDealGetTicket(i);
      if(dtkt == 0) continue;
      if(HistoryDealGetString(dtkt, DEAL_SYMBOL) != _Symbol) continue;
      if((long)HistoryDealGetInteger(dtkt, DEAL_MAGIC) != MAGIC_NUMBER) continue;
      if((long)HistoryDealGetInteger(dtkt, DEAL_ENTRY) != DEAL_ENTRY_OUT) continue;
      double profit = HistoryDealGetDouble(dtkt, DEAL_PROFIT);
      double commission = HistoryDealGetDouble(dtkt, DEAL_COMMISSION);
      double swap = HistoryDealGetDouble(dtkt, DEAL_SWAP);
      double pnl = profit + commission + swap;
      double exit_price = HistoryDealGetDouble(dtkt, DEAL_PRICE);
      LogClose("slexit", exit_price, pnl);
      Print("CLOSEDONE ", _Symbol, " pnl=", pnl, " hist_profit=", profit, " hist_comm=", commission);
      return dtkt;
   }
   return 0;
}

void ManagePosition() {
   if(g_direction == 0) return;
   if(!PositionSelect(_Symbol)) { LogCloseFromHistory(); ClearPosition(); return; }
   MqlTick tick;
   if(!SymbolInfoTick(_Symbol, tick)) return;

   double atr_v = GetATRValue();
   if(atr_v <= 0) return;
   double tg = TRIG_A * atr_v;
   double gp = GAP_A * atr_v;

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
   if(TimeCurrent() - g_entry_time >= MAX_HOLD_SECONDS) ClosePosition("expiry");
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

   double eff_z = GetEffectiveZThreshold();

   if(MathAbs(z) >= eff_z) {
      int dir = (z > 0) ? -1 : 1;
      if(!IsTrendFilterAllowed(dir)) {
         Print("TREND FILTER: skipping dir=", dir, " z=", z, " ema_rising=", (g_ema_value > g_ema_prev));
         return;
      }
      bool use_micro = ENABLE_MICRO_FILTER;
      if(ENABLE_VOL_REGIME_FILTER) {
         double short_atr = GetATRValue();
         double atr_pips = short_atr * 10000;
         use_micro = (atr_pips < VOL_REGIME_ATR_MIN_PIPS);
         Print("VOL REGIME: atr=", atr_pips, "p thresh=", VOL_REGIME_ATR_MIN_PIPS, "p micro=", use_micro);
      }
      if(use_micro) {
         double o_buf[1], h_buf[1], l_buf[1], c_buf[1];
         long tv_buf[1];
         bool micro_ok = true;
         if(CopyOpen(_Symbol, _Period, 1, 1, o_buf) == 1 &&
            CopyHigh(_Symbol, _Period, 1, 1, h_buf) == 1 &&
            CopyLow(_Symbol, _Period, 1, 1, l_buf) == 1 &&
            CopyClose(_Symbol, _Period, 1, 1, c_buf) == 1) {
            double body = MathAbs(c_buf[0] - o_buf[0]);
            double rng = h_buf[0] - l_buf[0];
            double body_pips = body * 10000;
            double smoothness = (rng > 1e-10) ? body / rng : 1.0;
            CopyTickVolume(_Symbol, _Period, 1, 1, tv_buf);
            double tv = (double)tv_buf[0];
            if(body_pips < MICRO_MIN_BODY_PIPS || smoothness < MICRO_MIN_SMOOTHNESS || tv < MICRO_MIN_TICK_VOL) {
               Print("MICRO FILTER: skip body=", body_pips, "p sm=", smoothness, " tv=", tv);
               micro_ok = false;
            }
         }
         if(!micro_ok) return;
      }
      OpenPosition(dir, av, z, av);
   }
}

void OnTick() {
   if(g_direction != 0) ManagePosition();
   if(IsNewBar()) {
      UpdateCloseBuffer();
      UpdateATRBuffer();
      UpdateTrendEMA();
      UpdateVolEWMA();
      if(g_direction == 0) CheckEntry();
   }
}