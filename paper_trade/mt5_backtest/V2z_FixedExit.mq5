#property version "1.00"
#property description "V2z FixedExit - Fade z-score extremes on M1, fixed bar-count exit. No trailing stop."

input double   Z_THRESHOLD         = 2.5;
input int      HOLD_BARS           = 2;
input double   CATASTROPHE_SL_MULT = 5.0;
input int      ATR_PERIOD          = 20;
input int      Z_WINDOW            = 50;
input double   BASE_LOT            = 0.05;
input double   MAX_DAILY_LOSS      = 1250.0;
input double   MAX_SPREAD_PIPS     = 3.0;
input int      MAGIC_NUMBER        = 202411;
input int      MAX_TRADES_DAY      = 200;
input double   SLIPPAGE_PIPS       = 0.3;
input double   COMMISSION_PER_LOT  = 5.0;
input bool     NEWS_FILTER         = true;
input int      TRADE_START_HOUR    = 0;
input int      TRADE_END_HOUR      = 24;

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
double   g_catastrophe_sl = 0;
int      g_bars_held = 0;
int      g_direction = 0;

double SlipPrice() {
   return SLIPPAGE_PIPS * _Point * 10.0;
}

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
   Print("V2z_FixedExit on ", _Symbol, " magic=", MAGIC_NUMBER,
         " z=", Z_THRESHOLD, " hold=", HOLD_BARS, "bar lot=", BASE_LOT,
         " comm=", COMMISSION_PER_LOT, "$/lot slip=", SLIPPAGE_PIPS, "pip");
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason) {
   if(g_atr_handle != INVALID_HANDLE) IndicatorRelease(g_atr_handle);
}

void ClearPosition() {
   g_entry_price = 0; g_bars_held = 0;
   g_catastrophe_sl = 0; g_direction = 0;
}

bool IsNewBar() {
   datetime times[1];
   if(CopyTime(_Symbol, PERIOD_M1, 0, 1, times) != 1) return false;
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
   long points = SymbolInfoInteger(_Symbol, SYMBOL_SPREAD);
   return (double)points / 10.0;
}

bool IsNearNews() {
   if(!NEWS_FILTER) return false;
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   int min = dt.hour * 60 + dt.min;
   int news_times[] = {375, 435, 555, 570, 745, 830, 870, 915, 1015, 1045, 1115, 1145, 1215, 1230, 1330, 1400, 1415, 1430};
   int total = ArraySize(news_times);
   for(int i = 0; i < total; i++) {
      if(min >= news_times[i] - 5 && min <= news_times[i] + 5)
         return true;
   }
   return false;
}

bool OpenPosition(int direction, double atr_v) {
   double cs = CATASTROPHE_SL_MULT * atr_v;
   MqlTick tick;
   if(!SymbolInfoTick(_Symbol, tick)) return false;
   double price = (direction > 0) ? tick.ask : tick.bid;
   double sl = (direction > 0) ? (price - cs) : (price + cs);
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
   req.comment = "V2z_FE";

   if(!OrderSend(req, res) || res.retcode != TRADE_RETCODE_DONE) {
      Print("OrderSend failed: ", res.retcode, " ", GetLastError());
      return false;
   }
   g_direction = direction;
   g_entry_price = res.price;
   double slip = SlipPrice();
   if(g_direction > 0) g_entry_price += slip;
   else g_entry_price -= slip;
   g_catastrophe_sl = sl;
   g_bars_held = 0;
   g_daily_trades++;
   Print("OPEN ", _Symbol, " dir=", direction, " entry=", g_entry_price,
         " cs=", cs, " atr=", atr_v, " slip=", slip);
   return true;
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
   ClearPosition();
   return true;
}

void CheckEntry() {
   if(g_direction != 0) return;
   MqlDateTime dt; TimeToStruct(TimeCurrent(), dt);
   int hour = dt.hour;
   if(TRADE_START_HOUR < TRADE_END_HOUR) {
      if(hour < TRADE_START_HOUR || hour >= TRADE_END_HOUR) return;
   } else {
      if(hour < TRADE_START_HOUR && hour >= TRADE_END_HOUR) return;
   }
   if(GetSpreadPips() > MAX_SPREAD_PIPS) return;
   if(IsNearNews()) return;
   UpdateDailyCounters();
   if(g_daily_trades >= MAX_TRADES_DAY) return;

   double z = ComputeZScore();
   double av = GetATRValue();
   if(av <= 0) return;

   if(MathAbs(z) >= Z_THRESHOLD) {
      int dir = (z > 0) ? -1 : 1;
      OpenPosition(dir, av);
   }
}

void OnTick() {
   if(!IsNewBar()) return;

   UpdateCloseBuffer();
   UpdateATRBuffer();

   if(g_direction != 0) {
      g_bars_held++;
      if(g_bars_held >= HOLD_BARS) {
         ClosePosition("hold_expiry");
      }
   }

   if(g_direction == 0) {
      CheckEntry();
   }
}
