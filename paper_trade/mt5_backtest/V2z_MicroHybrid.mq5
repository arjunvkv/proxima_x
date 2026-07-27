#property version "1.00"
#property description "MicroHybrid - z-score regime determines direction: extreme=reversal, normal=momentum"

input double   Z_THRESHOLD           = 3.5;
input double   MICRO_BODY_PIPS       = 4.0;
input double   MICRO_SMOOTHNESS      = 0.85;
input double   MICRO_TICK_VOL        = 200;
input double   MICRO_TP_MULTIPLIER   = 1.0;
input double   MICRO_SL_MULTIPLIER   = 1.5;
input int      MICRO_MAX_HOLD_SEC    = 300;
input double   BASE_LOT              = 0.75;
input double   MAX_SPREAD_PIPS       = 5.0;
input double   SLIPPAGE_PIPS         = 1.0;
input double   COMMISSION_PER_LOT    = 5.0;
input int      MAGIC_NUMBER          = 202413;
input int      MAX_TRADES_DAY        = 200;
input int      TRADE_START_HOUR      = 0;
input int      TRADE_END_HOUR        = 7;
input int      Z_WINDOW              = 50;

double   g_close_buf[];
int      g_close_count = 0;
datetime g_last_bar_time = 0;
int      g_daily_trades = 0;
string   g_daily_date = "";
double   g_entry_price = 0;
datetime g_entry_time = 0;
double   g_best_price = 0;
double   g_current_stop = 0;
double   g_take_profit = 0;
int      g_direction = 0;
double   g_z_at_entry = 0;

double SlipPrice() {
   return SLIPPAGE_PIPS * _Point * 10.0;
}

int OnInit() {
   if(!SymbolInfoInteger(_Symbol, SYMBOL_TRADE_MODE)) {
      Print("Symbol ", _Symbol, " not tradeable");
      return INIT_FAILED;
   }
   ArrayResize(g_close_buf, Z_WINDOW + 2);
   Print("MicroHybrid v1 on ", _Symbol, " magic=", MAGIC_NUMBER, " zthresh=", Z_THRESHOLD);
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason) {}

void ClearPosition() {
   g_entry_price = 0; g_entry_time = 0;
   g_best_price = 0; g_current_stop = 0; g_direction = 0;
   g_take_profit = 0; g_z_at_entry = 0;
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
   string today = IntegerToString(dt.year) + StringFormat("%02d%02d", dt.mon, dt.day);
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

double GetSpreadPips() {
   long spread = SymbolInfoInteger(_Symbol, SYMBOL_SPREAD);
   double pip = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   return (double)spread * pip / _Point;
}

bool OpenPosition(int direction, double entry, double sl, double tp) {
   int order_type = (direction > 0) ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
   MqlTradeRequest req = {};
   MqlTradeResult res = {};
   req.action = TRADE_ACTION_DEAL;
   req.symbol = _Symbol;
   req.volume = BASE_LOT;
   req.type = order_type;
   req.price = entry;
   req.sl = sl;
   req.tp = tp;
   req.deviation = (ulong)(SLIPPAGE_PIPS * 10);
   req.magic = MAGIC_NUMBER;
   req.comment = "MCR_HYB";
   if(!OrderSend(req, res) || res.retcode != TRADE_RETCODE_DONE) {
      Print("OrderSend failed: ", res.retcode, " ", GetLastError());
      return false;
   }
   g_direction = direction;
   g_entry_price = entry;
   g_current_stop = sl;
   g_take_profit = tp;
   g_entry_time = TimeCurrent();
   g_best_price = g_entry_price;
   g_daily_trades++;
   Print("HYBRID OPEN ", _Symbol, " dir=", direction, " entry=", entry, " sl=", sl, " tp=", tp,
         " z=", g_z_at_entry);
   return true;
}

void ClosePositionHybrid(string reason) {
   if(g_direction == 0) return;
   MqlTick tick;
   if(!SymbolInfoTick(_Symbol, tick)) return;
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
   req.comment = "MCR_HCLS";
   PositionSelect(_Symbol);
   ulong ticket = (ulong)PositionGetInteger(POSITION_TICKET);
   if(ticket == 0) { ClearPosition(); return; }
   req.position = ticket;
   if(!OrderSend(req, res) || res.retcode != TRADE_RETCODE_DONE) {
      Print("Close failed: ", GetLastError()); return;
   }
   double ep = res.price;
   double slip = SlipPrice();
   if(g_direction > 0) ep -= slip; else ep += slip;
   double cs = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_CONTRACT_SIZE);
   double raw = (g_direction > 0) ? (ep - g_entry_price) * BASE_LOT * cs : (g_entry_price - ep) * BASE_LOT * cs;
   double comm = BASE_LOT * COMMISSION_PER_LOT;
   double pnl = raw - comm;
   Print("HYBRID CLOSE ", _Symbol, " rsn=", reason, " z=", g_z_at_entry,
         " raw=", raw, " comm=", comm, " pnl=", pnl);
   ClearPosition();
}

void ManagePosition() {
   if(g_direction == 0) return;
   if(!PositionSelect(_Symbol)) { ClearPosition(); return; }
   MqlTick tick;
   if(!SymbolInfoTick(_Symbol, tick)) return;
   if(g_direction > 0) {
      if(tick.bid >= g_take_profit) { ClosePositionHybrid("tp"); return; }
      if(tick.bid <= g_current_stop) { ClosePositionHybrid("sl"); return; }
   } else {
      if(tick.ask <= g_take_profit) { ClosePositionHybrid("tp"); return; }
      if(tick.ask >= g_current_stop) { ClosePositionHybrid("sl"); return; }
   }
   if(TimeCurrent() - g_entry_time >= MICRO_MAX_HOLD_SEC) ClosePositionHybrid("expiry");
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

   double o_buf[1], h_buf[1], l_buf[1], c_buf[1];
   long tv_buf[1];
   if(CopyOpen(_Symbol, _Period, 1, 1, o_buf) != 1 ||
      CopyHigh(_Symbol, _Period, 1, 1, h_buf) != 1 ||
      CopyLow(_Symbol, _Period, 1, 1, l_buf) != 1 ||
      CopyClose(_Symbol, _Period, 1, 1, c_buf) != 1) return;
   if(CopyTickVolume(_Symbol, _Period, 1, 1, tv_buf) != 1) return;

   double body = MathAbs(c_buf[0] - o_buf[0]);
   double rng = h_buf[0] - l_buf[0];
   double body_pips = body * 10000;
   double smoothness = (rng > 1e-10) ? body / rng : 1.0;
   double tv = (double)tv_buf[0];

   if(body_pips < MICRO_BODY_PIPS || smoothness < MICRO_SMOOTHNESS || tv < MICRO_TICK_VOL) return;

   double z = ComputeZScore();
   g_z_at_entry = z;

   int bar_dir = (c_buf[0] > o_buf[0]) ? 1 : -1;

   int trade_dir;
   string mode;
   if(MathAbs(z) >= Z_THRESHOLD) {
      trade_dir = -bar_dir;
      mode = "REVERSAL";
   } else {
      trade_dir = bar_dir;
      mode = "MOMENTUM";
   }

   double entry_pip_distance = MICRO_SL_MULTIPLIER * rng;
   MqlTick tick;
   if(!SymbolInfoTick(_Symbol, tick)) return;
   double entry = (trade_dir > 0) ? tick.ask : tick.bid;
   double sl = (trade_dir > 0) ? (entry - entry_pip_distance) : (entry + entry_pip_distance);
   double tp_dist = MICRO_TP_MULTIPLIER * body;
   double tp = (trade_dir > 0) ? (entry + tp_dist) : (entry - tp_dist);

   Print("HYBRID CHECK: mode=", mode, " bar=", (bar_dir>0?"BULL":"BEAR"), " trade=", (trade_dir>0?"BUY":"SELL"),
         " z=", z, " body=", body_pips, "p sm=", smoothness, " tv=", tv);
   OpenPosition(trade_dir, entry, sl, tp);
}

void OnTick() {
   if(g_direction != 0) ManagePosition();
   if(IsNewBar()) {
      UpdateCloseBuffer();
      if(g_direction == 0) CheckEntry();
   }
}
