#property version "1.00"
#property description "MicroScalp - Fade clean shock bars (body/smoothness/tickvol entry, tight exit)"

input double   MICRO_BODY_PIPS       = 4.0;
input double   MICRO_SMOOTHNESS      = 0.85;
input double   MICRO_TICK_VOL        = 200;
input double   MICRO_TP_MULTIPLIER   = 1.0;
input double   MICRO_SL_MULTIPLIER   = 1.5;
input int      MICRO_MAX_HOLD_SEC    = 180;
input double   BASE_LOT              = 0.75;
input double   MAX_SPREAD_PIPS       = 5.0;
input double   SLIPPAGE_PIPS         = 1.0;
input double   COMMISSION_PER_LOT    = 5.0;
input int      MAGIC_NUMBER          = 202412;
input int      MAX_TRADES_DAY        = 200;
input int      TRADE_START_HOUR      = 0;
input int      TRADE_END_HOUR        = 7;

datetime g_last_bar_time = 0;
int      g_daily_trades = 0;
string   g_daily_date = "";
double   g_entry_price = 0;
datetime g_entry_time = 0;
double   g_best_price = 0;
double   g_current_stop = 0;
double   g_take_profit = 0;
int      g_direction = 0;
double   g_entry_body_pips = 0;

double SlipPrice() {
   return SLIPPAGE_PIPS * _Point * 10.0;
}

int OnInit() {
   if(!SymbolInfoInteger(_Symbol, SYMBOL_TRADE_MODE)) {
      Print("Symbol ", _Symbol, " not tradeable");
      return INIT_FAILED;
   }
   Print("MicroScalp v1 on ", _Symbol, " magic=", MAGIC_NUMBER);
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason) {}

void ClearPosition() {
   g_entry_price = 0; g_entry_time = 0;
   g_best_price = 0; g_current_stop = 0; g_direction = 0;
   g_take_profit = 0; g_entry_body_pips = 0;
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
   req.comment = "MCR_SCLP";
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
   Print("MICRO OPEN ", _Symbol, " dir=", direction, " entry=", entry, " sl=", sl, " tp=", tp);
   return true;
}

void ClosePositionMicro(string reason) {
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
   req.comment = "MCR_CLS";
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
   Print("MICRO CLOSE ", _Symbol, " rsn=", reason, " entry=", g_entry_price, " exit=", ep,
         " raw=", raw, " comm=", comm, " pnl=", pnl);
   ClearPosition();
}

void ManageMicroPosition() {
   if(g_direction == 0) return;
   if(!PositionSelect(_Symbol)) { ClearPosition(); return; }
   MqlTick tick;
   if(!SymbolInfoTick(_Symbol, tick)) return;
   if(g_direction > 0) {
      if(tick.bid >= g_take_profit) { ClosePositionMicro("tp"); return; }
      if(tick.bid <= g_current_stop) { ClosePositionMicro("sl"); return; }
   } else {
      if(tick.ask <= g_take_profit) { ClosePositionMicro("tp"); return; }
      if(tick.ask >= g_current_stop) { ClosePositionMicro("sl"); return; }
   }
   if(TimeCurrent() - g_entry_time >= MICRO_MAX_HOLD_SEC) ClosePositionMicro("expiry");
}

void CheckMicroEntry() {
   if(g_direction != 0) return;
   MqlDateTime dt; TimeToStruct(TimeCurrent(), dt);
   int hour = dt.hour;
   if(TRADE_START_HOUR < TRADE_END_HOUR) { if(hour < TRADE_START_HOUR || hour >= TRADE_END_HOUR) return; }
   else { if(hour < TRADE_START_HOUR && hour >= TRADE_END_HOUR) return; }
   double sp = GetSpreadPips();
   if(sp > MAX_SPREAD_PIPS) return;
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

   int dir = (c_buf[0] > o_buf[0]) ? -1 : 1;
   double entry_pip_distance = MICRO_SL_MULTIPLIER * rng;
   MqlTick tick;
   if(!SymbolInfoTick(_Symbol, tick)) return;
   double entry = (dir > 0) ? tick.ask : tick.bid;
   double sl = (dir > 0) ? (entry - entry_pip_distance) : (entry + entry_pip_distance);
   double tp_dist = MICRO_TP_MULTIPLIER * body;
   double tp = (dir > 0) ? (entry + tp_dist) : (entry - tp_dist);

   Print("MICRO CHECK: bar body=", body_pips, "p sm=", smoothness, " tv=", tv,
         " dir=", dir, " entry=", entry, " sl=", sl, " tp=", tp);
   OpenPosition(dir, entry, sl, tp);
}

void OnTick() {
   if(g_direction != 0) ManageMicroPosition();
   if(IsNewBar()) {
      if(g_direction == 0) CheckMicroEntry();
   }
}
