#property version "1.00"
#property description "Dark Consensus - EURUSD-only momentum on M1"

input double   MAG_THRESHOLD   = 0.000187;
input int      HOLD_BARS       = 5;
input double   BASE_LOT        = 0.3;
input double   SLIPPAGE_PIPS   = 1.0;
input int      MAGIC_NUMBER    = 95001;

datetime g_last_bar = 0;
datetime g_entry_t = 0;
int      g_last_bar_held = 0;
int      g_trades = 0;
int      g_bars = 0;
bool     g_debugged = false;

void ClosePos() {
   if(!PositionSelect(_Symbol)) { g_entry_t = 0; return; }
   ulong ticket = PositionGetInteger(POSITION_TICKET);
   double vol = PositionGetDouble(POSITION_VOLUME);
   long   pt   = PositionGetInteger(POSITION_TYPE);
   MqlTradeRequest req = {};
   req.action = TRADE_ACTION_DEAL;
   req.position = ticket;
   req.symbol = _Symbol;
   req.volume = vol;
   req.type = (pt == POSITION_TYPE_BUY) ? ORDER_TYPE_SELL : ORDER_TYPE_BUY;
   req.price = (pt == POSITION_TYPE_BUY) ? SymbolInfoDouble(_Symbol, SYMBOL_BID)
                                          : SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   req.deviation = 5;
   req.magic = MAGIC_NUMBER;
   MqlTradeResult res;
   OrderSend(req, res);
   g_entry_t = 0; g_last_bar_held = 0;
}

void OnTick() {
   datetime bt[1];
   if(CopyTime(_Symbol, PERIOD_M1, 0, 1, bt) != 1) return;
   bool new_bar = false;
   if(bt[0] != g_last_bar) {
      g_last_bar = bt[0];
      g_bars++;
      new_bar = true;
   }
   if(g_entry_t > 0) {
      if(new_bar) g_last_bar_held++;
      if(g_last_bar_held >= HOLD_BARS) ClosePos();
      return;
   }
   if(!new_bar) return;
   if(g_bars < 1440) return;

   MqlRates rr[2];
   if(CopyRates(_Symbol, PERIOD_M1, 0, 2, rr) < 2) return;
   double ret = (rr[0].close - rr[1].close) / rr[1].close;
   double mag = MathAbs(ret);

   MqlDateTime mdt; TimeToStruct(bt[0], mdt);
   if(mdt.hour < 7 || mdt.hour > 21) return;
   if(mag <= MAG_THRESHOLD) return;

   if(!g_debugged) {
      Print("DBG ret=", ret, " mag=", mag, " thr=", MAG_THRESHOLD);
      g_debugged = true;
   }

   ENUM_ORDER_TYPE dir = (ret > 0) ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
   double price = (dir == ORDER_TYPE_BUY) ? SymbolInfoDouble(_Symbol, SYMBOL_ASK)
                                          : SymbolInfoDouble(_Symbol, SYMBOL_BID);
   MqlTradeRequest req = {};
   req.action = TRADE_ACTION_DEAL;
   req.symbol = _Symbol;
   req.volume = BASE_LOT;
   req.type = dir;
   req.price = price;
   req.deviation = 5;
   req.magic = MAGIC_NUMBER;
   MqlTradeResult res;
   if(OrderSend(req, res)) {
      g_entry_t = bt[0];
      g_last_bar_held = 0;
      g_trades++;
   }
}
