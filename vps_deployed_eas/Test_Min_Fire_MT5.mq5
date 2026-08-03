//+------------------------------------------------------------------+
//|                                     Test_Min_Fire_MT5.mq5        |
//|   1-Minute Micro-Lot Test EA to Verify Trade Execution on VPS    |
//+------------------------------------------------------------------+
#property copyright "Proxima Trading"
#property version   "1.00"
#property strict

input double   TEST_LOT    = 0.01;    // Micro Lot Size
input ulong    MAGIC_NUM   = 999999;  // Test Magic Number

datetime g_last_minute = 0;
ulong    g_current_ticket = 0;

int OnInit() {
   Print("=== 🧪 TEST EA INIT: 1-Minute Micro-Lot Fire Test Active ===");
   return INIT_SUCCEEDED;
}

void OnDeinit(const int) {
   Print("=== 🧪 TEST EA DEINIT ===");
}

void CloseTestPosition() {
   if(g_current_ticket == 0) return;
   if(PositionSelectByTicket(g_current_ticket)) {
      string s = PositionGetString(POSITION_SYMBOL);
      double vol = PositionGetDouble(POSITION_VOLUME);
      ENUM_POSITION_TYPE pos_type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
      
      MqlTick tk;
      if(!SymbolInfoTick(s, tk)) return;
      
      MqlTradeRequest req = {}; MqlTradeResult res = {};
      req.action = TRADE_ACTION_DEAL;
      req.symbol = s;
      req.volume = vol;
      req.position = g_current_ticket;
      req.type = (pos_type == POSITION_TYPE_BUY) ? ORDER_TYPE_SELL : ORDER_TYPE_BUY;
      req.price = (pos_type == POSITION_TYPE_BUY) ? tk.bid : tk.ask;
      req.deviation = 10;
      req.magic = MAGIC_NUM;
      req.comment = "Test_1min_close";
      
      if(OrderSend(req, res) && res.retcode == TRADE_RETCODE_DONE) {
         Print("  🧪 TEST CLOSE SUCCESS | Symbol: ", s, " | Ticket: ", g_current_ticket, " | Close Price: ", res.price);
         g_current_ticket = 0;
      }
   } else {
      g_current_ticket = 0;
   }
}

void OpenTestPosition() {
   MqlTick tk;
   if(!SymbolInfoTick(_Symbol, tk)) {
      Print("  🧪 TEST FAIL: Could not fetch tick for ", _Symbol);
      return;
   }
   
   MqlTradeRequest req = {}; MqlTradeResult res = {};
   req.action = TRADE_ACTION_DEAL;
   req.symbol = _Symbol;
   req.volume = TEST_LOT;
   req.type = ORDER_TYPE_BUY;
   req.price = tk.ask;
   req.deviation = 10;
   req.magic = MAGIC_NUM;
   req.comment = "Test_1min_buy";
   
   if(OrderSend(req, res) && res.retcode == TRADE_RETCODE_DONE) {
      g_current_ticket = res.order;
      Print("  🧪 TEST BUY SUCCESS | Symbol: ", _Symbol, " | Ticket: ", g_current_ticket, " | Fill Price: ", res.price);
   } else {
      Print("  🧪 TEST BUY FAIL | Code: ", res.retcode, " | Comment: ", res.comment);
   }
}

void OnTick() {
   MqlDateTime dt;
   TimeCurrent(dt);
   
   // Fire every minute when minute changes
   datetime current_min = TimeCurrent() - (dt.sec);
   if(current_min != g_last_minute) {
      g_last_minute = current_min;
      
      // Close previous minute's test position if active
      CloseTestPosition();
      
      // Open new 1-minute test position
      OpenTestPosition();
   }
}
