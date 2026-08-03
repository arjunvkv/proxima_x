//+------------------------------------------------------------------+
//|                                           Test_Local_SLTP.mq5    |
//|  Local MT5 Live Execution Verification Script                    |
//+------------------------------------------------------------------+
#property copyright "Proxima Trading"
#property version   "1.00"
#property script_show_inputs

#include <Trade\Trade.mqh>
CTrade trade;

void OnStart() {
   Print("="*95);
   Print("=== 🧪 LOCAL MT5 LIVE VERIFICATION SCRIPT START ===");
   Print("="*95);

   string s = _Symbol;
   MqlTick tk;
   if(!SymbolInfoTick(s, tk)) return;

   double lot = 1.20;
   SymbolSelect(s, true);
   double min_vol  = SymbolInfoDouble(s, SYMBOL_VOLUME_MIN);
   double max_vol  = SymbolInfoDouble(s, SYMBOL_VOLUME_MAX);
   double step_vol = SymbolInfoDouble(s, SYMBOL_VOLUME_STEP);
   if(step_vol <= 0.0) step_vol = 0.01;

   double normalized_vol = MathRound(lot / step_vol) * step_vol;
   if(normalized_vol < min_vol) normalized_vol = min_vol;
   if(normalized_vol > max_vol) normalized_vol = max_vol;
   int digits = (step_vol == 0.01) ? 2 : ((step_vol == 0.1) ? 1 : 0);
   double final_vol = NormalizeDouble(normalized_vol, digits);

   Print("  1. VOLUME NORMALIZATION: Raw=", lot, " -> Normalized=", final_vol, " L 🟢");

   double pr = tk.ask;
   double sl_d = (StringFind(s, "JPY") >= 0) ? 0.35 : 0.0035;
   double tp_d = (StringFind(s, "JPY") >= 0) ? 0.45 : 0.0045;
   int price_digits = (int)SymbolInfoInteger(s, SYMBOL_DIGITS);
   double sl_val = NormalizeDouble(pr - sl_d, price_digits);
   double tp_val = NormalizeDouble(pr + tp_d, price_digits);

   Print("  2. PLACING MARKET BUY: Price=", pr, " SL=", sl_val, " TP=", tp_val);

   MqlTradeRequest req = {}; MqlTradeResult res = {};
   req.action = TRADE_ACTION_DEAL;
   req.symbol = s;
   req.volume = (float)final_vol;
   req.type = ORDER_TYPE_BUY;
   req.price = pr;
   req.deviation = 10;
   req.magic = 999999;
   req.comment = "Local_Script_Test";

   if(OrderSend(req, res) && res.retcode == TRADE_RETCODE_DONE) {
      Print("  🟢 MARKET BUY FILLED! Deal Ticket=", res.deal, " Order Ticket=", res.order);

      // ECN Post-Fill PositionSelect SL/TP Attachment
      Sleep(200);
      if(PositionSelect(s)) {
         ulong pos_tix = PositionGetInteger(POSITION_TICKET);
         double pos_vol = PositionGetDouble(POSITION_VOLUME);
         double pos_sl = PositionGetDouble(POSITION_SL);
         double pos_tp = PositionGetDouble(POSITION_TP);

         Print("  Position Before Modify: PosTicket=", pos_tix, " Vol=", pos_vol, " SL=", pos_sl, " TP=", pos_tp);

         if(trade.PositionModify(pos_tix, sl_val, tp_val)) {
            Sleep(200);
            if(PositionSelectByTicket(pos_tix)) {
               double updated_sl = PositionGetDouble(POSITION_SL);
               double updated_tp = PositionGetDouble(POSITION_TP);
               Print("  🟢 ECN POSITIONMODIFY SUCCESSFUL!");
               Print("     Attached SL: ", updated_sl, " (Target: ", sl_val, ") 🟢");
               Print("     Attached TP: ", updated_tp, " (Target: ", tp_val, ") 🟢");
            }
         } else {
            Print("  ⚠️ PositionModify failed: ", GetLastError());
         }

         // Close position cleanly after verification
         Sleep(1000);
         double close_vol = NormalizeDouble(pos_vol, digits);
         if(trade.PositionClosePartial(pos_tix, close_vol)) {
            Print("  🟢 TEST POSITION CLOSED CLEANLY AT ", tk.bid, "!");
         } else if(trade.PositionClose(pos_tix)) {
            Print("  🟢 TEST POSITION CLOSED VIA CTRADE AT ", tk.bid, "!");
         }
      }
   } else {
      Print("  ❌ OrderSend Failed: Retcode=", res.retcode, " Comment=", res.comment);
   }

   Print("="*95);
}
