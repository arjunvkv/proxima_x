#!/usr/bin/env python3
"""Patch and Deploy Verified Test_Min_Fire_MT5 v1.03 with PositionSelect SL/TP Engine to VPS."""

import os, subprocess, shutil

VPS_KEY = r"C:\Users\Arjun Sasi\Downloads\ssh-key-2026-07-28.key"
VPS_PATH = "ubuntu@140.245.234.92:/home/ubuntu/.wine/drive_c/Program Files/FTMO Global Markets MT5 Terminal/MQL5/Experts/"
LOCAL_DIR = r"C:\Trading\Agentic_Trading\proxima_x\paper_trade\mt5_backtest"

V103_CODE = """//+------------------------------------------------------------------+
//|                                     Test_Min_Fire_MT5.mq5        |
//|   1-Minute Live Fire Test EA v1.03 with Verified ECN SL/TP Engine|
//+------------------------------------------------------------------+
#property copyright "Proxima Trading"
#property version   "1.03"
#property strict

#include <Trade\\Trade.mqh>
CTrade trade;

input double   TEST_LOT    = 1.20;    // Standard Test Volume (1.20L)
input ulong    MAGIC_NUM   = 999999;  // Test Magic Number
input int      SL_PIPS     = 35;      // Emergency SL Pips
input int      TP_PIPS     = 45;      // Safety TP Pips

datetime g_last_minute = 0;

double NormalizeVolume(string symbol, double volume) {
   SymbolSelect(symbol, true);
   double min_vol  = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);
   double max_vol  = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MAX);
   double step_vol = SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP);
   if(step_vol <= 0.0) step_vol = 0.01;
   
   double normalized_vol = MathRound(volume / step_vol) * step_vol;
   if(normalized_vol < min_vol) normalized_vol = min_vol;
   if(normalized_vol > max_vol) normalized_vol = max_vol;
   
   int digits = (step_vol == 0.01) ? 2 : ((step_vol == 0.1) ? 1 : 0);
   return NormalizeDouble(normalized_vol, digits);
}

int OnInit() {
   Print("=== 🧪 TEST MIN FIRE v1.03: Live ECN SL/TP Attachment & Auto-Close Engine Active ===");
   return INIT_SUCCEEDED;
}

void OnDeinit(const int) {
   Print("=== 🧪 TEST MIN FIRE DEINIT ===");
}

void CloseTestPositions() {
   for(int i = PositionsTotal() - 1; i >= 0; i--) {
      ulong ticket = PositionGetTicket(i);
      if(ticket <= 0) continue;
      long magic = PositionGetInteger(POSITION_MAGIC);
      if(magic != MAGIC_NUM) continue;
      
      string s = PositionGetString(POSITION_SYMBOL);
      double raw_vol = PositionGetDouble(POSITION_VOLUME);
      double vol = NormalizeVolume(s, raw_vol);
      if(vol <= 0.0) continue;
      
      if(trade.PositionClosePartial(ticket, vol)) {
         Print("  🧪 TEST CLOSE SUCCESS (Partial) | Symbol: ", s, " | Ticket: ", ticket, " | Vol: ", vol);
      } else if(trade.PositionClose(ticket)) {
         Print("  🧪 TEST CLOSE SUCCESS (CTrade) | Symbol: ", s, " | Ticket: ", ticket);
      }
   }
}

void OpenTestPosition() {
   string s = _Symbol;
   MqlTick tk;
   if(!SymbolInfoTick(s, tk)) return;
   
   double pr = tk.ask;
   double sl_d = (StringFind(s, "JPY") >= 0) ? (SL_PIPS * 0.01) : (SL_PIPS * 0.0001);
   double tp_d = (StringFind(s, "JPY") >= 0) ? (TP_PIPS * 0.01) : (TP_PIPS * 0.0001);
   int digits = (int)SymbolInfoInteger(s, SYMBOL_DIGITS);
   double sl_val = NormalizeDouble(pr - sl_d, digits);
   double tp_val = NormalizeDouble(pr + tp_d, digits);
   
   MqlTradeRequest req = {}; MqlTradeResult res = {};
   req.action = TRADE_ACTION_DEAL;
   req.symbol = s;
   req.volume = (float)NormalizeVolume(s, TEST_LOT);
   req.type = ORDER_TYPE_BUY;
   req.price = pr;
   req.sl = sl_val;
   req.tp = tp_val;
   req.deviation = 10;
   req.magic = MAGIC_NUM;
   req.comment = "Test_1min_buy";
   
   if(OrderSend(req, res) && res.retcode == TRADE_RETCODE_DONE) {
      Print("  🧪 TEST BUY FILL SUCCESS | Symbol: ", s, " | Order: ", res.order, " | Price: ", res.price);
      
      // Guaranteed Post-Fill ECN PositionSelect SL/TP Attachment
      Sleep(200);
      if(PositionSelect(s)) {
         ulong pos_tix = PositionGetInteger(POSITION_TICKET);
         if(trade.PositionModify(pos_tix, sl_val, tp_val)) {
            Print("  🧪 TEST SL/TP ATTACHED SUCCESS | Symbol: ", s, " | PosTicket: ", pos_tix, " | SL: ", sl_val, " | TP: ", tp_val);
         } else {
            Print("  ⚠️ PositionModify failed on pos_tix ", pos_tix, " err: ", GetLastError());
         }
      }
   } else {
      Print("  ⚠️ TEST BUY FAIL | Code: ", res.retcode, " | Comment: ", res.comment);
   }
}

void OnTick() {
   MqlDateTime dt;
   TimeCurrent(dt);
   
   datetime current_min = TimeCurrent() - (dt.sec);
   if(current_min != g_last_minute) {
      g_last_minute = current_min;
      
      CloseTestPositions();
      OpenTestPosition();
   }
}
"""

def main():
    test_path = os.path.join(LOCAL_DIR, "Test_Min_Fire_MT5.mq5")
    with open(test_path, "w", encoding="utf-8") as f:
        f.write(V103_CODE)
    print("  • Updated Test_Min_Fire_MT5.mq5 locally to v1.03!")

    # SCP directly to VPS
    subprocess.run(["scp", "-i", VPS_KEY, test_path, VPS_PATH], check=True)
    print("🚀 Pushed Test_Min_Fire_MT5.mq5 v1.03 to VPS Experts folder!")

if __name__ == "__main__":
    main()
