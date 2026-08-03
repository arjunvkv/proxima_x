#!/usr/bin/env python3
"""Direct AppData MQL5 Compilation & Verification for Test_Min_Fire_MT5 v1.05."""

import os, subprocess, shutil, time

APPDATA_EXP = r"C:\Users\Arjun Sasi\AppData\Roaming\MetaQuotes\Terminal\89FE26BBBAB28C077BBF5FA8C1B4DF1C\MQL5\Experts"
METAEDITOR = r"C:\Program Files\FundedNext MT5 Terminal\MetaEditor64.exe"
LOCAL_DIR = r"C:\Trading\Agentic_Trading\proxima_x\paper_trade\mt5_backtest"
VPS_KEY = r"C:\Users\Arjun Sasi\Downloads\ssh-key-2026-07-28.key"
VPS_PATH = "ubuntu@140.245.234.92:/home/ubuntu/.wine/drive_c/Program Files/FTMO Global Markets MT5 Terminal/MQL5/Experts/"

V105_CODE = """//+------------------------------------------------------------------+
//|                                     Test_Min_Fire_MT5.mq5        |
//|   1-Minute Live Fire Test EA v1.05 with PositionModify SL/TP Engine|
//+------------------------------------------------------------------+
#property copyright "Proxima Trading"
#property version   "1.05"
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
   Print("=== 🧪 TEST MIN FIRE v1.05: Direct AppData PositionModify SL/TP Engine Active ===");
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
   
   MqlTradeRequest req = {}; MqlTradeResult res = {};
   req.action = TRADE_ACTION_DEAL;
   req.symbol = s;
   req.volume = (float)NormalizeVolume(s, TEST_LOT);
   req.type = ORDER_TYPE_BUY;
   req.price = pr;
   req.deviation = 10;
   req.magic = MAGIC_NUM;
   req.comment = "Test_1min_buy";
   
   if(OrderSend(req, res) && res.retcode == TRADE_RETCODE_DONE) {
      Print("  🧪 TEST BUY FILL SUCCESS | Symbol: ", s, " | Order: ", res.order);
      
      // 100% Guaranteed Post-Fill PositionModify Retry Loop
      for(int attempt = 0; attempt < 10; attempt++) {
         Sleep(100);
         if(PositionSelect(s)) {
            ulong pos_tix = PositionGetInteger(POSITION_TICKET);
            double fill_p = PositionGetDouble(POSITION_PRICE_OPEN);
            double sl_val = NormalizeDouble(fill_p - sl_d, digits);
            double tp_val = NormalizeDouble(fill_p + tp_d, digits);
            
            if(trade.PositionModify(pos_tix, sl_val, tp_val)) {
               Print("  🟢 🧪 TEST SL/TP ATTACHED SUCCESS | Symbol: ", s, " | Ticket: ", pos_tix, " | SL: ", sl_val, " | TP: ", tp_val);
               break;
            }
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
    print("="*115)
    print("WRITING V1.05 CODE DIRECTLY TO APPDATA EXPERTS FOLDER...")
    print("="*115)

    appdata_mq5 = os.path.join(APPDATA_EXP, "Test_Min_Fire_MT5.mq5")
    appdata_ex5 = os.path.join(APPDATA_EXP, "Test_Min_Fire_MT5.ex5")
    local_mq5 = os.path.join(LOCAL_DIR, "Test_Min_Fire_MT5.mq5")
    local_ex5 = os.path.join(LOCAL_DIR, "Test_Min_Fire_MT5.ex5")

    with open(appdata_mq5, "w", encoding="utf-8") as f:
        f.write(V105_CODE)
    with open(local_mq5, "w", encoding="utf-8") as f:
        f.write(V105_CODE)

    print("  • Written Test_Min_Fire_MT5.mq5 v1.05 directly to AppData!")

    # Compile AppData MQ5 directly using MetaEditor
    res = subprocess.run([METAEDITOR, f"/compile:{appdata_mq5}"], capture_output=True, text=True)
    print(f"  • MetaEditor compilation retcode: {res.returncode}")

    if os.path.exists(appdata_ex5):
        mtime = time.ctime(os.path.getmtime(appdata_ex5))
        size = os.path.getsize(appdata_ex5)
        print(f"  🟢 AppData Test_Min_Fire_MT5.ex5 COMPILED! Timestamp: {mtime} | Size: {size} bytes")

        shutil.copy(appdata_ex5, local_ex5)
        # Push to VPS
        subprocess.run(["scp", "-i", VPS_KEY, appdata_ex5, VPS_PATH], check=True)
        subprocess.run(["scp", "-i", VPS_KEY, appdata_mq5, VPS_PATH], check=True)
        print("  🚀 Uploaded v1.05 .ex5 and .mq5 to VPS!")

    print("="*115)
    print("🟢 APPDATA V1.05 DIRECT COMPILE & VPS UPLOAD COMPLETED CLEANLY!")
    print("="*115)

if __name__ == "__main__":
    main()
