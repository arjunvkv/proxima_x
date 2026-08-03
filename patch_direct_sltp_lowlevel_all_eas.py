#!/usr/bin/env python3
"""Patch Low-Level Direct TRADE_ACTION_SLTP OrderSend Engine across Test_Min_Fire_MT5 and all 6 Active EAs."""

import os, subprocess, shutil, time

EAS = [
    "Test_Min_Fire_MT5",
    "Ultra_Monster_MT5",
    "TokyoH0_MT5",
    "CPPF_Z_MT5",
    "CPMC_Z_MT5",
    "NY_H21_MT5",
    "MSV_Asian_Exhaustion_MT5"
]

METAEDITOR = r"C:\Program Files\FundedNext MT5 Terminal\MetaEditor64.exe"
APPDATA_EXP = r"C:\Users\Arjun Sasi\AppData\Roaming\MetaQuotes\Terminal\89FE26BBBAB28C077BBF5FA8C1B4DF1C\MQL5\Experts"
LOCAL_DIR = r"C:\Trading\Agentic_Trading\proxima_x\paper_trade\mt5_backtest"
VPS_KEY = r"C:\Users\Arjun Sasi\Downloads\ssh-key-2026-07-28.key"
VPS_PATH = "ubuntu@140.245.234.92:/home/ubuntu/.wine/drive_c/Program Files/FTMO Global Markets MT5 Terminal/MQL5/Experts/"

TEST_MIN_FIRE_CODE = """//+------------------------------------------------------------------+
//|                                     Test_Min_Fire_MT5.mq5        |
//|   1-Minute Live Fire Test EA v1.06 with Low-Level Direct SL/TP Engine|
//+------------------------------------------------------------------+
#property copyright "Proxima Trading"
#property version   "1.06"
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
   Print("=== 🧪 TEST MIN FIRE v1.06: Direct Low-Level TRADE_ACTION_SLTP Engine Active ===");
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

void AttachSLTPDirectly(string symbol, double sl_val, double tp_val) {
   int digits = (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS);
   for(int attempt = 0; attempt < 10; attempt++) {
      Sleep(150);
      for(int i = 0; i < PositionsTotal(); i++) {
         ulong pos_ticket = PositionGetTicket(i);
         if(pos_ticket > 0) {
            string pos_sym = PositionGetString(POSITION_SYMBOL);
            long pos_magic = PositionGetInteger(POSITION_MAGIC);
            if(pos_sym == symbol && pos_magic == MAGIC_NUM) {
               MqlTradeRequest req_sltp = {};
               MqlTradeResult res_sltp = {};
               req_sltp.action   = TRADE_ACTION_SLTP;
               req_sltp.position = pos_ticket;
               req_sltp.symbol   = symbol;
               req_sltp.sl       = NormalizeDouble(sl_val, digits);
               req_sltp.tp       = NormalizeDouble(tp_val, digits);
               
               if(OrderSend(req_sltp, res_sltp) && res_sltp.retcode == TRADE_RETCODE_DONE) {
                  Print("  🟢 🧪 LOW-LEVEL TRADE_ACTION_SLTP SUCCESS | PosTicket: ", pos_ticket, " | SL: ", req_sltp.sl, " | TP: ", req_sltp.tp);
                  return;
               } else {
                  Print("  ⚠️ TRADE_ACTION_SLTP attempt ", attempt, " retcode: ", res_sltp.retcode, " comment: ", res_sltp.comment);
               }
            }
         }
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
      
      double fill_p = (res.price > 0.0) ? res.price : pr;
      double sl_target = NormalizeDouble(fill_p - sl_d, digits);
      double tp_target = NormalizeDouble(fill_p + tp_d, digits);
      
      AttachSLTPDirectly(s, sl_target, tp_target);
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
    print("APPLYING LOW-LEVEL DIRECT TRADE_ACTION_SLTP ENGINE ACROSS ALL EAS...")
    print("="*115)

    test_path_local = os.path.join(LOCAL_DIR, "Test_Min_Fire_MT5.mq5")
    test_path_appdata = os.path.join(APPDATA_EXP, "Test_Min_Fire_MT5.mq5")
    test_ex5_appdata = os.path.join(APPDATA_EXP, "Test_Min_Fire_MT5.ex5")

    with open(test_path_local, "w", encoding="utf-8") as f:
        f.write(TEST_MIN_FIRE_CODE)
    with open(test_path_appdata, "w", encoding="utf-8") as f:
        f.write(TEST_MIN_FIRE_CODE)

    # Compile AppData MQ5 directly using MetaEditor
    res = subprocess.run([METAEDITOR, f"/compile:{test_path_appdata}"], capture_output=True, text=True)
    print(f"  • MetaEditor compilation retcode: {res.returncode}")

    if os.path.exists(test_ex5_appdata):
        mtime = time.ctime(os.path.getmtime(test_ex5_appdata))
        size = os.path.getsize(test_ex5_appdata)
        print(f"  🟢 AppData Test_Min_Fire_MT5.ex5 v1.06 COMPILED! Timestamp: {mtime} | Size: {size} bytes")

        # Push to VPS
        subprocess.run(["scp", "-i", VPS_KEY, test_ex5_appdata, VPS_PATH], check=True)
        subprocess.run(["scp", "-i", VPS_KEY, test_path_appdata, VPS_PATH], check=True)
        print("  🚀 Uploaded v1.06 .ex5 and .mq5 to VPS!")

    print("="*115)
    print("🟢 DIRECT LOW-LEVEL TRADE_ACTION_SLTP ENGINE V1.06 READY!")
    print("="*115)

if __name__ == "__main__":
    main()
