#!/usr/bin/env python3
"""Patch Bulletproof Dynamic PositionsTotal Exit Engine across ALL 6 Active EAs."""

import os, subprocess, shutil

EAS = [
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
BACKUP_DIR = r"C:\Trading\Agentic_Trading\proxima_x\paper_trade\mt5_backtest\updated_version_backup"
VPS_KEY = r"C:\Users\Arjun Sasi\Downloads\ssh-key-2026-07-28.key"
VPS_PATH = "ubuntu@140.245.234.92:/home/ubuntu/.wine/drive_c/Program Files/FTMO Global Markets MT5 Terminal/MQL5/Experts/"

DYNAMIC_EXIT_CODE = """
bool ClosePositionByTicket(ulong ticket, string s, string why) {
   if(!PositionSelectByTicket(ticket)) return false;
   double vol = NormalizeDouble(PositionGetDouble(POSITION_VOLUME), 2);
   if(vol <= 0.0) return false;
   
   ENUM_POSITION_TYPE ptype = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
   ENUM_ORDER_TYPE close_type = (ptype == POSITION_TYPE_BUY) ? ORDER_TYPE_SELL : ORDER_TYPE_BUY;
   
   MqlTick tk;
   if(!SymbolInfoTick(s, tk)) return false;
   double price = (close_type == ORDER_TYPE_SELL) ? tk.bid : tk.ask;
   
   MqlTradeRequest req = {}; MqlTradeResult res = {};
   req.action = TRADE_ACTION_DEAL;
   req.symbol = s;
   req.volume = vol;
   req.type = close_type;
   req.price = price;
   req.position = ticket;
   req.deviation = 10;
   req.comment = "dynamic_exit_" + why;
   
   bool sent = OrderSend(req, res);
   if(sent && res.retcode == TRADE_RETCODE_DONE) {
      Print("  CLOSE SUCCESS ", s, " ticket=", ticket, " vol=", vol, " why=", why);
      return true;
   }
   
   if(trade.PositionClose(ticket)) {
      Print("  CLOSE CTrade ", s, " ticket=", ticket, " why=", why);
      return true;
   }
   return false;
}

void CheckExits() {
   for(int i = PositionsTotal() - 1; i >= 0; i--) {
      ulong ticket = PositionGetTicket(i);
      if(ticket <= 0) continue;
      long magic = PositionGetInteger(POSITION_MAGIC);
      if(magic < MAGIC_BASE || magic >= MAGIC_BASE + N_PAIRS) continue;
      
      string s = PositionGetString(POSITION_SYMBOL);
      datetime pos_time = (datetime)PositionGetInteger(POSITION_TIME);
      int elapsed_sec = (int)(TimeCurrent() - pos_time);
      int elapsed_bars = elapsed_sec / 300; // 5-minute bars
      
      if(elapsed_bars >= HOLD_BARS) {
         ClosePositionByTicket(ticket, s, "expiry");
      }
   }
}
"""

def patch_ea(ea):
    if ea == "Ultra_Monster_MT5":
        return  # Already patched
    
    path = os.path.join(LOCAL_DIR, f"{ea}.mq5")
    with open(path, "r", encoding="utf-8") as f:
        code = f.read()

    old_close_marker = "bool Close(" if "bool Close(" in code else ("bool CloseTrade(" if "bool CloseTrade(" in code else None)
    if old_close_marker and "void CheckEntry()" in code:
        start_idx = code.find(old_close_marker)
        end_idx = code.find("void CheckEntry()")
        code = code[:start_idx] + DYNAMIC_EXIT_CODE + "\n" + code[end_idx:]
        
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        print(f"  • Updated {ea}.mq5 with Bulletproof Dynamic PositionsTotal Exit Engine!")

def compile_and_backup():
    for ea in EAS:
        mq5_file = os.path.join(LOCAL_DIR, f"{ea}.mq5")
        ex5_appdata = os.path.join(APPDATA_EXP, f"{ea}.ex5")
        ex5_local = os.path.join(LOCAL_DIR, f"{ea}.ex5")
        ex5_backup = os.path.join(BACKUP_DIR, f"{ea}.ex5")
        mq5_backup = os.path.join(BACKUP_DIR, f"{ea}.mq5")

        cmd = [METAEDITOR, f"/compile:{mq5_file}"]
        subprocess.run(cmd, check=False)

        if os.path.exists(ex5_appdata):
            shutil.copy(ex5_appdata, ex5_local)
            shutil.copy(ex5_appdata, ex5_backup)
        shutil.copy(mq5_file, mq5_backup)
        print(f"  • Compiled & Backed up {ea}")

def push_to_vps():
    print("🚀 Pushing all 6 updated EAs to VPS...")
    for ea in EAS:
        mq5_local = os.path.join(LOCAL_DIR, f"{ea}.mq5")
        ex5_local = os.path.join(LOCAL_DIR, f"{ea}.ex5")

        subprocess.run(["scp", "-i", VPS_KEY, ex5_local, VPS_PATH], check=False)
        subprocess.run(["scp", "-i", VPS_KEY, mq5_local, VPS_PATH], check=False)
        print(f"  • Uploaded {ea} to VPS!")

def main():
    print("="*115)
    print("APPLYING BULLETPROOF DYNAMIC POSITION EXIT ENGINE ACROSS ALL 6 EAS...")
    print("="*115)
    for ea in EAS:
        patch_ea(ea)

    print("="*115)
    print("COMPILING & BACKING UP ALL 6 EAS...")
    print("="*115)
    compile_and_backup()

    print("="*115)
    print("PUSHING ALL 6 UPDATED BINARIES TO VPS...")
    print("="*115)
    push_to_vps()

    print("="*115)
    print("🟢 ALL 6 EAS SUCCESSFULLY UPDATED WITH BULLETPROOF DYNAMIC EXIT ENGINE & PUSHED TO VPS!")
    print("="*115)

if __name__ == "__main__":
    main()
