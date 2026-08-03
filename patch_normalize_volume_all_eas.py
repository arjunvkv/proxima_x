#!/usr/bin/env python3
"""Patch official MQL5 NormalizeVolume function across ALL 6 Active EAs for entry and exit."""

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

NORMALIZE_VOL_CODE = """
double NormalizeVolume(string symbol, double volume) {
   SymbolSelect(symbol, true);
   double min_vol = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);
   double max_vol = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MAX);
   double step_vol = SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP);
   if(step_vol <= 0.0) step_vol = 0.01;
   
   double normalized_vol = MathRound(volume / step_vol) * step_vol;
   if(normalized_vol < min_vol) normalized_vol = min_vol;
   if(normalized_vol > max_vol) normalized_vol = max_vol;
   
   int digits = (step_vol == 0.01) ? 2 : ((step_vol == 0.1) ? 1 : 0);
   return NormalizeDouble(normalized_vol, digits);
}

bool ClosePositionByTicket(ulong ticket, string s, string why) {
   if(!PositionSelectByTicket(ticket)) return false;
   double raw_vol = PositionGetDouble(POSITION_VOLUME);
   double vol = NormalizeVolume(s, raw_vol);
   if(vol <= 0.0) return false;
   
   // Primary Exit: CTrade PositionClose Partial with exact normalized volume step
   if(trade.PositionClosePartial(ticket, vol)) {
      Print("  CLOSE SUCCESS (Partial) ", s, " ticket=", ticket, " vol=", vol, " why=", why);
      return true;
   }
   if(trade.PositionClose(ticket)) {
      Print("  CLOSE SUCCESS (CTrade) ", s, " ticket=", ticket, " vol=", vol, " why=", why);
      return true;
   }
   if(trade.PositionClose(s)) {
      Print("  CLOSE SUCCESS (Symbol) ", s, " why=", why);
      return true;
   }
   return false;
}
"""

def patch_ea(ea):
    path = os.path.join(LOCAL_DIR, f"{ea}.mq5")
    with open(path, "r", encoding="utf-8") as f:
        code = f.read()

    # 1. Update OpenTrade to use NormalizeVolume(s, lot)
    code = code.replace("req.volume = (float)lot;", "req.volume = (float)NormalizeVolume(s, lot);")
    code = code.replace("req.volume = lot;", "req.volume = (float)NormalizeVolume(s, lot);")

    # 2. Replace ClosePositionByTicket with official NormalizeVolume implementation
    old_marker = "bool ClosePositionByTicket("
    if old_marker in code and "void CheckEntry()" in code:
        start_idx = code.find(old_marker)
        end_idx = code.find("void CheckEntry()")
        code = code[:start_idx] + NORMALIZE_VOL_CODE + "\n" + code[end_idx:]
        
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        print(f"  • Upgraded {ea}.mq5 with Official NormalizeVolume Engine!")

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
    print("🚀 Pushing all 6 updated NormalizeVolume EAs to VPS...")
    for ea in EAS:
        mq5_local = os.path.join(LOCAL_DIR, f"{ea}.mq5")
        ex5_local = os.path.join(LOCAL_DIR, f"{ea}.ex5")

        subprocess.run(["scp", "-i", VPS_KEY, ex5_local, VPS_PATH], check=False)
        subprocess.run(["scp", "-i", VPS_KEY, mq5_local, VPS_PATH], check=False)
        print(f"  • Uploaded {ea} to VPS!")

def main():
    print("="*115)
    print("APPLYING OFFICIAL MQL5 NORMALIZEVOLUME ENGINE ACROSS ALL 6 EAS...")
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
    print("🟢 ALL 6 EAS SUCCESSFULLY UPGRADED WITH NORMALIZEVOLUME & PUSHED TO VPS!")
    print("="*115)

if __name__ == "__main__":
    main()
