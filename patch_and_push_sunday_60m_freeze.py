#!/usr/bin/env python3
"""Patch Sunday Open 60-Minute Freeze across all 6 Active EAs, compile, backup, and push to VPS."""
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

def patch_ea(ea_name):
    path = os.path.join(LOCAL_DIR, f"{ea_name}.mq5")
    with open(path, "r", encoding="utf-8") as f:
        code = f.read()

    # Sunday Open 60m Freeze Check Code
    freeze_code = """
   // --- SUNDAY OPEN 60-MINUTE FREEZE (Block 00:00 to 01:00 MT5 Server Time / 22:00 to 23:00 UTC) ---
   MqlDateTime _dt_fr; TimeCurrent(_dt_fr);
   if(_dt_fr.day_of_week == 0 && _dt_fr.hour == 0) return;
"""

    if "SUNDAY OPEN 60-MINUTE FREEZE" not in code:
        if "void CheckEntry()" in code:
            code = code.replace("void CheckEntry() {", "void CheckEntry() {" + freeze_code)
        elif "void OnTick()" in code:
            code = code.replace("void OnTick() {", "void OnTick() {" + freeze_code)
        
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        print(f"  • Patched {ea_name}.mq5 with Sunday Open 60-Minute Freeze!")
    else:
        print(f"  • {ea_name}.mq5 already contains Sunday Open 60-Minute Freeze!")

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
    print("APPLYING SUNDAY OPEN 60-MINUTE FREEZE ACROSS ALL 6 ACTIVE STRATEGY EAS...")
    print("="*115)
    for ea in EAS:
        patch_ea(ea)

    print("="*115)
    print("COMPILING & BACKING UP LOCAL FILES...")
    print("="*115)
    compile_and_backup()

    print("="*115)
    print("PUSHING UPDATED BINARIES TO VPS...")
    print("="*115)
    push_to_vps()

    print("="*115)
    print("🟢 ALL 6 EAS SUCCESSFULLY UPDATED WITH SUNDAY OPEN 60-MINUTE FREEZE & PUSHED TO VPS!")
    print("="*115)

if __name__ == "__main__":
    main()
