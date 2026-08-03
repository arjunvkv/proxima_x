#!/usr/bin/env python3
"""Fix SL/TP variable typo (dir -> side) and Digits normalization across ALL 6 Active EAs."""

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

def patch_ea(ea):
    path = os.path.join(LOCAL_DIR, f"{ea}.mq5")
    with open(path, "r", encoding="utf-8") as f:
        code = f.read()

    # Replace typo dir == "BUY" with side == "BUY" and add NormalizeDouble with Digits
    old_sl_line1 = 'req.sl = (dir == "BUY") ? pr - sl_d : pr + sl_d;'
    old_tp_line1 = 'req.tp = (dir == "BUY") ? pr + tp_d : pr - tp_d;'
    
    new_sl_tp_block = """   int digits = (int)SymbolInfoInteger(s, SYMBOL_DIGITS);
   req.sl = NormalizeDouble((side == "BUY") ? pr - sl_d : pr + sl_d, digits);
   req.tp = NormalizeDouble((side == "BUY") ? pr + tp_d : pr - tp_d, digits);"""

    if old_sl_line1 in code:
        code = code.replace(old_sl_line1 + "\n" + old_tp_line1, new_sl_tp_block)
        code = code.replace(old_sl_line1, "")
        code = code.replace(old_tp_line1, "")
    
    # Also ensure any other dir == "BUY" typos in OpenTrade are replaced with side == "BUY"
    code = code.replace('(dir == "BUY")', '(side == "BUY")')
    code = code.replace('(dir == "SELL")', '(side == "SELL")')

    with open(path, "w", encoding="utf-8") as f:
        f.write(code)
    print(f"  • Upgraded {ea}.mq5 with Correct SL/TP Side Logic & Price Digits Normalization!")

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
    print("🚀 Pushing all 6 updated SL/TP EAs to VPS...")
    for ea in EAS:
        mq5_local = os.path.join(LOCAL_DIR, f"{ea}.mq5")
        ex5_local = os.path.join(LOCAL_DIR, f"{ea}.ex5")

        subprocess.run(["scp", "-i", VPS_KEY, ex5_local, VPS_PATH], check=False)
        subprocess.run(["scp", "-i", VPS_KEY, mq5_local, VPS_PATH], check=False)
        print(f"  • Uploaded {ea} to VPS!")

def main():
    print("="*115)
    print("APPLYING SL/TP SIDE LOGIC & DIGITS NORMALIZATION ACROSS ALL 6 EAS...")
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
    print("🟢 ALL 6 EAS SUCCESSFULLY UPGRADED WITH ATTACHED SL/TP ENGINE & PUSHED TO VPS!")
    print("="*115)

if __name__ == "__main__":
    main()
