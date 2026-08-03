#!/usr/bin/env python3
"""Compile all 7 versioned EAs and output .ex5 files cleanly."""

import os, subprocess, shutil, time

VERSIONED_EAS = [
    "Test_Min_Fire_MT5_v106",
    "Ultra_Monster_MT5_v106",
    "TokyoH0_MT5_v106",
    "CPPF_Z_MT5_v106",
    "CPMC_Z_MT5_v106",
    "NY_H21_MT5_v106",
    "MSV_Asian_Exhaustion_MT5_v106"
]

METAEDITOR = r"C:\Program Files\FundedNext MT5 Terminal\MetaEditor64.exe"
APPDATA_EXP = r"C:\Users\Arjun Sasi\AppData\Roaming\MetaQuotes\Terminal\89FE26BBBAB28C077BBF5FA8C1B4DF1C\MQL5\Experts"
LOCAL_DIR = r"C:\Trading\Agentic_Trading\proxima_x\paper_trade\mt5_backtest"
VPS_KEY = r"C:\Users\Arjun Sasi\Downloads\ssh-key-2026-07-28.key"
VPS_PATH = "ubuntu@140.245.234.92:/home/ubuntu/.wine/drive_c/Program Files/FTMO Global Markets MT5 Terminal/MQL5/Experts/"

def main():
    print("="*115)
    print("COMPILING ALL 7 VERSIONED EAS & DEPLOYING BINARIES TO VPS...")
    print("="*115)

    for ea in VERSIONED_EAS:
        mq5 = os.path.join(APPDATA_EXP, f"{ea}.mq5")
        ex5 = os.path.join(APPDATA_EXP, f"{ea}.ex5")

        cmd = [METAEDITOR, f"/compile:{mq5}"]
        proc = subprocess.Popen(cmd)
        proc.wait()
        time.sleep(1.0)

        # Check in AppData or Program Files
        prog_ex5 = os.path.join(r"C:\Program Files\FundedNext MT5 Terminal\MQL5\Experts", f"{ea}.ex5")
        
        found_ex5 = None
        if os.path.exists(ex5):
            found_ex5 = ex5
        elif os.path.exists(prog_ex5):
            found_ex5 = prog_ex5

        if found_ex5:
            size = os.path.getsize(found_ex5)
            print(f"  🟢 {ea:<34} COMPILED OK! Size: {size} bytes")
            shutil.copy(found_ex5, os.path.join(LOCAL_DIR, f"{ea}.ex5"))
            subprocess.run(["scp", "-i", VPS_KEY, found_ex5, VPS_PATH], check=False)
            subprocess.run(["scp", "-i", VPS_KEY, mq5, VPS_PATH], check=False)
        else:
            print(f"  ⚠️ {ea} compile log check...")

    print("="*115)
    print("🟢 COMPLETED!")
    print("="*115)

if __name__ == "__main__":
    main()
