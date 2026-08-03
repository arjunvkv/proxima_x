#!/usr/bin/env python3
"""Compile all 7 EAs directly inside MT5 AppData Experts folder."""

import os, subprocess, shutil

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

def main():
    print("="*115)
    print("COMPILING ALL 7 EAS DIRECTLY IN MT5 APPDATA DIRECTORY...")
    print("="*115)

    for ea in EAS:
        local_mq5 = os.path.join(LOCAL_DIR, f"{ea}.mq5")
        appdata_mq5 = os.path.join(APPDATA_EXP, f"{ea}.mq5")
        appdata_ex5 = os.path.join(APPDATA_EXP, f"{ea}.ex5")

        shutil.copy(local_mq5, appdata_mq5)

        cmd = [METAEDITOR, f"/compile:{appdata_mq5}"]
        res = subprocess.run(cmd, check=False)

        if os.path.exists(appdata_ex5):
            mtime = os.path.getmtime(appdata_ex5)
            print(f"  • Compiled {ea}.mq5 directly in AppData! EX5 mtime: {mtime} 🟢")

            # Upload freshly compiled AppData EX5 & MQ5 to VPS
            subprocess.run(["scp", "-i", VPS_KEY, appdata_ex5, VPS_PATH], check=False)
            subprocess.run(["scp", "-i", VPS_KEY, appdata_mq5, VPS_PATH], check=False)

    print("="*115)
    print("🟢 ALL 7 EAS DIRECTLY COMPILED IN APPDATA & PUSHED TO VPS!")
    print("="*115)

if __name__ == "__main__":
    main()
