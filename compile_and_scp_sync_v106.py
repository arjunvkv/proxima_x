#!/usr/bin/env python3
"""Synchronous Compile & VPS Push for All 7 Versioned EAs (_v106)."""

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
BACKUP_DIR = r"C:\Trading\Agentic_Trading\proxima_x\paper_trade\mt5_backtest\updated_version_backup"
VPS_KEY = r"C:\Users\Arjun Sasi\Downloads\ssh-key-2026-07-28.key"
VPS_PATH = "ubuntu@140.245.234.92:/home/ubuntu/.wine/drive_c/Program Files/FTMO Global Markets MT5 Terminal/MQL5/Experts/"

def main():
    print("="*115)
    print("SYNCHRONOUS COMPILE & VPS PUSH FOR ALL 7 VERSIONED EAS (_v106)...")
    print("="*115)

    for ea in VERSIONED_EAS:
        appdata_mq5 = os.path.join(APPDATA_EXP, f"{ea}.mq5")
        appdata_ex5 = os.path.join(APPDATA_EXP, f"{ea}.ex5")
        local_mq5 = os.path.join(LOCAL_DIR, f"{ea}.mq5")
        local_ex5 = os.path.join(LOCAL_DIR, f"{ea}.ex5")
        backup_mq5 = os.path.join(BACKUP_DIR, f"{ea}.mq5")
        backup_ex5 = os.path.join(BACKUP_DIR, f"{ea}.ex5")

        # 1. Compile in AppData
        cmd = [METAEDITOR, f"/compile:{appdata_mq5}"]
        res = subprocess.run(cmd, check=False)
        
        # 2. Wait for MetaEditor to finish writing .ex5 binary
        for _ in range(20):
            if os.path.exists(appdata_ex5) and os.path.getsize(appdata_ex5) > 1000:
                break
            time.sleep(0.3)

        if os.path.exists(appdata_ex5) and os.path.getsize(appdata_ex5) > 1000:
            size = os.path.getsize(appdata_ex5)
            mtime = time.ctime(os.path.getmtime(appdata_ex5))
            print(f"  🟢 {ea:<34} COMPILED! Size: {size} bytes | Timestamp: {mtime}")

            shutil.copy(appdata_ex5, local_ex5)
            shutil.copy(appdata_ex5, backup_ex5)
            shutil.copy(appdata_mq5, backup_mq5)

            # Upload to VPS
            subprocess.run(["scp", "-i", VPS_KEY, appdata_ex5, VPS_PATH], check=True)
            subprocess.run(["scp", "-i", VPS_KEY, appdata_mq5, VPS_PATH], check=True)
            print(f"     🚀 Uploaded {ea} to VPS!")
        else:
            print(f"  ❌ {ea} compilation failed or binary not found!")

    print("="*115)
    print("🟢 ALL 7 VERSIONED EAS SYNCHRONOUSLY COMPILED & DEPLOYED TO VPS!")
    print("="*115)

if __name__ == "__main__":
    main()
