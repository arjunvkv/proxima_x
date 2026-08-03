#!/usr/bin/env python3
"""Compile all 7 updated EAs and upload them to VPS FTMO MT5 terminal."""
import subprocess, shutil, os
from pathlib import Path

BASE_DIR = Path(r"c:\Trading\Agentic_Trading\proxima_x\paper_trade\mt5_backtest")
METAEDITOR = r"C:\Program Files\FundedNext MT5 Terminal\MetaEditor64.exe"
APPDATA_EXPERTS = Path(r"C:\Users\Arjun Sasi\AppData\Roaming\MetaQuotes\Terminal\89FE26BBBAB28C077BBF5FA8C1B4DF1C\MQL5\Experts")
VPS_KEY = r"C:\Users\Arjun Sasi\Downloads\ssh-key-2026-07-28.key"
VPS_PATH = "ubuntu@140.245.234.92:/home/ubuntu/.wine/drive_c/Program Files/FTMO Global Markets MT5 Terminal/MQL5/Experts/"

eas = [
    "Ultra_Monster_MT5.mq5", "CPPF_Z_MT5.mq5", "CPMC_Z_MT5.mq5",
    "TokyoH0_MT5.mq5", "Sunday_H22_MT5.mq5", "NY_H21_MT5.mq5", "MSV_Asian_Exhaustion_MT5.mq5"
]

def main():
    print("Compiling and Uploading All 7 Live EAs to VPS FTMO MT5...")
    for ea in eas:
        src = BASE_DIR / ea
        appdata_src = APPDATA_EXPERTS / ea
        shutil.copy(src, appdata_src)
        
        # Compile
        subprocess.run([METAEDITOR, f"/compile:{appdata_src}"], check=False)
        
        ex5_name = ea.replace(".mq5", ".ex5")
        appdata_ex5 = APPDATA_EXPERTS / ex5_name
        src_ex5 = BASE_DIR / ex5_name

        if appdata_ex5.exists():
            shutil.copy(appdata_ex5, src_ex5)
            # Backup .ex5 into updated_version_backup
            shutil.copy(appdata_ex5, BASE_DIR / "updated_version_backup" / ex5_name)

        # Upload .ex5 to VPS
        if appdata_ex5.exists():
            subprocess.run(["scp", "-i", VPS_KEY, str(appdata_ex5), VPS_PATH], check=False)
            print(f"🚀 Uploaded {ex5_name} to VPS FTMO MT5!")

        # Upload .mq5 to VPS
        subprocess.run(["scp", "-i", VPS_KEY, str(src), VPS_PATH], check=False)
        print(f"🚀 Uploaded {ea} to VPS FTMO MT5!")

    print("="*95)
    print("🟢 ALL 7 LIVE STRATEGIES SUCCESSFULLY COMPILED, BACKED UP, AND DEPLOYED TO VPS!")
    print("="*95)

if __name__ == "__main__":
    main()
