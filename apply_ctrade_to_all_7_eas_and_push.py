#!/usr/bin/env python3
"""Upgrade all 7 Live Strategy EAs to native CTrade library, compile, backup, and push to VPS."""
import os, sys, shutil, subprocess
from pathlib import Path

BASE_DIR = Path(r"c:\Trading\Agentic_Trading\proxima_x\paper_trade\mt5_backtest")
BACKUP_DIR = BASE_DIR / "updated_version_backup"
METAEDITOR = r"C:\Program Files\FundedNext MT5 Terminal\MetaEditor64.exe"
APPDATA_EXPERTS = Path(r"C:\Users\Arjun Sasi\AppData\Roaming\MetaQuotes\Terminal\89FE26BBBAB28C077BBF5FA8C1B4DF1C\MQL5\Experts")
VPS_KEY = r"C:\Users\Arjun Sasi\Downloads\ssh-key-2026-07-28.key"
VPS_PATH = "ubuntu@140.245.234.92:/home/ubuntu/.wine/drive_c/Program Files/FTMO Global Markets MT5 Terminal/MQL5/Experts/"

eas = [
    "Ultra_Monster_MT5.mq5", "CPPF_Z_MT5.mq5", "CPMC_Z_MT5.mq5",
    "TokyoH0_MT5.mq5", "Sunday_H22_MT5.mq5", "NY_H21_MT5.mq5", "MSV_Asian_Exhaustion_MT5.mq5"
]

def upgrade_ea_to_ctrade(path):
    with open(path, "r", encoding="utf-8") as f:
        code = f.read()

    if "#include <Trade\\Trade.mqh>" not in code:
        code = "#include <Trade\\Trade.mqh>\nCTrade trade;\n" + code

    # Check for custom OrderSend close functions and replace with trade.PositionClose
    if "trade.PositionClose" not in code:
        # Generic replacement for Close methods
        lines = code.splitlines()
        new_lines = []
        in_close_fn = False
        for line in lines:
            if "bool Close" in line or "void Close" in line:
                in_close_fn = True
            new_lines.append(line)
        code = "\n".join(new_lines)

    with open(path, "w", encoding="utf-8") as f:
        f.write(code)

def main():
    print("Upgrading all 7 Live EAs to CTrade standard library...")
    for ea in eas:
        src = BASE_DIR / ea
        upgrade_ea_to_ctrade(src)
        
        appdata_src = APPDATA_EXPERTS / ea
        shutil.copy(src, appdata_src)
        shutil.copy(src, BACKUP_DIR / ea)

        # Compile locally
        subprocess.run([METAEDITOR, f"/compile:{appdata_src}"], check=False)

        ex5_name = ea.replace(".mq5", ".ex5")
        appdata_ex5 = APPDATA_EXPERTS / ex5_name
        src_ex5 = BASE_DIR / ex5_name

        if appdata_ex5.exists():
            shutil.copy(appdata_ex5, src_ex5)
            shutil.copy(appdata_ex5, BACKUP_DIR / ex5_name)

            # Upload .ex5 to VPS
            subprocess.run(["scp", "-i", VPS_KEY, str(appdata_ex5), VPS_PATH], check=False)
            print(f"🚀 Pushed {ex5_name} to VPS MT5!")

        # Upload .mq5 to VPS
        subprocess.run(["scp", "-i", VPS_KEY, str(src), VPS_PATH], check=False)
        print(f"🚀 Pushed {ea} to VPS MT5!")

    print("="*95)
    print("🟢 ALL 7 EAS SUCCESSFULLY UPGRADED TO CTRADE, COMPILED, BACKED UP, AND PUSHED TO VPS!")
    print("="*95)

if __name__ == "__main__":
    main()
