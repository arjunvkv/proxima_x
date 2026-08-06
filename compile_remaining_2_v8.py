import os
import subprocess
import shutil
import time
import re

METAEDITOR = r"C:\Program Files\FundedNext MT5 Terminal\MetaEditor64.exe"
APPDATA_EXP = r"C:\Users\Arjun Sasi\AppData\Roaming\MetaQuotes\Terminal\89FE26BBBAB28C077BBF5FA8C1B4DF1C\MQL5\Experts"
LOCAL_DIR = r"C:\Trading\Agentic_Trading\proxima_x\paper_trade\mt5_backtest"
BACKUP_DIR = r"C:\Trading\Agentic_Trading\proxima_x\paper_trade\mt5_backtest\updated_version_backup"
VAULT_SOURCE_DIR = r"C:\Trading\Agentic_Trading\proxima_x\PROVEN_7_STRATEGY_PORTFOLIO_VAULT\source_eas"
VPS_KEY = r"C:\Users\Arjun Sasi\Downloads\ssh-key-2026-07-28.key"
VPS_PATH = "ubuntu@140.245.234.92:/home/ubuntu/.wine/drive_c/Program Files/FTMO Global Markets MT5 Terminal/MQL5/Experts/"

targets = [
    ("NY_H21_MT5_v8", "NY_H21_MT5_v107.mq5", "1.50"),
    ("CPMC_Z_MT5_v8", "CPMC_Z_MT5_v107.mq5", "1.40"),
]

for name, source, lot in targets:
    src_path = os.path.join(VAULT_SOURCE_DIR, source)
    with open(src_path, "r", encoding="utf-8") as f:
        code = f.read()

    code = code.replace('#property version   "1.07"', '#property version   "8.00"')
    code = code.replace('#property version   "1.06"', '#property version   "8.00"')
    code = re.sub(r'input double\s+BASE_LOT\s*=\s*[\d\.]+;', f'input double   BASE_LOT            = {lot};', code)
    code = code.replace('v107', 'v8')
    code = code.replace('v106', 'v8')

    appdata_mq5 = os.path.join(APPDATA_EXP, f"{name}.mq5")
    appdata_ex5 = os.path.join(APPDATA_EXP, f"{name}.ex5")
    local_mq5   = os.path.join(LOCAL_DIR, f"{name}.mq5")
    local_ex5   = os.path.join(LOCAL_DIR, f"{name}.ex5")
    backup_mq5  = os.path.join(BACKUP_DIR, f"{name}.mq5")
    backup_ex5  = os.path.join(BACKUP_DIR, f"{name}.ex5")

    with open(appdata_mq5, "w", encoding="utf-8") as f: f.write(code)
    with open(local_mq5, "w", encoding="utf-8") as f: f.write(code)

    print(f"Compiling {name} (Lot: {lot}L)...")
    subprocess.run([METAEDITOR, f"/compile:{appdata_mq5}", "/log"], check=False)
    time.sleep(3.5)

    if os.path.exists(appdata_ex5):
        size = os.path.getsize(appdata_ex5)
        print(f"  🟢 {name} COMPILED SUCCESS! Size: {size:,} bytes | Lot: {lot}L")
        shutil.copy(appdata_ex5, local_ex5)
        shutil.copy(appdata_ex5, backup_ex5)
        shutil.copy(appdata_mq5, backup_mq5)
        subprocess.run(["scp", "-i", VPS_KEY, appdata_ex5, VPS_PATH], check=False)
        subprocess.run(["scp", "-i", VPS_KEY, appdata_mq5, VPS_PATH], check=False)
    else:
        print(f"  ❌ Compilation failed for {name}")
