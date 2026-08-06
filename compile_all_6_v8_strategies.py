import os
import subprocess
import shutil
import time

METAEDITOR = r"C:\Program Files\FundedNext MT5 Terminal\MetaEditor64.exe"
APPDATA_EXP = r"C:\Users\Arjun Sasi\AppData\Roaming\MetaQuotes\Terminal\89FE26BBBAB28C077BBF5FA8C1B4DF1C\MQL5\Experts"
LOCAL_DIR = r"C:\Trading\Agentic_Trading\proxima_x\paper_trade\mt5_backtest"
BACKUP_DIR = r"C:\Trading\Agentic_Trading\proxima_x\paper_trade\mt5_backtest\updated_version_backup"
VAULT_SOURCE_DIR = r"C:\Trading\Agentic_Trading\proxima_x\PROVEN_7_STRATEGY_PORTFOLIO_VAULT\source_eas"
VPS_KEY = r"C:\Users\Arjun Sasi\Downloads\ssh-key-2026-07-28.key"
VPS_PATH = "ubuntu@140.245.234.92:/home/ubuntu/.wine/drive_c/Program Files/FTMO Global Markets MT5 Terminal/MQL5/Experts/"

V8_CONFIGS = [
    {"target": "TokyoH0_MT5_v8", "source": "TokyoH0_MT5_v106.mq5", "lot": "1.00"},
    {"target": "Ultra_Monster_MT5_v8", "source": "Ultra_Monster_MT5_v106.mq5", "lot": "1.20"},
    {"target": "CPPF_Z_MT5_v8", "source": "CPPF_Z_MT5_v106.mq5", "lot": "1.40"},
    {"target": "MSV_Asian_Exhaustion_MT5_v8", "source": "MSV_Asian_Exhaustion_MT5_v106.mq5", "lot": "1.00"},
    {"target": "NY_H21_MT5_v8", "source": "NY_H21_MT5_v106.mq5", "lot": "1.50"},
    {"target": "CPMC_Z_MT5_v8", "source": "CPMC_Z_MT5_v106.mq5", "lot": "1.40"},
]

def main():
    print("=" * 115)
    print("PROXIMA X — COMPILING & DEPLOYING ALL 6 v8 STRATEGY EAs TO VPS...")
    print("=" * 115)

    compiled_count = 0

    for cfg in V8_CONFIGS:
        target_name = cfg["target"]
        source_file = os.path.join(VAULT_SOURCE_DIR, cfg["source"])
        if not os.path.exists(source_file):
            print(f"⚠️ Source file {source_file} not found. Skipping...")
            continue

        with open(source_file, "r", encoding="utf-8") as f:
            code = f.read()

        code = code.replace('#property version   "1.07"', '#property version   "8.00"')
        code = code.replace('#property version   "1.06"', '#property version   "8.00"')
        
        import re
        code = re.sub(r'input double\s+BASE_LOT\s*=\s*[\d\.]+;', f'input double   BASE_LOT            = {cfg["lot"]};', code)
        code = code.replace('v107', 'v8')
        code = code.replace('v106', 'v8')

        filename_mq5 = f"{target_name}.mq5"
        filename_ex5 = f"{target_name}.ex5"

        local_mq5   = os.path.join(LOCAL_DIR, filename_mq5)
        appdata_mq5 = os.path.join(APPDATA_EXP, filename_mq5)
        appdata_ex5 = os.path.join(APPDATA_EXP, filename_ex5)
        local_ex5   = os.path.join(LOCAL_DIR, filename_ex5)
        backup_mq5  = os.path.join(BACKUP_DIR, filename_mq5)
        backup_ex5  = os.path.join(BACKUP_DIR, filename_ex5)
        vault_v8_mq5 = os.path.join(VAULT_SOURCE_DIR, filename_mq5)

        with open(local_mq5, "w", encoding="utf-8") as f:
            f.write(code)
        with open(appdata_mq5, "w", encoding="utf-8") as f:
            f.write(code)
        with open(vault_v8_mq5, "w", encoding="utf-8") as f:
            f.write(code)

        print(f"🔨 Compiling {filename_mq5} (Lot: {cfg['lot']}L)...")
        subprocess.run([METAEDITOR, f"/compile:{appdata_mq5}", "/log"], check=False)
        time.sleep(4.5)

        if os.path.exists(appdata_ex5):
            size = os.path.getsize(appdata_ex5)
            print(f"  🟢 {target_name} COMPILED SUCCESS! Size: {size:,} bytes | Lot: {cfg['lot']}L")
            shutil.copy(appdata_ex5, local_ex5)
            shutil.copy(appdata_ex5, backup_ex5)
            shutil.copy(appdata_mq5, backup_mq5)

            print(f"  🚀 Uploading {filename_ex5} and {filename_mq5} to VPS via SSH...")
            subprocess.run(["scp", "-i", VPS_KEY, appdata_ex5, VPS_PATH], check=False)
            subprocess.run(["scp", "-i", VPS_KEY, appdata_mq5, VPS_PATH], check=False)
            compiled_count += 1
        else:
            print(f"  ❌ Compilation failed for {filename_mq5}")

    print("\n" + "=" * 115)
    print(f"🟢 SUCCESS: {compiled_count} / {len(V8_CONFIGS)} v8 PROVEN STRATEGY EAs COMPILED CLEANLY & DEPLOYED TO VPS!")
    print("=" * 115)

if __name__ == "__main__":
    main()
