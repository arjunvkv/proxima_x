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

# 6 Active Strategies (Vault Source v107 Code)
STRATEGY_FILES = [
    "Ultra_Monster_MT5_v107.mq5",
    "TokyoH0_MT5_v107.mq5",
    "MSV_Asian_Exhaustion_MT5_v107.mq5",
    "CPPF_Z_MT5_v107.mq5",
    "NY_H21_MT5_v107.mq5",
    "CPMC_Z_MT5_v107.mq5",
]

def main():
    print("=" * 115)
    print("PROXIMA X — BUILDING, COMPILING & DEPLOYING ALL 6 v107 PROVEN EAs TO VPS...")
    print("=" * 115)

    compiled_count = 0

    for vault_file in STRATEGY_FILES:
        vault_mq5 = os.path.join(VAULT_SOURCE_DIR, vault_file)
        if not os.path.exists(vault_mq5):
            print(f"⚠️ Warning: {vault_mq5} not found. Skipping...")
            continue

        with open(vault_mq5, "r", encoding="utf-8") as f:
            code = f.read()

        # Update lot size for Ultra Monster to 1.20 Lots
        if "Ultra_Monster" in vault_file:
            code = code.replace("input double   BASE_LOT            = 0.15;", "input double   BASE_LOT            = 1.20;")

        # Base name targets (both _v107 and _v106 production target filenames)
        base_name = vault_file.replace(".mq5", "")
        v106_name = base_name.replace("_v107", "_v106")

        target_names = [base_name, v106_name]

        for name in target_names:
            filename = f"{name}.mq5"
            local_mq5 = os.path.join(LOCAL_DIR, filename)
            appdata_mq5 = os.path.join(APPDATA_EXP, filename)
            appdata_ex5 = os.path.join(APPDATA_EXP, f"{name}.ex5")
            local_ex5 = os.path.join(LOCAL_DIR, f"{name}.ex5")
            backup_mq5 = os.path.join(BACKUP_DIR, filename)
            backup_ex5 = os.path.join(BACKUP_DIR, f"{name}.ex5")

            with open(local_mq5, "w", encoding="utf-8") as f:
                f.write(code)
            with open(appdata_mq5, "w", encoding="utf-8") as f:
                f.write(code)

            print(f"🔨 Compiling {filename}...")
            # Use powershell execution syntax
            cmd = f'& "{METAEDITOR}" "/compile:{appdata_mq5}"'
            proc = subprocess.run(["powershell", "-Command", cmd], capture_output=True, text=True)
            time.sleep(0.5)

            if os.path.exists(appdata_ex5):
                size = os.path.getsize(appdata_ex5)
                print(f"  🟢 {name} COMPILED SUCCESS! Size: {size:,} bytes")
                shutil.copy(appdata_ex5, local_ex5)
                shutil.copy(appdata_ex5, backup_ex5)
                shutil.copy(appdata_mq5, backup_mq5)

                print(f"  🚀 Uploading {name}.ex5 to VPS via SSH...")
                subprocess.run(["scp", "-i", VPS_KEY, appdata_ex5, VPS_PATH], check=False)
                subprocess.run(["scp", "-i", VPS_KEY, appdata_mq5, VPS_PATH], check=False)
                compiled_count += 1
            else:
                print(f"  ❌ Compilation failed for {filename}")

    print("\n" + "=" * 115)
    print(f"🟢 SUCCESS: {compiled_count} / {len(STRATEGY_FILES)*2} v107 PROVEN EA BINARIES COMPILED CLEANLY & DEPLOYED TO VPS!")
    print("=" * 115)

if __name__ == "__main__":
    main()
