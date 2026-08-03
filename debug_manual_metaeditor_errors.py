#!/usr/bin/env python3
"""Debug MetaEditor manual compilation errors across all 7 EAs in local directory and AppData."""

import os, subprocess, shutil

METAEDITOR = r"C:\Program Files\FundedNext MT5 Terminal\MetaEditor64.exe"
LOCAL_DIR = r"C:\Trading\Agentic_Trading\proxima_x\paper_trade\mt5_backtest"
APPDATA_EXP = r"C:\Users\Arjun Sasi\AppData\Roaming\MetaQuotes\Terminal\89FE26BBBAB28C077BBF5FA8C1B4DF1C\MQL5\Experts"

EAS = [
    "Test_Min_Fire_MT5",
    "Ultra_Monster_MT5",
    "TokyoH0_MT5",
    "CPPF_Z_MT5",
    "CPMC_Z_MT5",
    "NY_H21_MT5",
    "MSV_Asian_Exhaustion_MT5"
]

def compile_and_get_log(ea, target_dir):
    mq5_file = os.path.join(target_dir, f"{ea}.mq5")
    log_file = os.path.join(target_dir, f"{ea}_compile_test.log")

    if os.path.exists(log_file):
        os.remove(log_file)

    cmd = [METAEDITOR, f"/compile:{mq5_file}", f"/log:{log_file}"]
    res = subprocess.run(cmd, capture_output=True, text=True)

    log_content = ""
    if os.path.exists(log_file):
        try:
            with open(log_file, "r", encoding="utf-16le", errors="ignore") as f:
                log_content = f.read().strip()
        except Exception:
            with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                log_content = f.read().strip()

    return res.returncode, log_content

def main():
    print("="*115)
    print("DEBUGGING METAEDITOR MANUAL COMPILATION ERRORS ACROSS ALL 7 EAS...")
    print("="*115)

    for ea in EAS:
        print(f"\n--- TESTING {ea}.mq5 (LOCAL DIRECTORY) ---")
        ret_local, log_local = compile_and_get_log(ea, LOCAL_DIR)
        print(f"Retcode: {ret_local}")
        if log_local:
            print("Log Output:")
            for l in log_local.splitlines():
                print("  ", l)
        else:
            print("  (No log output produced)")

        print(f"\n--- TESTING {ea}.mq5 (APPDATA DIRECTORY) ---")
        ret_app, log_app = compile_and_get_log(ea, APPDATA_EXP)
        print(f"Retcode: {ret_app}")
        if log_app:
            print("Log Output:")
            for l in log_app.splitlines():
                print("  ", l)
        else:
            print("  (No log output produced)")

    print("="*115)

if __name__ == "__main__":
    main()
