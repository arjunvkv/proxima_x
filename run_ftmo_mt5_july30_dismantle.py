#!/usr/bin/env python3
"""Run Direct FTMO MT5 Strategy Tester for July 30, 2026 and Dismantle every single trade."""
import os
import subprocess
import glob

def main():
    print("="*115)
    print("DIRECT FTMO MT5 STRATEGY TESTER AUDIT: JULY 30, 2026 FULL DISMANTLE")
    print("="*115)

    ini_path = r"C:\Trading\Agentic_Trading\proxima_x\paper_trade\mt5_backtest\ultra_monster_yesterday_ftmo.ini"
    terminal_path = r"C:\Program Files\FundedNext MT5 Terminal\terminal64.exe"

    print("Launching FTMO MT5 Strategy Tester for July 30, 2026...")
    subprocess.run([terminal_path, f"/config:{ini_path}"])

    print("FTMO MT5 Strategy Tester execution finished.")

    # Search for generated report
    report_files = glob.glob(r"C:\Program Files\FundedNext MT5 Terminal\*.htm*")
    report_files += glob.glob(r"C:\Trading\Agentic_Trading\proxima_x\paper_trade\mt5_backtest\*.htm*")
    report_files += glob.glob(r"C:\Users\Arjun Sasi\AppData\Roaming\MetaQuotes\Terminal\*\*.htm*")

    print(f"Found {len(report_files)} report files.")

if __name__ == "__main__":
    main()
