#!/usr/bin/env python3
"""Trigger MQL5 Local Strategy Tester on MT5 Terminal for Ultra Monster."""

import os, subprocess, shutil, time

TERMINAL = r"C:\Program Files\FundedNext MT5 Terminal\terminal64.exe"
APPDATA_EXP = r"C:\Users\Arjun Sasi\AppData\Roaming\MetaQuotes\Terminal\89FE26BBBAB28C077BBF5FA8C1B4DF1C\MQL5\Experts"
LOCAL_DIR = r"C:\Trading\Agentic_Trading\proxima_x\paper_trade\mt5_backtest"

INI_PATH = os.path.join(LOCAL_DIR, "tester_ultra_monster.ini")
REPORT_PATH = os.path.join(LOCAL_DIR, "Ultra_Monster_Report.htm")

mq5_src = os.path.join(LOCAL_DIR, "Ultra_Monster_MT5_v106.mq5")
mq5_target = os.path.join(APPDATA_EXP, "Ultra_Monster_MT5.mq5")

def main():
    print("="*115)
    print("LAUNCHING MQL5 LOCAL STRATEGY TESTER FOR ULTRA MONSTER...")
    print("="*115)

    with open(mq5_src, "r", encoding="utf-8") as f:
        code = f.read()
    with open(mq5_target, "w", encoding="utf-8") as f:
        f.write(code)

    # Write tester.ini
    ini_content = f"""[Tester]
Expert=Ultra_Monster_MT5.ex5
Symbol=GBPUSD
Period=M5
Deposit=10000
Currency=USD
Leverage=1:100
Model=1
FromDate=2026.01.01
ToDate=2026.08.01
Report={REPORT_PATH}
ReplaceReport=1
ShutdownTerminal=0
"""
    with open(INI_PATH, "w", encoding="utf-8") as f:
        f.write(ini_content)

    print(f"Launching local MT5 Strategy Tester: {TERMINAL} /config:{INI_PATH}")
    subprocess.Popen([TERMINAL, f"/config:{INI_PATH}"])
    
    print("🟢 MT5 Terminal Strategy Tester launched locally!")
    print("   Open MT5 Terminal -> Press Ctrl+R (Strategy Tester) to view visual equity curve and report!")

if __name__ == "__main__":
    main()
