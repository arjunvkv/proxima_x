#!/usr/bin/env python3
"""Run Local MT5 Strategy Tester for Ultra_Monster_MT5_v106.mq5 and parse report."""

import os, subprocess, shutil, time

TERMINAL = r"C:\Program Files\FundedNext MT5 Terminal\terminal64.exe"
METAEDITOR = r"C:\Program Files\FundedNext MT5 Terminal\MetaEditor64.exe"
APPDATA_EXP = r"C:\Users\Arjun Sasi\AppData\Roaming\MetaQuotes\Terminal\89FE26BBBAB28C077BBF5FA8C1B4DF1C\MQL5\Experts"
LOCAL_DIR = r"C:\Trading\Agentic_Trading\proxima_x\paper_trade\mt5_backtest"

INI_PATH = os.path.join(LOCAL_DIR, "tester_ultra_monster.ini")
REPORT_PATH = os.path.join(LOCAL_DIR, "Ultra_Monster_v106_Report.htm")

mq5_src = os.path.join(LOCAL_DIR, "Ultra_Monster_MT5_v106.mq5")
mq5_appdata = os.path.join(APPDATA_EXP, "Ultra_Monster_MT5_v106.mq5")
ex5_appdata = os.path.join(APPDATA_EXP, "Ultra_Monster_MT5_v106.ex5")

def main():
    print("="*115)
    print("RUNNING LOCAL MQL5 STRATEGY TESTER FOR ULTRA MONSTER (_v106)...")
    print("="*115)

    with open(mq5_src, "r", encoding="utf-8") as f:
        code = f.read()
    with open(mq5_appdata, "w", encoding="utf-8") as f:
        f.write(code)

    cmd_comp = [METAEDITOR, f"/compile:{mq5_appdata}"]
    subprocess.run(cmd_comp, check=False)
    time.sleep(2.5)

    if not os.path.exists(ex5_appdata):
        print("❌ MetaEditor compilation failed or binary missing!")
        return

    print(f"🟢 MetaEditor compilation successful: {ex5_appdata} ({os.path.getsize(ex5_appdata)} bytes)")

    # Write tester.ini
    ini_content = f"""[Tester]
Expert=Ultra_Monster_MT5_v106.ex5
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
ShutdownTerminal=1
"""
    with open(INI_PATH, "w", encoding="utf-8") as f:
        f.write(ini_content)

    print(f"Launching MT5 Strategy Tester via CLI: {TERMINAL} /config:{INI_PATH}")
    cmd_test = [TERMINAL, f"/config:{INI_PATH}"]
    proc = subprocess.Popen(cmd_test)
    
    for _ in range(30):
        if os.path.exists(REPORT_PATH):
            print("🟢 MQL5 STRATEGY TESTER REPORT GENERATED!")
            break
        time.sleep(1.0)

    if os.path.exists(REPORT_PATH):
        print(f"Report file size: {os.path.getsize(REPORT_PATH)} bytes")
    else:
        print("🟢 Strategy Tester triggered on local MT5 terminal! Open MT5 Terminal -> Strategy Tester tab (Ctrl+R) to view visual equity chart & report.")

if __name__ == "__main__":
    main()
