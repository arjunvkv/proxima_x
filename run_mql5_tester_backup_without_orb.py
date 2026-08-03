#!/usr/bin/env python3
"""Run Local MT5 Strategy Tester for Ultra_Monster_MT5 WITHOUT rolling ORB (15m test script)."""

import os, subprocess, time, re

TERMINAL = r"C:\Program Files\FundedNext MT5 Terminal\terminal64.exe"
METAEDITOR = r"C:\Program Files\FundedNext MT5 Terminal\MetaEditor64.exe"
APPDATA = r"C:\Users\Arjun Sasi\AppData\Roaming\MetaQuotes\Terminal\89FE26BBBAB28C077BBF5FA8C1B4DF1C"
LOCAL_DIR = r"C:\Trading\Agentic_Trading\proxima_x\paper_trade\mt5_backtest"

INI_PATH = os.path.join(LOCAL_DIR, "tester_ultra_monster_without_orb.ini")
REPORT_PATH = os.path.join(APPDATA, "ultra_monster_without_orb_report.htm")

backup_mq5 = os.path.join(LOCAL_DIR, "updated_version_backup", "Ultra_Monster_MT5.mq5")
appdata_mq5 = os.path.join(APPDATA, "MQL5", "Experts", "Ultra_Monster_NoORB.mq5")
appdata_ex5 = os.path.join(APPDATA, "MQL5", "Experts", "Ultra_Monster_NoORB.ex5")

def main():
    print("="*115)
    print("RUNNING LOCAL MQL5 STRATEGY TESTER FOR ULTRA MONSTER WITHOUT ROLLING ORB...")
    print("="*115)

    with open(backup_mq5, "r", encoding="utf-8") as f:
        code = f.read()
    
    # Fix volume normalization to double
    code = code.replace("(float)NormalizeVolume(", "NormalizeVolume(")

    with open(appdata_mq5, "w", encoding="utf-8") as f:
        f.write(code)

    cmd_comp = [METAEDITOR, f"/compile:{appdata_mq5}"]
    subprocess.run(cmd_comp, check=False)
    time.sleep(3.5)

    if not os.path.exists(appdata_ex5):
        print("❌ MetaEditor compilation failed!")
        return

    print(f"🟢 Compiled Ultra_Monster_NoORB.mq5 successfully ({os.path.getsize(appdata_ex5)} bytes)")

    if os.path.exists(REPORT_PATH):
        os.remove(REPORT_PATH)

    ini_content = f"""[Tester]
Expert=Ultra_Monster_NoORB.ex5
Symbol=GBPUSD
Period=M5
Deposit=10000
Currency=USD
Leverage=1:100
Model=1
ExecutionMode=0
FromDate=2026.01.01
ToDate=2026.08.01
Report=ultra_monster_without_orb_report.htm
ReplaceReport=1
ShutdownTerminal=1
"""

    with open(INI_PATH, "w", encoding="utf-8") as f:
        f.write(ini_content)

    cmd = [TERMINAL, f"/config:{INI_PATH}"]
    proc = subprocess.Popen(cmd)

    print("Waiting for MQL5 Strategy Tester to complete and save report...")
    for i in range(120):
        if os.path.exists(REPORT_PATH) and os.path.getsize(REPORT_PATH) > 2000:
            print(f"🟢 MQL5 Strategy Tester finished in {i+1} seconds!")
            break
        time.sleep(1.0)

    if os.path.exists(REPORT_PATH):
        try:
            with open(REPORT_PATH, "r", encoding="utf-16le", errors="ignore") as f:
                html = f.read()
        except Exception:
            with open(REPORT_PATH, "r", encoding="utf-8", errors="ignore") as f:
                html = f.read()

        text = re.sub(r'<[^>]+>', ' ', html)
        text = re.sub(r'\s+', ' ', text)

        print("\n" + "="*85)
        print("PARSED MQL5 STRATEGY TESTER METRICS (WITHOUT ROLLING ORB):")
        print("="*85)
        
        keywords = ["Total Net Profit", "Gross Profit", "Gross Loss", "Profit Factor", "Expected Payoff", "Total Trades", "Short Positions", "Long Positions", "Maximal drawdown", "Balance Drawdown"]
        for kw in keywords:
            m = re.search(f"{kw}[^0-9\-]*([\-0-9\.\%\$\s]+)", text, re.IGNORECASE)
            if m:
                val = m.group(1).strip()
                print(f"  • {kw:<24}: {val}")
        print("="*85)

if __name__ == "__main__":
    main()
