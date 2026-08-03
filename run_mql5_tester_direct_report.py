#!/usr/bin/env python3
"""Run MT5 Strategy Tester for Ultra_Monster_MT5.mq5 and parse exact MQL5 results."""

import os, subprocess, time, re

TERMINAL = r"C:\Program Files\FundedNext MT5 Terminal\terminal64.exe"
APPDATA = r"C:\Users\Arjun Sasi\AppData\Roaming\MetaQuotes\Terminal\89FE26BBBAB28C077BBF5FA8C1B4DF1C"
LOCAL_DIR = r"C:\Trading\Agentic_Trading\proxima_x\paper_trade\mt5_backtest"

INI_PATH = os.path.join(LOCAL_DIR, "ultra_monster_test.ini")
REPORT_PATH = os.path.join(APPDATA, "ultra_monster_mql5_report.htm")

def main():
    print("="*115)
    print("RUNNING DIRECT MQL5 STRATEGY TESTER FOR ULTRA MONSTER...")
    print("="*115)

    if os.path.exists(REPORT_PATH):
        os.remove(REPORT_PATH)

    ini_content = f"""[Tester]
Expert=Ultra_Monster_MT5.ex5
Symbol=GBPUSD
Period=M5
Deposit=10000
Currency=USD
Leverage=1:100
Model=1
ExecutionMode=0
FromDate=2026.01.01
ToDate=2026.08.01
Report=ultra_monster_mql5_report.htm
ReplaceReport=1
ShutdownTerminal=1
"""

    with open(INI_PATH, "w", encoding="utf-8") as f:
        f.write(ini_content)

    cmd = [TERMINAL, f"/config:{INI_PATH}"]
    print(f"Executing MT5 Terminal: {cmd}")
    proc = subprocess.Popen(cmd)

    print("Waiting for MQL5 Strategy Tester to complete and save report...")
    for i in range(120):
        if os.path.exists(REPORT_PATH) and os.path.getsize(REPORT_PATH) > 2000:
            print(f"🟢 MQL5 Strategy Tester finished in {i+1} seconds!")
            break
        time.sleep(1.0)

    if os.path.exists(REPORT_PATH):
        print(f"Report path: {REPORT_PATH} ({os.path.getsize(REPORT_PATH)} bytes)")
        try:
            with open(REPORT_PATH, "r", encoding="utf-16le", errors="ignore") as f:
                html = f.read()
        except Exception:
            with open(REPORT_PATH, "r", encoding="utf-8", errors="ignore") as f:
                html = f.read()

        # Clean HTML tags to read raw report text
        text = re.sub(r'<[^>]+>', ' ', html)
        text = re.sub(r'\s+', ' ', text)

        print("\n" + "="*85)
        print("PARSED MQL5 STRATEGY TESTER METRICS FOR ULTRA MONSTER:")
        print("="*85)
        
        keywords = ["Total Net Profit", "Gross Profit", "Gross Loss", "Profit Factor", "Expected Payoff", "Total Trades", "Short Positions", "Long Positions", "Maximal drawdown", "Balance Drawdown"]
        for kw in keywords:
            m = re.search(f"{kw}[^0-9\-]*([\-0-9\.\%\$\s]+)", text, re.IGNORECASE)
            if m:
                val = m.group(1).strip()
                print(f"  • {kw:<24}: {val}")
        print("="*85)

    else:
        print("⚠️ Report file not yet saved by MT5 terminal process. The Strategy Tester is running in MT5 GUI.")

if __name__ == "__main__":
    main()
