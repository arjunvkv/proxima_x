#!/usr/bin/env python3
"""Run Local MT5 Strategy Tester for Ultra Monster across all 9 Pairs in the Universe."""

import os, subprocess, time, re
import pandas as pd

TERMINAL = r"C:\Program Files\FundedNext MT5 Terminal\terminal64.exe"
APPDATA = r"C:\Users\Arjun Sasi\AppData\Roaming\MetaQuotes\Terminal\89FE26BBBAB28C077BBF5FA8C1B4DF1C"
LOCAL_DIR = r"C:\Trading\Agentic_Trading\proxima_x\paper_trade\mt5_backtest"

PAIRS_ALL = ["EURUSD", "GBPUSD", "USDJPY", "EURAUD", "GBPAUD", "EURJPY", "GBPJPY", "EURNZD", "GBPNZD"]

def run_single_pair_test(symbol):
    ini_path = os.path.join(LOCAL_DIR, f"test_ultra_{symbol}.ini")
    report_name = f"ultra_monster_{symbol}_report.htm"
    report_path = os.path.join(APPDATA, report_name)

    if os.path.exists(report_path):
        os.remove(report_path)

    ini_content = f"""[Tester]
Expert=Ultra_Monster_MT5.ex5
Symbol={symbol}
Period=M5
Deposit=10000
Currency=USD
Leverage=1:100
Model=1
ExecutionMode=0
FromDate=2026.01.01
ToDate=2026.08.01
Report={report_name}
ReplaceReport=1
ShutdownTerminal=1
"""
    with open(ini_path, "w", encoding="utf-8") as f:
        f.write(ini_content)

    print(f"Testing {symbol:<8} on local MT5 Strategy Tester...")
    cmd = [TERMINAL, f"/config:{ini_path}"]
    proc = subprocess.Popen(cmd)

    for _ in range(60):
        if os.path.exists(report_path) and os.path.getsize(report_path) > 2000:
            break
        time.sleep(1.0)

    if not os.path.exists(report_path):
        return None

    try:
        with open(report_path, "r", encoding="utf-16le", errors="ignore") as f:
            html = f.read()
    except Exception:
        with open(report_path, "r", encoding="utf-8", errors="ignore") as f:
            html = f.read()

    text = re.sub(r'<[^>]+>', ' ', html)
    text = re.sub(r'\s+', ' ', text)

    def get_val(kw):
        m = re.search(f"{kw}[^0-9\-]*([\-0-9\.\%\$\s]+)", text, re.IGNORECASE)
        return m.group(1).strip() if m else "0"

    net_pnl = get_val("Total Net Profit")
    trades = get_val("Total Trades")
    pf = get_val("Profit Factor")
    wins_match = re.search(r"Short Positions \(win \%\)[^0-9]*([0-9]+)\s*\(([0-9\.\%]+)\)", text)
    
    return {
        "Symbol": symbol,
        "Total Trades": trades,
        "Net Profit": net_pnl,
        "Profit Factor": pf
    }

def main():
    print("="*115)
    print("RUNNING MULTI-PAIR MT5 STRATEGY TESTER FOR ULTRA MONSTER (ALL 9 PAIRS)...")
    print("="*115)

    results = []
    for pair in PAIRS_ALL:
        res = run_single_pair_test(pair)
        if res:
            results.append(res)
            print(f"  🟢 {pair:<8} | Trades: {res['Total Trades']:<5} | Net Profit: {res['Net Profit']:<10} | PF: {res['Profit Factor']}")
        else:
            print(f"  ⚠️ {pair:<8} | MT5 Tester timed out or failed to output report.")

    print("\n" + "="*85)
    print("MQL5 STRATEGY TESTER MULTI-PAIR RESULTS FOR ULTRA MONSTER:")
    print("="*85)
    if results:
        df = pd.DataFrame(results)
        print(df.to_string(index=False))
    print("="*85)

if __name__ == "__main__":
    main()
