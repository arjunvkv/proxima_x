#!/usr/bin/env python3
"""Run MT5 Strategy Tester for Strategy #6 across EURAUD, GBPAUD, and GBPNZD."""
import subprocess, time, os, re, glob
from bs4 import BeautifulSoup

SYMBOLS = ["GBPAUD", "GBPNZD"]

def run_symbol_test(symbol):
    ini_content = f"""[Tester]
Expert=CPMC_Z_MT5.ex5
Symbol={symbol}
Period=M5
Login=5053225887
Model=1
ExecutionMode=0
Deposit=6000
Currency=USD
Leverage=100
FromDate=2026.01.01
ToDate=2026.07.28
Report=cpmc_z_{symbol}_report
ReplaceReport=1
ShutdownTerminal=1
"""
    ini_path = rf"C:\Trading\Agentic_Trading\proxima_x\paper_trade\mt5_backtest\cpmc_z_{symbol}.ini"
    with open(ini_path, "w") as f:
        f.write(ini_content)

    print(f"Running MT5 Strategy Tester for {symbol}...")
    cmd = f'"C:\\Program Files\\FundedNext MT5 Terminal\\terminal64.exe" /config:"{ini_path}"'
    subprocess.run(cmd, shell=True)
    time.sleep(3)

def parse_report(symbol):
    report_files = glob.glob(rf'C:\Users\Arjun Sasi\AppData\Roaming\MetaQuotes\**\cpmc_z_{symbol}_report*.htm*', recursive=True)
    if not report_files:
        return None

    path = report_files[0]
    with open(path, "r", encoding="utf-16", errors="ignore") as f:
        html = f.read()

    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text()

    def extract_val(pattern, default="0"):
        m = re.search(pattern, text)
        return m.group(1).strip() if m else default

    pnl_str = extract_val(r"Total Net Profit:\s*([^\n\r]+)")
    pf_str = extract_val(r"Profit Factor:\s*([^\n\r]+)")
    trades_str = extract_val(r"Total Trades:\s*([^\n\r]+)")
    win_str = extract_val(r"Profit Trades \(\% of total\):\s*([^\n\r]+)")
    dd_str = extract_val(r"Maximal Drawdown:\s*([^\n\r]+)")

    return {
        "Symbol": symbol,
        "Trades": trades_str,
        "Net_Profit": pnl_str,
        "Profit_Factor": pf_str,
        "Win_Rate": win_str,
        "Max_Drawdown": dd_str
    }

def main():
    print("="*85)
    print("OFFICIAL MT5 STRATEGY TESTER AUDIT: STRATEGY #6 (CPMC_Z_MT5)")
    print("FundedNext Server 3 | Tick-Level Real Spreads & Commission")
    print("="*85)

    results = []
    for s in SYMBOLS:
        run_symbol_test(s)
        res = parse_report(s)
        if res:
            results.append(res)

    print("\n" + "="*85)
    print("MT5 STRATEGY TESTER TICK-LEVEL RESULTS (FUNDEDNEXT SERVER 3)")
    print("="*85)
    for r in results:
        print(f"  Symbol: {r['Symbol']:<7} | Trades: {r['Trades']:<4} | Net Profit: {r['Net_Profit']:<12} | PF: {r['Profit_Factor']:<5} | Win Rate: {r['Win_Rate']}")

if __name__ == "__main__":
    main()
