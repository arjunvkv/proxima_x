#!/usr/bin/env python3
"""Parse MT5 Strategy Tester Report for Strategy #6 (CPMC_Z_MT5)."""
import re, glob
from bs4 import BeautifulSoup

def main():
    report_files = glob.glob(r'C:\Users\Arjun Sasi\AppData\Roaming\MetaQuotes\**\cpmc_z_report*.htm*', recursive=True)
    if not report_files:
        print("No report file found")
        return

    path = report_files[0]
    with open(path, "r", encoding="utf-16", errors="ignore") as f:
        html = f.read()

    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text()

    print("="*85)
    print("OFFICIAL METATRADER 5 STRATEGY TESTER REPORT: STRATEGY #6 (CPMC_Z_MT5)")
    print("Server: FundedNext Server 3 | Model: Every Tick Based on Real Ticks")
    print("="*85)

    # Extract key metrics
    def extract_val(pattern, text, default="N/A"):
        m = re.search(pattern, text)
        return m.group(1).strip() if m else default

    pnl = extract_val(r"Total Net Profit:\s*([^\n\r]+)", text)
    pf = extract_val(r"Profit Factor:\s*([^\n\r]+)", text)
    trades = extract_val(r"Total Trades:\s*([^\n\r]+)", text)
    win_pct = extract_val(r"Profit Trades \(\% of total\):\s*([^\n\r]+)", text)
    dd_pct = extract_val(r"Maximal Drawdown:\s*([^\n\r]+)", text)
    sharpe = extract_val(r"Sharpe Ratio:\s*([^\n\r]+)", text)

    print(f"  Total Trades        : {trades}")
    print(f"  Total Net Profit    : {pnl}")
    print(f"  Profit Factor       : {pf}")
    print(f"  Win Rate            : {win_pct}")
    print(f"  Maximal Drawdown    : {dd_pct}")
    print(f"  Sharpe Ratio        : {sharpe}")
    print("="*85)

if __name__ == "__main__":
    main()
