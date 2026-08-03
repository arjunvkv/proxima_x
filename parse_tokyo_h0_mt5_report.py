#!/usr/bin/env python3
"""Parse MT5 Strategy Tester Report for TokyoH0_MT5."""
import re, glob
from bs4 import BeautifulSoup

def main():
    report_files = glob.glob(r'C:\Users\Arjun Sasi\AppData\Roaming\MetaQuotes\**\tokyo_h0_report*.htm*', recursive=True)
    if not report_files:
        print("No report file found")
        return

    path = report_files[0]
    with open(path, "r", encoding="utf-16", errors="ignore") as f:
        html = f.read()

    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text()

    print("="*85)
    print("OFFICIAL METATRADER 5 STRATEGY TESTER REPORT: TOKYOH0_MT5 (v2.00)")
    print("Server: FundedNext Server 3 | Model: Every Tick Based on Real Ticks")
    print("Period: Jan 1, 2026 – Jul 28, 2026 (7 Full Months / 30 Weeks)")
    print("="*85)

    def extract_val(pattern, default="N/A"):
        m = re.search(pattern, text)
        return m.group(1).strip() if m else default

    pnl = extract_val(r"Total Net Profit:\s*([^\n\r]+)")
    pf = extract_val(r"Profit Factor:\s*([^\n\r]+)")
    trades = extract_val(r"Total Trades:\s*([^\n\r]+)")
    win_pct = extract_val(r"Profit Trades \(\% of total\):\s*([^\n\r]+)")
    dd_pct = extract_val(r"Maximal Drawdown:\s*([^\n\r]+)")

    print(f"  Total MT5 Trades    : {trades}")
    print(f"  Total Net Profit    : {pnl}")
    print(f"  Profit Factor       : {pf}")
    print(f"  Net Win Rate        : {win_pct}")
    print(f"  Maximal Drawdown    : {dd_pct}")
    print("="*85)

if __name__ == "__main__":
    main()
