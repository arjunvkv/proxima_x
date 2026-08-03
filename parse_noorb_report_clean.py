#!/usr/bin/env python3
"""Parse noorb_report.htm metrics cleanly."""

import os, re

path = r"C:\Users\Arjun Sasi\AppData\Roaming\MetaQuotes\Terminal\89FE26BBBAB28C077BBF5FA8C1B4DF1C\noorb_report.htm"

def main():
    if not os.path.exists(path):
        print("No report file found.")
        return

    with open(path, "r", encoding="utf-16le", errors="ignore") as f:
        html = f.read()

    text = re.sub(r'<[^>]+>', ' ', html)
    text = re.sub(r'\s+', ' ', text)

    print("="*85)
    print("MQL5 STRATEGY TESTER REPORT METRICS (ULTRA MONSTER PREVIOUS TESTED BACKUP):")
    print("="*85)

    keywords = [
        "Total Net Profit",
        "Gross Profit",
        "Gross Loss",
        "Profit Factor",
        "Expected Payoff",
        "Total Trades",
        "Short Positions",
        "Long Positions",
        "Maximal drawdown",
        "Balance Drawdown"
    ]

    for kw in keywords:
        m = re.search(f"{kw}[^0-9\-]*([\-0-9\.\%\$\s]+)", text, re.IGNORECASE)
        if m:
            val = m.group(1).strip()
            print(f"  • {kw:<24}: {val}")

    print("="*85)

if __name__ == "__main__":
    main()
