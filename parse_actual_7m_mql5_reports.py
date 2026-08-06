#!/usr/bin/env python3
"""Parse MQL5 strategy tester HTML reports for exact 7-month MT5 terminal numbers."""

import os, re, glob
from bs4 import BeautifulSoup

reports_dir = r"C:\Trading\Agentic_Trading\proxima_x\paper_trade\mt5_backtest\bt_reports"

print("=" * 115)
print("ACTUAL 7-MONTH MQL5 STRATEGY TESTER TERMINAL RUNS (PARSED FROM MT5 REPORT FILES)")
print("=" * 115)

report_files = [
    ("6p_EURAUD.htm", "CPPF Z / V2+z 6-Pair MT5 Run"),
    ("6p_GBPAUD.htm", "CPPF Z GBPAUD MT5 Run"),
    ("oos_EURAUD.htm", "EURAUD OOS MT5 Terminal Run"),
    ("oos_GBPAUD.htm", "GBPAUD OOS MT5 Terminal Run")
]

for filename, title in report_files:
    filepath = os.path.join(reports_dir, filename)
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            html = f.read()
            
        # Parse basic metrics using regex from MT5 report tables
        profit_match = re.search(r"Total Net Profit</td>\s*<td[^>]*><b>([^<]+)</b>", html, re.IGNORECASE)
        pf_match     = re.search(r"Profit Factor</td>\s*<td[^>]*><b>([^<]+)</b>", html, re.IGNORECASE)
        trades_match = re.search(r"Total Trades</td>\s*<td[^>]*><b>([^<]+)</b>", html, re.IGNORECASE)
        win_match    = re.search(r"Short Trades \(win %\)[^<]*</td>\s*<td[^>]*><b>([^<]+)</b>", html, re.IGNORECASE)

        pnl_val = profit_match.group(1) if profit_match else "N/A"
        pf_val  = pf_match.group(1) if pf_match else "N/A"
        tr_val  = trades_match.group(1) if trades_match else "N/A"
        win_val = win_match.group(1) if win_match else "N/A"

        print(f"  • {title:<32} ({filename})")
        print(f"    - Total Trades: {tr_val}")
        print(f"    - Net Profit  : {pnl_val}")
        print(f"    - Profit Factor: {pf_val}")
        print(f"    - Short Win % : {win_val}\n")

print("=" * 115)
