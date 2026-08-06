#!/usr/bin/env python3
"""Parse ALL actual UTF-16 MT5 HTML report files from disk for exact MT5 terminal generated numbers."""

import os, glob
from bs4 import BeautifulSoup
import MetaTrader5 as mt5
from datetime import datetime, timedelta

reports_dir = r"C:\Trading\Agentic_Trading\proxima_x\paper_trade\mt5_backtest\bt_reports"

print("=" * 125)
print("RUN 1: ACTUAL 7-MONTH MQL5 STRATEGY TESTER TERMINAL RUNS (PARSED DIRECTLY FROM MT5 .HTM REPORT FILES)")
print("=" * 125)

report_files = [
    ("6p_EURAUD.htm", "CPPF Z EURAUD MT5 Terminal Run"),
    ("6p_GBPAUD.htm", "CPPF Z GBPAUD MT5 Terminal Run"),
    ("6p_AUDNZD.htm", "CPPF Z AUDNZD MT5 Terminal Run"),
    ("6p_EURNZD.htm", "CPPF Z EURNZD MT5 Terminal Run"),
    ("6p_GBPCAD.htm", "CPPF Z GBPCAD MT5 Terminal Run"),
    ("6p_GBPNZD.htm", "CPPF Z GBPNZD MT5 Terminal Run")
]

print(f"{'Strategy / Pair MT5 Report':36} {'Trades':8} {'Win Rate (%)':14} {'Net Profit ($)':16} {'Profit Factor':14} {'Max DD (%)'}")
print("-" * 125)

for filename, title in report_files:
    filepath = os.path.join(reports_dir, filename)
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-16") as f:
                text = f.read()
            soup = BeautifulSoup(text, 'html.parser')
            
            pnl_val = "N/A"
            pf_val = "N/A"
            tr_val = "N/A"
            wr_val = "N/A"
            dd_val = "N/A"
            
            for tr in soup.find_all('tr'):
                row = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
                row_str = ' '.join(row)
                if 'Total Net Profit:' in row_str:
                    pnl_val = row[1] if len(row) > 1 else "N/A"
                if 'Profit Factor:' in row_str:
                    pf_val = row[1] if len(row) > 1 else "N/A"
                if 'Total Trades:' in row_str:
                    tr_val = row[1] if len(row) > 1 else "N/A"
                if 'Profit Trades (% of total):' in row_str:
                    wr_val = row[1] if len(row) > 1 else "N/A"
                if 'Balance Drawdown Maximal:' in row_str:
                    dd_val = row[1] if len(row) > 1 else "N/A"
            
            print(f"{title:36} {tr_val:<8} {wr_val:14} {pnl_val:16} {pf_val:14} {dd_val}")
        except Exception as e:
            print(f"Error parsing {filename}: {e}")

print("=" * 125)

# --- RUN 2: ACTUAL TODAY MT5 LIVE TERMINAL ACCOUNT HISTORY (AUG 3, 2026) ---
print("\n" + "=" * 125)
print("RUN 2: ACTUAL TODAY LIVE MT5 ACCOUNT DEALS HISTORY (DIRECT FROM FTMO DEMO TERMINAL LOGIN #1514168544)")
print("=" * 125)

if mt5.initialize():
    if mt5.login(1514168544, "$!4fwBIc", "FTMO-Demo"):
        acc = mt5.account_info()
        deals = mt5.history_deals_get(datetime(2026, 8, 3, 0, 0, 0), datetime(2026, 8, 4, 23, 59, 59))
        
        positions = {}
        if deals:
            for d in deals:
                if not d.symbol or d.entry not in [0, 1]:
                    continue
                pid = d.position_id
                if pid not in positions:
                    positions[pid] = {"in": None, "out": None}
                if d.entry == 0:
                    positions[pid]["in"] = d
                elif d.entry == 1:
                    positions[pid]["out"] = d

        actual_mt5_trades_today = []
        for pid, p in positions.items():
            d_in, d_out = p["in"], p["out"]
            if not d_in or not d_out:
                continue
            net = d_out.profit + d_out.swap + d_out.commission
            actual_mt5_trades_today.append({
                "pid": pid,
                "symbol": d_in.symbol,
                "lot": d_in.volume,
                "pnl": net,
                "win": net > 0,
                "comment": d_in.comment or ""
            })
        
        total_deals = len(actual_mt5_trades_today)
        wins = sum(1 for t in actual_mt5_trades_today if t["win"])
        losses = total_deals - wins
        wr = (wins / total_deals * 100.0) if total_deals > 0 else 0.0
        total_pnl = sum(t["pnl"] for t in actual_mt5_trades_today)
        gross_w = sum(t["pnl"] for t in actual_mt5_trades_today if t["win"])
        gross_l = abs(sum(t["pnl"] for t in actual_mt5_trades_today if not t["win"]))
        pf = (gross_w / gross_l) if gross_l > 0 else 0.0
        
        print(f"  • MT5 Account Balance : ${acc.balance:.2f} USD")
        print(f"  • MT5 Total Executed  : {total_deals} closed positions")
        print(f"  • MT5 Winning Trades  : {wins}")
        print(f"  • MT5 Losing Trades   : {losses}")
        print(f"  • MT5 Win Rate        : {wr:.2f}%")
        print(f"  • MT5 Gross Profit    : +${gross_w:.2f}")
        print(f"  • MT5 Gross Loss      : -${gross_l:.2f}")
        print(f"  • MT5 Net Realized PnL: ${total_pnl:.2f}")
        print(f"  • MT5 Profit Factor   : {pf:.2f}")
        mt5.shutdown()

print("=" * 125)
