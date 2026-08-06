#!/usr/bin/env python3
"""Parse actual MT5 terminal executed history deals and actual MT5 tester report files."""

import MetaTrader5 as mt5
import os, glob
from datetime import datetime, timedelta

print("=" * 115)
print("ACTUAL MT5 TERMINAL EXECUTION RUNS (DIRECT MT5 ENGINE READ)")
print("=" * 115)

# --- RUN 1: TODAY's ACTUAL MT5 TERMINAL ACCOUNT HISTORY (AUG 3, 2026) ---
if mt5.initialize():
    if mt5.login(1514168544, "$!4fwBIc", "FTMO-Demo"):
        acc = mt5.account_info()
        print(f"MT5 DEMO TERMINAL LOGIN: {acc.login} | SERVER: {acc.server}")
        
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
        
        print("\n--- ACTUAL MT5 ACCOUNT DEALS HISTORY TODAY (AUG 3, 2026) ---")
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

# --- RUN 2: ACTUAL 7-MONTH MT5 TESTER REPORTS ON DISK ---
print("\n--- ACTUAL 7-MONTH MT5 TESTER HTML REPORT FILES ON DISK ---")
reports_dir = r"C:\Trading\Agentic_Trading\proxima_x\paper_trade\mt5_backtest\bt_reports"

if os.path.exists(reports_dir):
    report_files = glob.glob(os.path.join(reports_dir, "*.htm"))
    print(f"Total MT5 .htm reports found in bt_reports/: {len(report_files)}")
    for rf in report_files[:10]:
        sz = os.path.getsize(rf)
        dt = datetime.fromtimestamp(os.path.getmtime(rf)).strftime("%Y-%m-%d %H:%M")
        print(f"  • {os.path.basename(rf):<35} | {sz:>8} bytes | Modified: {dt}")

print("=" * 115)
