#!/usr/bin/env python3
"""Audit Past 4 Days (July 28 - July 31, 2026) MT5 Portfolio Performance."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd
import numpy as np

from proxima_honest_backtest.strategies.sunday_h22.sweep import load_and_align
from ultra_buff_rolling_orb import run_ultra_buffed_orb

PAIRS_ALL = ["EURUSD", "GBPUSD", "USDJPY", "EURAUD", "GBPAUD", "EURJPY", "GBPJPY", "EURNZD", "GBPNZD"]

def main():
    print("Loading M5 dataset for Past 4 Days (July 28 - July 31, 2026) Audit...")
    raw, pre_align = load_and_align()
    pieces = [df.set_index("time")[["close","open","high","low"]] for df in raw.values()]
    for i, p in enumerate(raw.keys()):
        pieces[i].columns = [p, f"{p}_open", f"{p}_high", f"{p}_low"]
    df_all = pd.concat(pieces, axis=1, sort=True).ffill().bfill()
    times = pd.to_datetime(df_all.index)

    close_mat = df_all[[p for p in PAIRS_ALL]].values
    open_mat = df_all[[f"{p}_open" for p in PAIRS_ALL]].values
    high_mat = df_all[[f"{p}_high" for p in PAIRS_ALL]].values
    low_mat = df_all[[f"{p}_low" for p in PAIRS_ALL]].values

    hours = times.hour.values
    minutes = times.minute.values

    # Run Ultra Monster Engine
    df_u = run_ultra_buffed_orb(df_all, close_mat, open_mat, high_mat, low_mat, hours, minutes, PAIRS_ALL, 6.0, range(0, 24), [0, 30], 3)
    df_u["date"] = pd.to_datetime(df_u["time"]).dt.date

    target_dates = [
        pd.to_datetime("2026-07-28").date(),
        pd.to_datetime("2026-07-29").date(),
        pd.to_datetime("2026-07-30").date(),
        pd.to_datetime("2026-07-31").date()
    ]

    four_day_results = []
    cum_squeeze = 0.0

    for d in target_dates:
        sub = df_u[df_u["date"] == d]
        pnls_015 = sub["net_pnl"].values
        n_trades = len(pnls_015)
        wins = sum(1 for p in pnls_015 if p > 0)
        losses = sum(1 for p in pnls_015 if p <= 0)
        wr = (wins / n_trades * 100.0) if n_trades > 0 else 0.0

        pnl_015 = sum(pnls_015)
        pnl_squeeze = pnl_015 / 0.15 * 1.00
        cum_squeeze += pnl_squeeze

        four_day_results.append({
            "Date": d.strftime("%Y-%m-%d (%a)"),
            "Total Trades": n_trades,
            "Wins": wins,
            "Losses": losses,
            "Win Rate": f"{wr:.1f}%" if n_trades > 0 else "N/A (Skipped)",
            "Net PnL (0.15 Lot Base)": f"+${pnl_015:,.2f}" if pnl_015 >= 0 else f"-${abs(pnl_015):,.2f}",
            "Net PnL (1.00 Lot Squeeze)": f"+${pnl_squeeze:,.2f}" if pnl_squeeze >= 0 else f"-${abs(pnl_squeeze):,.2f}",
            "Cumulative Squeeze PnL": f"+${cum_squeeze:,.2f}"
        })

    df_4d = pd.DataFrame(four_day_results)
    print("="*110)
    print("PAST 4 DAYS MT5 PORTFOLIO PERFORMANCE AUDIT (JULY 28 - JULY 31, 2026)")
    print("="*110)
    print(df_4d.to_string(index=False))
    print("="*110)
    print(f"CUMULATIVE 4-DAY NET PROFIT (MAX SQUEEZE): +${cum_squeeze:,.2f} NET PROFIT")
    print("="*110)

if __name__ == "__main__":
    main()
