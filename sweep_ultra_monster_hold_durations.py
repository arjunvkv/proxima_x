#!/usr/bin/env python3
"""Sweep Hold Durations for Ultra_Monster_MT5 across 7-Month Dataset."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd
import numpy as np

from proxima_honest_backtest.strategies.sunday_h22.sweep import load_and_align
from ultra_buff_rolling_orb import run_ultra_buffed_orb

PAIRS_ALL = ["EURUSD", "GBPUSD", "USDJPY", "EURAUD", "GBPAUD", "EURJPY", "GBPJPY", "EURNZD", "GBPNZD"]

def main():
    print("Sweeping All Hold Durations for Ultra_Monster_MT5 across 7-Month Dataset...")
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

    holds_to_test = [3, 6, 9, 12, 18, 24]  # 15m, 30m, 45m, 60m, 90m, 120m
    hold_results = []

    for hb in holds_to_test:
        df_u = run_ultra_buffed_orb(df_all, close_mat, open_mat, high_mat, low_mat, hours, minutes, PAIRS_ALL, 12.0, range(0, 24), [0, 30], hb)
        df_u["pnl_1lot"] = df_u["net_pnl"] / 0.15 * 1.00

        pnls = df_u["pnl_1lot"].values
        n_t = len(pnls)
        wins = sum(1 for p in pnls if p > 0)
        losses = sum(1 for p in pnls if p <= 0)
        wr = wins / n_t * 100.0 if n_t > 0 else 0
        tot_pnl = sum(pnls)
        gross_w = sum(p for p in pnls if p > 0)
        gross_l = abs(sum(p for p in pnls if p <= 0))
        pf = gross_w / max(1, gross_l)
        avg_w = np.mean([p for p in pnls if p > 0]) if wins > 0 else 0
        avg_l = np.mean([p for p in pnls if p <= 0]) if losses > 0 else 0

        hold_results.append({
            "Hold Bars": hb,
            "Hold Duration": f"{hb * 5} Mins",
            "Total Trades": n_t,
            "Net Win Rate (%)": f"{wr:.1f}%",
            "Profit Factor": f"{pf:.2f}",
            "Average Win ($)": f"+${avg_w:.2f}",
            "Average Loss ($)": f"-${abs(avg_l):.2f}",
            "Cumulative Net Profit": f"+${tot_pnl:,.2f}"
        })

    df_res = pd.DataFrame(hold_results)
    print("="*115)
    print("ULTRA_MONSTER HOLD DURATION OPTIMIZATION MATRIX (7-MONTH AUDITED DATASET)")
    print("="*115)
    print(df_res.to_string(index=False))
    print("="*115)

if __name__ == "__main__":
    main()
