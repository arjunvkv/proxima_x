#!/usr/bin/env python3
"""Optimize NYH21_MT5 to Increase Win Rate AND Event Frequency."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd
import numpy as np

from proxima_honest_backtest.strategies.sunday_h22.sweep import load_and_align

PAIRS_JPY = ["EURJPY", "GBPJPY", "USDJPY", "EURAUD", "GBPAUD", "EURUSD", "GBPUSD"]

def run_nyh_sim(df_all, close_mat, hours, minutes, pairs, target_hours=[20, 21], min_decline_pips=2.0, hold_bars=12):
    trades = []
    
    for t_idx in range(12, len(df_all) - hold_bars):
        if hours[t_idx] in target_hours and minutes[t_idx] == 0:
            for p_idx, pair in enumerate(pairs):
                c_now = close_mat[t_idx, p_idx]
                c_lookback = close_mat[t_idx-12, p_idx]
                c_exit = close_mat[t_idx+12, p_idx]

                pip_mult = 100.0 if "JPY" in pair else 10000.0
                ret_60m = (c_now - c_lookback) * pip_mult

                # Fade decline
                if ret_60m <= -min_decline_pips:
                    c_entry = close_mat[t_idx+1, p_idx]
                    pnl_pip = (c_exit - c_entry) * pip_mult
                    pnl_usd = pnl_pip * 10.0 * 1.00
                    trades.append(pnl_usd)

    return trades

def main():
    print("Optimizing NYH21_MT5 Win Rate AND Event Frequency across 7-Month Dataset...")
    raw, pre_align = load_and_align()
    pieces = [df.set_index("time")[["close","open","high","low"]] for df in raw.values()]
    for i, p in enumerate(raw.keys()):
        pieces[i].columns = [p, f"{p}_open", f"{p}_high", f"{p}_low"]
    df_all = pd.concat(pieces, axis=1, sort=True).ffill().bfill()
    times = pd.to_datetime(df_all.index)

    close_mat = df_all[[p for p in PAIRS_JPY]].values
    hours = times.hour.values
    minutes = times.minute.values

    tot_days = 148

    configs = [
        {"name": "v1.00 Baseline (2 pairs, 21:00 UTC, h=60m)", "pairs": ["EURJPY", "GBPJPY"], "hours": [21], "min_p": 2.0, "hold": 12},
        {"name": "Option A: Add USDJPY (3 JPY pairs, 21:00 UTC)", "pairs": ["EURJPY", "GBPJPY", "USDJPY"], "hours": [21], "min_p": 2.0, "hold": 12},
        {"name": "Option B: Expand Hours (3 JPY pairs, 20:00 & 21:00 UTC)", "pairs": ["EURJPY", "GBPJPY", "USDJPY"], "hours": [20, 21], "min_p": 2.0, "hold": 12},
        {"name": "Option C: 90m Hold Optimization (3 JPY pairs, 20-21 UTC, h=90m)", "pairs": ["EURJPY", "GBPJPY", "USDJPY"], "hours": [20, 21], "min_p": 2.0, "hold": 18},
        {"name": "Option D: 5-Pair JPY Cross Expansion (5 pairs, 20-21 UTC, h=90m)", "pairs": ["EURJPY", "GBPJPY", "USDJPY", "EURAUD", "GBPAUD"], "hours": [20, 21], "min_p": 3.0, "hold": 18},
    ]

    results = []

    for cfg in configs:
        pnls = run_nyh_sim(df_all, close_mat, hours, minutes, cfg["pairs"], cfg["hours"], cfg["min_p"], cfg["hold"])
        n_t = len(pnls)
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        wr = len(wins) / n_t * 100.0 if n_t > 0 else 0
        tot_pnl = sum(pnls)
        pf = sum(wins) / max(1, abs(sum(losses)))
        avg_w = np.mean(wins) if wins else 0
        t_per_day = n_t / tot_days

        results.append({
            "Configuration": cfg["name"],
            "Total Trades": n_t,
            "Trades / Day": f"{t_per_day:.2f} / day",
            "Wins 🟢": len(wins),
            "Losses 🔴": len(losses),
            "Net Win Rate (%)": f"{wr:.1f}%",
            "Profit Factor": f"{pf:.2f}",
            "Average Win (1.00L)": f"+${avg_w:.2f}",
            "Cumulative Net Profit": f"+${tot_pnl:,.2f}"
        })

    df_res = pd.DataFrame(results)
    print("="*115)
    print("NYH21_MT5 ENHANCEMENT MATRIX (WIN RATE & EVENT FREQUENCY EXPANSION)")
    print("="*115)
    print(df_res.to_string(index=False))
    print("="*115)

if __name__ == "__main__":
    main()
