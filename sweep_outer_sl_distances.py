#!/usr/bin/env python3
"""Sweep Hard SL Distances to find exact threshold for 100% Win Rate preservation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd
import numpy as np

from proxima_honest_backtest.strategies.sunday_h22.sweep import load_and_align
from ultra_buff_rolling_orb import run_ultra_buffed_orb

PAIRS_ALL = ["EURUSD", "GBPUSD", "USDJPY", "EURAUD", "GBPAUD", "EURJPY", "GBPJPY", "EURNZD", "GBPNZD"]

def run_sl_sweep_sim(df_all, close_mat, open_mat, high_mat, low_mat, hours, minutes, pairs, sl_pips):
    trades = []
    n_bars = len(df_all)

    for t_idx in range(30, n_bars - 3):
        h = hours[t_idx]
        m = minutes[t_idx]

        if m not in [0, 30]:
            continue

        for p_idx, pair in enumerate(pairs):
            orb_high = np.max(high_mat[t_idx-6:t_idx, p_idx])
            orb_low = np.min(low_mat[t_idx-6:t_idx, p_idx])
            
            pip_mult = 100.0 if "JPY" in pair else 10000.0
            range_pips = (orb_high - orb_low) * pip_mult

            if range_pips < 12.0:
                continue

            c_now = close_mat[t_idx, p_idx]
            c_entry = open_mat[t_idx+1, p_idx]
            
            max_h_hold = np.max(high_mat[t_idx+1:t_idx+1+3, p_idx])
            min_l_hold = np.min(low_mat[t_idx+1:t_idx+1+3, p_idx])
            c_exit_timed = close_mat[t_idx+1+3, p_idx]

            buf = 1.0 / pip_mult

            if c_now > (orb_high + buf):
                pnl_sl = -sl_pips * 10.0 * 1.00
                pnl_timed = (c_exit_timed - c_entry) * pip_mult * 10.0 * 1.00

                if (min_l_hold - c_entry) * pip_mult <= -sl_pips:
                    trades.append(pnl_sl)
                else:
                    trades.append(pnl_timed)

            elif c_now < (orb_low - buf):
                pnl_sl = -sl_pips * 10.0 * 1.00
                pnl_timed = (c_entry - c_exit_timed) * pip_mult * 10.0 * 1.00

                if (c_entry - max_h_hold) * pip_mult <= -sl_pips:
                    trades.append(pnl_sl)
                else:
                    trades.append(pnl_timed)

    return trades

def main():
    print("Sweeping Outer SL Distances for 100% Win Rate Preservation...")
    raw, pre_align = load_and_align()
    pieces = [df.set_index("time")[["close","open","high","low"]] for df in raw.values()]
    for i, p in enumerate(raw.keys()):
        pieces[i].columns = [p, f"{p}_open", f"{p}_high", f"{p}_low"]
    df_all = pd.concat(pieces, axis=1, sort=True).ffill().bfill()

    close_mat = df_all[[p for p in PAIRS_ALL]].values
    open_mat = df_all[[f"{p}_open" for p in PAIRS_ALL]].values
    high_mat = df_all[[f"{p}_high" for p in PAIRS_ALL]].values
    low_mat = df_all[[f"{p}_low" for p in PAIRS_ALL]].values
    hours = pd.to_datetime(df_all.index).hour.values
    minutes = pd.to_datetime(df_all.index).minute.values

    # Baseline (No hard SL)
    df_u_base = run_ultra_buffed_orb(df_all, close_mat, open_mat, high_mat, low_mat, hours, minutes, PAIRS_ALL, 12.0, range(0, 24), [0, 30], 3)
    pnls_base = df_u_base["net_pnl"] / 0.15 * 1.00
    n_b = len(pnls_base)
    w_b = sum(1 for p in pnls_base if p > 0)
    wr_b = w_b / n_b * 100.0
    tot_b = sum(pnls_base)

    sl_levels = [40, 50, 60, 80, 100, 120, 150]
    results = []

    results.append({
        "SL Distance": "No SL Baseline",
        "Total Trades": n_b,
        "Net Win Rate (%)": f"{wr_b:.1f}%",
        "Cumulative Profit": f"+${tot_b:,.2f}",
        "Variance from Baseline": "Baseline Reference"
    })

    for sl in sl_levels:
        pnls_sl = run_sl_sweep_sim(df_all, close_mat, open_mat, high_mat, low_mat, hours, minutes, PAIRS_ALL, float(sl))
        n_s = len(pnls_sl)
        w_s = sum(1 for p in pnls_sl if p > 0)
        wr_s = w_s / n_s * 100.0 if n_s > 0 else 0
        tot_s = sum(pnls_sl)

        results.append({
            "SL Distance": f"{sl}.0 Pips Hard SL",
            "Total Trades": n_s,
            "Net Win Rate (%)": f"{wr_s:.1f}%",
            "Cumulative Profit": f"+${tot_s:,.2f}",
            "Variance from Baseline": f"{wr_s - wr_b:+.1f}% Win Rate"
        })

    df_res = pd.DataFrame(results)
    print("="*115)
    print("OUTER SL DISTANCE SWEEP: ULTRA_MONSTER_MT5 (PRESERVING 74.9% WIN RATE)")
    print("="*115)
    print(df_res.to_string(index=False))
    print("="*115)

if __name__ == "__main__":
    main()
