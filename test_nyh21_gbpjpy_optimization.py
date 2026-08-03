#!/usr/bin/env python3
"""Test GBPJPY + EURJPY Standalone Optimization for NYH21_MT5."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd
import numpy as np

from proxima_honest_backtest.strategies.sunday_h22.sweep import load_and_align

def run_gbpjpy_nyh_sim(df_all, close_mat, hours, minutes, pairs, target_hours=[21], hold_bars=9):
    trades = []
    
    for t_idx in range(12, len(df_all) - hold_bars):
        if hours[t_idx] in target_hours and minutes[t_idx] == 0:
            for p_idx, pair in enumerate(pairs):
                c_now = close_mat[t_idx, p_idx]
                c_lookback = close_mat[t_idx-12, p_idx]
                c_exit = close_mat[t_idx+hold_bars, p_idx]

                pip_mult = 100.0 if "JPY" in pair else 10000.0
                ret_60m = (c_now - c_lookback) * pip_mult

                # Fade decline
                if ret_60m <= -2.0:
                    c_entry = close_mat[t_idx+1, p_idx]
                    pnl_pip = (c_exit - c_entry) * pip_mult
                    pnl_usd = pnl_pip * 10.0 * 1.00
                    trades.append(pnl_usd)

    return trades

def main():
    print("Auditing GBPJPY Standalone Optimization for NYH21_MT5...")
    raw, pre_align = load_and_align()
    pieces = [df.set_index("time")[["close","open","high","low"]] for df in raw.values()]
    for i, p in enumerate(raw.keys()):
        pieces[i].columns = [p, f"{p}_open", f"{p}_high", f"{p}_low"]
    df_all = pd.concat(pieces, axis=1, sort=True).ffill().bfill()
    times = pd.to_datetime(df_all.index)

    close_mat_gbp = df_all[["GBPJPY"]].values
    close_mat_both = df_all[["GBPJPY", "EURJPY"]].values
    hours = times.hour.values
    minutes = times.minute.values
    tot_days = 148

    # 1. GBPJPY 45m hold (9 bars)
    pnls_g9 = run_gbpjpy_nyh_sim(df_all, close_mat_gbp, hours, minutes, ["GBPJPY"], [21], 9)
    
    # 2. Both 45m hold (9 bars)
    pnls_b9 = run_gbpjpy_nyh_sim(df_all, close_mat_both, hours, minutes, ["GBPJPY", "EURJPY"], [21], 9)

    def stats(t_list):
        n = len(t_list)
        w = [t for t in t_list if t > 0]
        l = [t for t in t_list if t <= 0]
        wr = len(w) / n * 100.0 if n > 0 else 0
        pf = sum(w) / max(1, abs(sum(l)))
        tot = sum(t_list)
        avg_w = np.mean(w) if w else 0
        t_per_day = n / tot_days
        return n, t_per_day, len(w), len(l), wr, pf, tot, avg_w

    ng9, dg9, wg9, lg9, wrg9, pfg9, totg9, avgwg9 = stats(pnls_g9)
    nb9, db9, wb9, lb9, wrb9, pfb9, totb9, avgwb9 = stats(pnls_b9)

    print("="*115)
    print("NYH21_MT5 PAIR-SPECIFIC EDGE OPTIMIZATION REPORT")
    print("="*115)
    print(f"Configuration                        Trades Fired   Trades / Day   Wins 🟢  Losses 🔴  Net Win Rate (%)  Profit Factor  Cumulative Profit")
    print(f"---------------------------------------------------------------------------------------------------")
    print(f"GBPJPY Standalone (45m Hold / 21:00)  {ng9} Trades       {dg9:.2f} / day    {wg9}       {lg9}       {wrg9:.1f}% WR 🟢        {pfg9:.2f} PF 🚀    +${totg9:,.2f}")
    print(f"GBPJPY + EURJPY (45m Hold / 21:00)    {nb9} Trades       {db9:.2f} / day    {wb9}       {lb9}       {wrb9:.1f}% WR 🟢        {pfb9:.2f} PF 🚀    +${totb9:,.2f}")
    print("="*115)

if __name__ == "__main__":
    main()
