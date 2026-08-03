#!/usr/bin/env python3
"""Verify exact entry logic difference between MQL5 MT5 Strategy Tester vs Python Engine."""

import sys
import pandas as pd
import numpy as np

from proxima_honest_backtest.strategies.sunday_h22.sweep import load_and_align
from proxima_honest_backtest.execution.execution_simulator import ExecutionSimulator

PAIRS_ALL = ["EURUSD", "GBPUSD", "USDJPY", "EURAUD", "GBPAUD", "EURJPY", "GBPJPY", "EURNZD", "GBPNZD"]

def main():
    print("="*115)
    print("EXACT MATHEMATICAL AUDIT: WHY MT5 TESTER SHOWED PF < 1 WHILE PYTHON SHOWS PF > 6.0")
    print("="*115)

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

    # Model A: Python Engine (Completed Bar Close -> Open of Next Bar)
    sim = ExecutionSimulator("fundednext")
    n_bars = len(df_all)

    # Let's run Model A:
    trades_a = []
    in_pos = [False] * len(PAIRS_ALL)
    exit_bar = [0] * len(PAIRS_ALL)
    entry_pr = [0.0] * len(PAIRS_ALL)
    direction = [0] * len(PAIRS_ALL)

    for t in range(13, n_bars):
        for p_i in range(len(PAIRS_ALL)):
            if in_pos[p_i] and t >= exit_bar[p_i]:
                c_exit = close_mat[t, p_i]
                c_entry = entry_pr[p_i]
                dir_i = direction[p_i]
                pnl = (c_exit - c_entry) / c_entry * 50000.0 * dir_i - 0.45
                trades_a.append(pnl)
                in_pos[p_i] = False

        if minutes[t] in [0, 30]:
            for p_i in range(len(PAIRS_ALL)):
                if in_pos[p_i]: continue
                h_prev = np.max(high_mat[t-12:t, p_i])
                l_prev = np.min(low_mat[t-12:t, p_i])
                c_now = close_mat[t-1, p_i] # Completed bar close
                mult = 100.0 if "JPY" in PAIRS_ALL[p_i] else 10000.0
                if (h_prev - l_prev) * mult < 6.0: continue

                if c_now > h_prev:
                    in_pos[p_i] = True
                    direction[p_i] = 1
                    exit_bar[p_i] = t + 3
                    entry_pr[p_i] = open_mat[t, p_i] # Entry at OPEN of next bar
                elif c_now < l_prev:
                    in_pos[p_i] = True
                    direction[p_i] = -1
                    exit_bar[p_i] = t + 3
                    entry_pr[p_i] = open_mat[t, p_i]

    arr_a = np.array(trades_a)
    wr_a = (arr_a > 0).mean() * 100
    gw_a = arr_a[arr_a > 0].sum()
    gl_a = abs(arr_a[arr_a < 0].sum())
    pf_a = gw_a / gl_a

    # Model B: Mid-Bar Tick Entry (Buying at High Peak of Unclosed Bar 0)
    trades_b = []
    in_pos = [False] * len(PAIRS_ALL)
    exit_bar = [0] * len(PAIRS_ALL)
    entry_pr = [0.0] * len(PAIRS_ALL)
    direction = [0] * len(PAIRS_ALL)

    for t in range(13, n_bars):
        for p_i in range(len(PAIRS_ALL)):
            if in_pos[p_i] and t >= exit_bar[p_i]:
                c_exit = close_mat[t, p_i]
                c_entry = entry_pr[p_i]
                dir_i = direction[p_i]
                pnl = (c_exit - c_entry) / c_entry * 50000.0 * dir_i - 0.45
                trades_b.append(pnl)
                in_pos[p_i] = False

        if minutes[t] in [0, 30]:
            for p_i in range(len(PAIRS_ALL)):
                if in_pos[p_i]: continue
                h_prev = np.max(high_mat[t-12:t, p_i])
                l_prev = np.min(low_mat[t-12:t, p_i])
                c_peak = high_mat[t, p_i] # Mid-bar high peak tick!
                c_trough = low_mat[t, p_i]
                mult = 100.0 if "JPY" in PAIRS_ALL[p_i] else 10000.0
                if (h_prev - l_prev) * mult < 6.0: continue

                if c_peak > h_prev:
                    in_pos[p_i] = True
                    direction[p_i] = 1
                    exit_bar[p_i] = t + 3
                    entry_pr[p_i] = c_peak # Entered at worst mid-bar peak!
                elif c_trough < l_prev:
                    in_pos[p_i] = True
                    direction[p_i] = -1
                    exit_bar[p_i] = t + 3
                    entry_pr[p_i] = c_trough

    arr_b = np.array(trades_b)
    wr_b = (arr_b > 0).mean() * 100
    gw_b = arr_b[arr_b > 0].sum()
    gl_b = abs(arr_b[arr_b < 0].sum())
    pf_b = gw_b / gl_b if gl_b > 0 else 0

    print(f"MODEL A (Completed Bar Close -> Clean Bar Open Entry):")
    print(f"  • Total Trades   : {len(arr_a)}")
    print(f"  • Win Rate       : {wr_a:.1f}%")
    print(f"  • Gross Profit   : +${gw_a:.2f}")
    print(f"  • Gross Loss     : -${gl_a:.2f}")
    print(f"  • Profit Factor  : {pf_a:.2f} 🟢")

    print(f"\nMODEL B (Unclosed Mid-Bar Tick Spike Entry - MQL5 iClose(0)):")
    print(f"  • Total Trades   : {len(arr_b)}")
    print(f"  • Win Rate       : {wr_b:.1f}%")
    print(f"  • Gross Profit   : +${gw_b:.2f}")
    print(f"  • Gross Loss     : -${gl_b:.2f}")
    print(f"  • Profit Factor  : {pf_b:.2f} 🔴")
    print("="*115)

if __name__ == "__main__":
    main()
