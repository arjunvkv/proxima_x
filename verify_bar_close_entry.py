#!/usr/bin/env python3
"""Verify exact Bar-Close Entry logic vs Mid-Bar Tick Entry logic."""

import sys
import pandas as pd
import numpy as np

from proxima_honest_backtest.strategies.sunday_h22.sweep import load_and_align

PAIRS_ALL = ["EURUSD", "GBPUSD", "USDJPY", "EURAUD", "GBPAUD", "EURJPY", "GBPJPY", "EURNZD", "GBPNZD"]

def main():
    print("="*115)
    print("EXACT DISCREPANCY AUDIT: CLOSED BAR ENTRY (PF > 6.0) vs MID-BAR TICK SPIKE ENTRY (PF < 1.0)")
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
    n_bars = len(df_all)

    # 1. Closed Bar Entry Model (Evaluating Bar t-1, entering at Open of Bar t)
    trades_closed = []
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
                trades_closed.append(pnl)
                in_pos[p_i] = False

        if minutes[t] in [0, 30]:
            for p_i in range(len(PAIRS_ALL)):
                if in_pos[p_i]: continue
                h_prev = np.max(high_mat[t-13:t-1, p_i])
                l_prev = np.min(low_mat[t-13:t-1, p_i])
                c_closed = close_mat[t-1, p_i]
                mult = 100.0 if "JPY" in PAIRS_ALL[p_i] else 10000.0
                if (h_prev - l_prev) * mult < 6.0: continue

                if c_closed > h_prev:
                    in_pos[p_i] = True
                    direction[p_i] = 1
                    exit_bar[p_i] = t + 3
                    entry_pr[p_i] = open_mat[t, p_i]
                elif c_closed < l_prev:
                    in_pos[p_i] = True
                    direction[p_i] = -1
                    exit_bar[p_i] = t + 3
                    entry_pr[p_i] = open_mat[t, p_i]

    arr_c = np.array(trades_closed)
    wr_c = (arr_c > 0).mean() * 100
    gw_c = arr_c[arr_c > 0].sum()
    gl_c = abs(arr_c[arr_c < 0].sum())
    pf_c = gw_c / gl_c

    # 2. Mid-Bar Tick Spike Model (Evaluating Bar 0 during active tick)
    trades_spike = []
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
                trades_spike.append(pnl)
                in_pos[p_i] = False

        if minutes[t] in [0, 30]:
            for p_i in range(len(PAIRS_ALL)):
                if in_pos[p_i]: continue
                h_prev = np.max(high_mat[t-12:t, p_i])
                l_prev = np.min(low_mat[t-12:t, p_i])
                c_spike = high_mat[t, p_i] # Mid-bar high peak tick
                c_trough = low_mat[t, p_i]
                mult = 100.0 if "JPY" in PAIRS_ALL[p_i] else 10000.0
                if (h_prev - l_prev) * mult < 6.0: continue

                if c_spike > h_prev:
                    in_pos[p_i] = True
                    direction[p_i] = 1
                    exit_bar[p_i] = t + 3
                    entry_pr[p_i] = c_spike # Mid-bar peak tick
                elif c_trough < l_prev:
                    in_pos[p_i] = True
                    direction[p_i] = -1
                    exit_bar[p_i] = t + 3
                    entry_pr[p_i] = c_trough

    arr_s = np.array(trades_spike)
    wr_s = (arr_s > 0).mean() * 100
    gw_s = arr_s[arr_s > 0].sum()
    gl_s = abs(arr_s[arr_s < 0].sum())
    pf_s = gw_s / gl_s

    print(f"1. CLOSED BAR ENTRY (Bar t-1 Close -> Bar t Open Entry):")
    print(f"   • Total Trades  : {len(arr_c)}")
    print(f"   • Win Rate      : {wr_c:.1f}% 🟢")
    print(f"   • Gross Profit  : +${gw_c:,.2f}")
    print(f"   • Gross Loss    : -${gl_c:,.2f}")
    print(f"   • Net PnL       : +${gw_c - gl_c:,.2f}")
    print(f"   • Profit Factor : {pf_c:.2f} 🚀")

    print(f"\n2. UNCLOSED MID-BAR TICK SPIKE ENTRY (iClose(0) / Mid-Bar Tick Spike):")
    print(f"   • Total Trades  : {len(arr_s)}")
    print(f"   • Win Rate      : {wr_s:.1f}% 🔴")
    print(f"   • Gross Profit  : +${gw_s:,.2f}")
    print(f"   • Gross Loss    : -${gl_s:,.2f}")
    print(f"   • Net PnL       : -${gl_s - gw_s:,.2f}")
    print(f"   • Profit Factor : {pf_s:.2f} 🔴")
    print("="*115)

if __name__ == "__main__":
    main()
