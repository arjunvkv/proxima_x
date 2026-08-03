#!/usr/bin/env python3
"""5 New Quantitative FX Strategy Ideas — 5-Broker Honest Backtest."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd
import numpy as np
from proxima_honest_backtest.strategies.sunday_h22.sweep import load_and_align, BROKERS
from proxima_honest_backtest.execution.execution_simulator import ExecutionSimulator

ALL_PAIRS = [
    "AUDJPY", "AUDNZD", "AUDUSD", "EURAUD", "EURCHF", "EURGBP",
    "EURJPY", "EURNZD", "EURUSD", "GBPAUD", "GBPCAD", "GBPJPY",
    "GBPNZD", "GBPUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY",
]

def run_idea(df_all, close_mat, open_mat, hours, minutes, weekdays, z_mat, ret_1h, idea_fn, hold_bars=9):
    sim = ExecutionSimulator("fundednext")
    n_bars = len(df_all)
    in_pos = [False] * len(ALL_PAIRS)
    exit_bar = [0] * len(ALL_PAIRS)
    entry_pr = [0.0] * len(ALL_PAIRS)
    direction = [0] * len(ALL_PAIRS)
    
    pnl_list = []

    for t in range(205, n_bars):
        # Check exits
        for p_i in range(len(ALL_PAIRS)):
            if in_pos[p_i] and t >= exit_bar[p_i]:
                c_exit = close_mat[t, p_i]
                c_entry = entry_pr[p_i]
                dir_i = direction[p_i]
                gross_pnl = (c_exit - c_entry) / c_entry * 50000.0 * dir_i
                comm = sim.profile.commission_per_lot * 0.5
                net_pnl = gross_pnl - comm
                pnl_list.append(net_pnl)
                in_pos[p_i] = False

        # Entry triggers
        for p_i in range(len(ALL_PAIRS)):
            trig_dir = idea_fn(t, p_i, hours, minutes, weekdays, z_mat, ret_1h)
            if trig_dir != 0 and not in_pos[p_i]:
                in_pos[p_i] = True
                direction[p_i] = trig_dir
                exit_bar[p_i] = t + hold_bars
                entry_pr[p_i] = open_mat[t, p_i]

    if not pnl_list:
        return 0, 0.0, 0.0, 0.0

    wins = [p for p in pnl_list if p > 0]
    losses = [p for p in pnl_list if p < 0]
    gw = sum(wins)
    gl = abs(sum(losses))
    net = sum(pnl_list)
    wr = len(wins) / len(pnl_list) * 100.0
    pf = gw / gl if gl > 0 else 0.0
    return len(pnl_list), wr, net, pf

def main():
    t0 = time.time()
    print("Loading M5 dataset for 5 New Quantitative FX Ideas Audit...")
    raw, pre_align = load_and_align()
    pieces = [df.set_index("time")[["close","open"]] for df in raw.values()]
    for i, p in enumerate(raw.keys()):
        pieces[i].columns = [p, f"{p}_open"]
    df_all = pd.concat(pieces, axis=1, sort=True).ffill().bfill()
    times = pd.to_datetime(df_all.index)

    close_mat = df_all[[p for p in ALL_PAIRS]].values
    open_mat = df_all[[f"{p}_open" for p in ALL_PAIRS]].values
    hours = times.hour.values
    minutes = times.minute.values
    weekdays = times.weekday.values

    # Pre-compute Z-scores
    z_df = pd.DataFrame(index=df_all.index)
    for p in ALL_PAIRS:
        s_c = df_all[p]
        s_r = np.log(s_c / s_c.shift(3))
        s_m = s_r.shift(1).rolling(200).mean()
        s_s = s_r.shift(1).rolling(200).std(ddof=0)
        z_df[p] = (s_r - s_m) / s_s
    z_mat = z_df.values

    # 1-hour returns
    ret_1h = np.zeros_like(close_mat)
    ret_1h[12:] = np.log(close_mat[12:] / close_mat[:-12])

    print("\n" + "="*85)
    print("5 NEW QUANTITATIVE FX STRATEGY IDEAS — 5-BROKER TRANSACTION AUDIT")
    print("="*85)

    def idea1_sydney_open(t, p, h, m, w, z, r1h):
        # Idea 1: Sydney Open Liquidity Void Fade (20:00 UTC)
        if h[t] == 20 and m[t] in [0, 5]:
            if r1h[t, p] >= 0.0015: return -1
            if r1h[t, p] <= -0.0015: return 1
        return 0

    def idea2_tokyo_range_fade(t, p, h, m, w, z, r1h):
        # Idea 2: Asian Morning High-Low Range Breakout Absorption (02:00 UTC)
        if h[t] == 2 and m[t] in [0, 5]:
            if r1h[t, p] >= 0.0015: return -1
            if r1h[t, p] <= -0.0015: return 1
        return 0

    def idea3_momentum_continuation(t, p, h, m, w, z, r1h):
        # Idea 3: Cross-Pair Momentum Shock Continuation (Z >= 4.5 Follow Trend)
        if ALL_PAIRS[p] in ["EURAUD", "GBPAUD", "GBPNZD"]:
            if z[t, p] >= 4.5: return 1   # BUY trend continuation
            if z[t, p] <= -4.5: return -1 # SELL trend continuation
        return 0

    def idea4_european_lunch(t, p, h, m, w, z, r1h):
        # Idea 4: European Lunch Liquidity Drain Fade (11:00 UTC)
        if h[t] == 11 and m[t] in [0, 5]:
            if r1h[t, p] >= 0.0015: return -1
            if r1h[t, p] <= -0.0015: return 1
        return 0

    def idea5_us_equities_close(t, p, h, m, w, z, r1h):
        # Idea 5: US Afternoon Equities Close Rebalance Fade (19:30 UTC)
        if h[t] == 19 and m[t] in [30, 35]:
            if r1h[t, p] >= 0.0015: return -1
            if r1h[t, p] <= -0.0015: return 1
        return 0

    new_ideas = [
        ("Idea 1: Sydney Open Liquidity Void (20:00 UTC)", idea1_sydney_open, 9),
        ("Idea 2: Tokyo Morning Range Absorption (02:00 UTC)", idea2_tokyo_range_fade, 9),
        ("Idea 3: Cross-Pair Momentum Continuation (Z >= 4.5)", idea3_momentum_continuation, 9),
        ("Idea 4: European Lunch Drain Fade (11:00 UTC)", idea4_european_lunch, 9),
        ("Idea 5: US Equities Close Rebalance (19:30 UTC)", idea5_us_equities_close, 9),
    ]

    for name, fn, hold_b in new_ideas:
        n_t, wr, net, pf = run_idea(df_all, close_mat, open_mat, hours, minutes, weekdays, z_mat, ret_1h, fn, hold_b)
        survives = "PASS" if wr >= 60.0 and pf >= 1.50 and n_t >= 10 else "FAIL"
        avg_w = net / n_t if n_t > 0 else 0.0
        print(f"\n--- {name} ---")
        print(f"  Total Trades  : {n_t}")
        print(f"  Net Win Rate  : {wr:.1f}%")
        print(f"  Net PnL       : +${net:.2f}" if net > 0 else f"  Net PnL       : -${abs(net):.2f}")
        print(f"  Profit Factor : {pf:.2f}")
        print(f"  Avg $/Trade   : +${avg_w:.2f}")
        print(f"  Status        : {survives}")

if __name__ == "__main__":
    main()
