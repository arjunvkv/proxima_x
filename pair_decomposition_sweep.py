#!/usr/bin/env python3
"""Exhaustive Pair-by-Pair Decomposition Sweep on All Failed Ideas."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd
import numpy as np
from proxima_honest_backtest.strategies.sunday_h22.sweep import load_and_align
from proxima_honest_backtest.execution.execution_simulator import ExecutionSimulator

ALL_PAIRS = [
    "AUDJPY", "AUDNZD", "AUDUSD", "EURAUD", "EURCHF", "EURGBP",
    "EURJPY", "EURNZD", "EURUSD", "GBPAUD", "GBPCAD", "GBPJPY",
    "GBPNZD", "GBPUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY",
]

def eval_pair_idea(df_all, close_mat, open_mat, hours, minutes, weekdays, p_i, trigger_fn, hold_bars=12):
    sim = ExecutionSimulator("fundednext")
    n_bars = len(df_all)
    in_pos = False
    exit_bar = 0
    entry_pr = 0.0
    direction = 0
    
    pnl_list = []

    for t in range(205, n_bars):
        if in_pos and t >= exit_bar:
            c_exit = close_mat[t, p_i]
            c_entry = entry_pr
            p_name = ALL_PAIRS[p_i]
            gross_pnl = (c_exit - c_entry) / c_entry * 50000.0 * direction
            comm = sim.profile.commission_per_lot * 0.5
            net_pnl = gross_pnl - comm
            pnl_list.append(net_pnl)
            in_pos = False

        if not in_pos:
            trig_dir = trigger_fn(t, p_i)
            if trig_dir != 0:
                in_pos = True
                direction = trig_dir
                exit_bar = t + hold_bars
                entry_pr = open_mat[t, p_i]

    if not pnl_list:
        return 0, 0.0, 0.0, 0.0

    wins = [p for p in pnl_list if p > 0]
    losses = [p for p in pnl_list if p < 0]
    gw = sum(wins)
    gl = abs(sum(losses))
    net = sum(pnl_list)
    wr = len(wins) / len(pnl_list) * 100.0
    pf = gw / gl if gl > 0 else (99.0 if gw > 0 else 0.0)
    return len(pnl_list), wr, net, pf

def main():
    t0 = time.time()
    print("Loading M5 dataset for Exhaustive Pair Decomposition Sweep...")
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

    # Pre-compute Z-scores for CZR
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
    print("EXHAUSTIVE PER-PAIR DECOMPOSITION SWEEP ACROSS ALL FAILED IDEAS")
    print("="*85)

    ideas = [
        ("CZR (z <= -3.5 LONG)", lambda t, p: 1 if z_mat[t, p] <= -3.5 else 0, 9),
        ("Wed Triple Swap (21:00 UTC)", lambda t, p: (-1 if ret_1h[t, p] >= 0.0005 else 0) if weekdays[t] == 2 and hours[t] == 21 and minutes[t] in [0, 5] else 0, 9),
        ("Tokyo Afternoon (06:00 UTC)", lambda t, p: (-1 if ret_1h[t, p] >= 0.0015 else (1 if ret_1h[t, p] <= -0.0015 else 0)) if hours[t] == 6 and minutes[t] in [0, 5] else 0, 9),
        ("London Open (07:00 UTC)", lambda t, p: (-1 if ret_1h[t, p] >= 0.0015 else (1 if ret_1h[t, p] <= -0.0015 else 0)) if hours[t] == 7 and minutes[t] in [0, 5] else 0, 9),
        ("NY Eur Close (17:00 UTC)", lambda t, p: (-1 if ret_1h[t, p] >= 0.0015 else (1 if ret_1h[t, p] <= -0.0015 else 0)) if hours[t] == 17 and minutes[t] in [0, 5] else 0, 9),
    ]

    all_pair_survivors = []

    for name, trig_fn, hold_b in ideas:
        print(f"\n--- Testing Idea: {name} ---")
        rows = []
        for p_i, p_name in enumerate(ALL_PAIRS):
            n_t, wr, net, pf = eval_pair_idea(df_all, close_mat, open_mat, hours, minutes, weekdays, p_i, trig_fn, hold_b)
            survives = "PASS" if wr >= 60.0 and pf >= 1.50 and n_t >= 10 else "FAIL"
            rows.append({
                "Pair": p_name,
                "Trades": n_t,
                "Win Rate": f"{wr:.1f}%",
                "Net PnL": f"+${net:.2f}" if net > 0 else f"-${abs(net):.2f}",
                "PF": round(pf, 2),
                "Status": survives
            })
            if survives == "PASS":
                all_pair_survivors.append((name, p_name, n_t, wr, net, pf))

        df_p = pd.DataFrame(rows)
        passes = df_p[df_p["Status"] == "PASS"]
        if not passes.empty:
            print("  🟢 SURVIVING PAIRS FOUND:")
            print(passes.to_string(index=False))
        else:
            print("  ❌ 0/18 Pairs Passed (All 18 pairs negative or coin-flip)")

    print("\n" + "="*85)
    print("MASTER SUMMARY OF SURVIVING PAIRS ACROSS ALL IDEAS")
    print("="*85)
    if all_pair_survivors:
        for name, p_name, n_t, wr, net, pf in all_pair_survivors:
            print(f"  🟢 {name:<30} | Pair: {p_name:<7} | Trades: {n_t:<3} | WR: {wr:5.1f}% | Net: +${net:<7.2f} | PF: {pf:.2f}")
    else:
        print("  ❌ ZERO SURVIVING PAIRS FOUND across all ideas and all 18 pairs.")

if __name__ == "__main__":
    main()
