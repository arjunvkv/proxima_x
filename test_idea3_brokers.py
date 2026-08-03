#!/usr/bin/env python3
"""Idea 3: Cross-Pair Momentum Shock Continuation (Z >= 4.5) 5-Broker Audit."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd
import numpy as np
from proxima_honest_backtest.strategies.sunday_h22.sweep import load_and_align, BROKERS
from proxima_honest_backtest.execution.execution_simulator import ExecutionSimulator

PAIRS = ["EURAUD", "GBPAUD", "GBPNZD"]

def main():
    t0 = time.time()
    print("Loading M5 dataset for Idea 3 (Cross-Pair Momentum Continuation) 5-Broker Audit...")
    raw, pre_align = load_and_align()
    pieces = [df.set_index("time")[["close","open"]] for df in raw.values()]
    for i, p in enumerate(raw.keys()):
        pieces[i].columns = [p, f"{p}_open"]
    df_all = pd.concat(pieces, axis=1, sort=True).ffill().bfill()
    times = pd.to_datetime(df_all.index)

    close_mat = df_all[[p for p in PAIRS]].values
    open_mat = df_all[[f"{p}_open" for p in PAIRS]].values

    # Pre-compute Z-scores
    z_df = pd.DataFrame(index=df_all.index)
    for p in PAIRS:
        s_c = df_all[p]
        s_r = np.log(s_c / s_c.shift(3))
        s_m = s_r.shift(1).rolling(200).mean()
        s_s = s_r.shift(1).rolling(200).std(ddof=0)
        z_df[p] = (s_r - s_m) / s_s
    z_mat = z_df.values

    print("\n" + "="*85)
    print("IDEA 3: CROSS-PAIR MOMENTUM SHOCK CONTINUATION (Z >= 4.5) — 5-BROKER AUDIT")
    print("="*85)

    hold_bars = 9 # 45-min hold
    broker_rows = []

    for b in BROKERS:
        sim = ExecutionSimulator(b)
        n_bars = len(df_all)
        in_pos = [False] * len(PAIRS)
        exit_bar = [0] * len(PAIRS)
        entry_pr = [0.0] * len(PAIRS)
        direction = [0] * len(PAIRS)
        
        pnl_list = []

        for t in range(205, n_bars):
            # Check exits
            for p_i in range(len(PAIRS)):
                if in_pos[p_i] and t >= exit_bar[p_i]:
                    c_exit = close_mat[t, p_i]
                    c_entry = entry_pr[p_i]
                    dir_i = direction[p_i]
                    # 0.50 Lot PnL
                    gross_pnl = (c_exit - c_entry) / c_entry * 50000.0 * dir_i
                    comm = sim.profile.commission_per_lot * 0.5
                    net_pnl = gross_pnl - comm
                    pnl_list.append(net_pnl)
                    in_pos[p_i] = False

            # Check momentum trigger (Z >= 4.5 buy trend, Z <= -4.5 sell trend)
            for p_i in range(len(PAIRS)):
                z_val = z_mat[t, p_i]
                if abs(z_val) >= 4.5 and not in_pos[p_i]:
                    in_pos[p_i] = True
                    direction[p_i] = 1 if z_val > 0 else -1 # Trend continuation!
                    exit_bar[p_i] = t + hold_bars
                    entry_pr[p_i] = open_mat[t, p_i]

        wins = [p for p in pnl_list if p > 0]
        losses = [p for p in pnl_list if p < 0]
        gw = sum(wins)
        gl = abs(sum(losses))
        net = sum(pnl_list)
        wr = len(wins) / len(pnl_list) * 100 if pnl_list else 0.0
        pf = gw / gl if gl > 0 else 0.0
        avg_w = net / len(pnl_list) if pnl_list else 0.0

        broker_rows.append({
            "Broker": b.upper(),
            "Trades": len(pnl_list),
            "Win Rate": f"{wr:.1f}%",
            "Net PnL": f"+${net:.2f}" if net > 0 else f"-${abs(net):.2f}",
            "PF": round(pf, 2),
            "Avg$/Trade": f"+${avg_w:.2f}",
            "Status": "PASS" if net > 0 and pf > 1.2 else "FAIL"
        })

    print(pd.DataFrame(broker_rows).to_string(index=False))

if __name__ == "__main__":
    main()
