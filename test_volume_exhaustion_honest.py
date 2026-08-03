#!/usr/bin/env python3
"""Tick Volume Surge Exhaustion Fade — 5-Broker Honest Backtest."""
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

def main():
    t0 = time.time()
    print("Loading M5 dataset for Volume Exhaustion Surge Audit...")
    raw, pre_align = load_and_align()
    pieces = [df.set_index("time")[["close","open","tick_volume"]] for df in raw.values()]
    for i, p in enumerate(raw.keys()):
        pieces[i].columns = [p, f"{p}_open", f"{p}_volume"]
    df_all = pd.concat(pieces, axis=1, sort=True).ffill().bfill()
    times = pd.to_datetime(df_all.index)

    close_mat = df_all[[p for p in ALL_PAIRS]].values
    open_mat = df_all[[f"{p}_open" for p in ALL_PAIRS]].values
    vol_mat = df_all[[f"{p}_volume" for p in ALL_PAIRS]].values

    print("\n" + "="*85)
    print("TICK VOLUME EXHAUSTION SURGE FADE — 5-BROKER AUDIT")
    print("="*85)

    hold_bars = 6 # 30-min hold
    broker_rows = []

    for b in BROKERS:
        sim = ExecutionSimulator(b)
        n_bars = len(df_all)
        in_pos = [False] * len(ALL_PAIRS)
        exit_bar = [0] * len(ALL_PAIRS)
        entry_pr = [0.0] * len(ALL_PAIRS)
        direction = [0] * len(ALL_PAIRS)
        
        pnl_list = []

        for t in range(55, n_bars):
            # Check exits
            for p_i in range(len(ALL_PAIRS)):
                if in_pos[p_i] and t >= exit_bar[p_i]:
                    c_exit = close_mat[t, p_i]
                    c_entry = entry_pr[p_i]
                    p_name = ALL_PAIRS[p_i]
                    dir_i = direction[p_i]
                    gross_pnl = (c_exit - c_entry) / c_entry * 50000.0 * dir_i
                    comm = sim.profile.commission_per_lot * 0.5
                    net_pnl = gross_pnl - comm
                    pnl_list.append(net_pnl)
                    in_pos[p_i] = False

            # Check volume surge & price spike condition
            for p_i in range(len(ALL_PAIRS)):
                cur_vol = vol_mat[t, p_i]
                avg_vol = np.mean(vol_mat[t-50:t, p_i])
                if avg_vol <= 0:
                    continue
                vol_ratio = cur_vol / avg_vol
                
                c_curr = close_mat[t, p_i]
                c_prev = open_mat[t, p_i]
                if c_curr <= 0 or c_prev <= 0:
                    continue
                bar_ret = (c_curr - c_prev) / c_prev

                # Volume ratio >= 3.5x AND 1-bar return >= 0.15% (~15-20 pips)
                if vol_ratio >= 3.5 and abs(bar_ret) >= 0.0015 and not in_pos[p_i]:
                    in_pos[p_i] = True
                    direction[p_i] = -1 if bar_ret > 0 else 1 # Fade direction
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
