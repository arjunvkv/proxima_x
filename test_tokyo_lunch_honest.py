#!/usr/bin/env python3
"""Tokyo Lunch Break Re-Entry (04:00 UTC / 09:30 AM IST) — 5-Broker Honest Backtest."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd
import numpy as np
from proxima_honest_backtest.strategies.sunday_h22.sweep import load_and_align, BROKERS
from proxima_honest_backtest.execution.execution_simulator import ExecutionSimulator

PAIRS = ["AUDJPY", "EURJPY", "GBPJPY", "USDJPY", "EURAUD", "GBPAUD"]

def main():
    t0 = time.time()
    print("Loading M5 dataset for Tokyo Lunch Re-Entry 04:00 UTC Audit...")
    raw, pre_align = load_and_align()
    pieces = [df.set_index("time")[["close","open"]] for df in raw.values()]
    for i, p in enumerate(raw.keys()):
        pieces[i].columns = [p, f"{p}_open"]
    df_all = pd.concat(pieces, axis=1, sort=True).ffill().bfill()
    times = pd.to_datetime(df_all.index)

    close_mat = df_all[[p for p in PAIRS]].values
    open_mat = df_all[[f"{p}_open" for p in PAIRS]].values
    hours = times.hour.values
    minutes = times.minute.values

    print("\n" + "="*85)
    print("TOKYO LUNCH RE-ENTRY (04:00 UTC / 09:30 AM IST) — 5-BROKER AUDIT")
    print("="*85)

    hold_bars = 12 # 60-min hold
    broker_rows = []

    for b in BROKERS:
        sim = ExecutionSimulator(b)
        n_bars = len(df_all)
        in_pos = [False] * len(PAIRS)
        exit_bar = [0] * len(PAIRS)
        entry_pr = [0.0] * len(PAIRS)
        direction = [0] * len(PAIRS)
        
        pnl_list = []

        for t in range(25, n_bars):
            # Check exits
            for p_i in range(len(PAIRS)):
                if in_pos[p_i] and t >= exit_bar[p_i]:
                    c_exit = close_mat[t, p_i]
                    c_entry = entry_pr[p_i]
                    p_name = PAIRS[p_i]
                    dir_i = direction[p_i]
                    gross_pnl = (c_exit - c_entry) / c_entry * 50000.0 * dir_i
                    comm = sim.profile.commission_per_lot * 0.5
                    net_pnl = gross_pnl - comm
                    pnl_list.append(net_pnl)
                    in_pos[p_i] = False

            # Trigger condition: 04:00 UTC (Tokyo afternoon open post-lunch)
            # 1-hour morning drive from 03:00 to 04:00 UTC (12 M5 bars)
            if hours[t] == 4 and minutes[t] in [0, 5]:
                for p_i in range(len(PAIRS)):
                    c_curr = close_mat[t, p_i]
                    c_prev = close_mat[t - 12, p_i]
                    if c_curr <= 0 or c_prev <= 0:
                        continue
                    ret_1h = (c_curr - c_prev) / c_prev
                    
                    if abs(ret_1h) >= 0.0015 and not in_pos[p_i]:
                        in_pos[p_i] = True
                        direction[p_i] = -1 if ret_1h > 0 else 1 # Fade direction
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
