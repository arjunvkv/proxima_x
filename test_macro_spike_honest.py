#!/usr/bin/env python3
"""US Macro Event Post-Spike Fade (NFP/CPI 13:00 UTC) — 5-Broker Honest Backtest."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd
import numpy as np
from proxima_honest_backtest.strategies.sunday_h22.sweep import load_and_align, BROKERS
from proxima_honest_backtest.execution.execution_simulator import ExecutionSimulator

MACRO_PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]

def main():
    t0 = time.time()
    print("Loading M5 dataset for US Macro Post-Spike Fade Audit...")
    raw, pre_align = load_and_align()
    pieces = [df.set_index("time")[["close","open"]] for df in raw.values()]
    for i, p in enumerate(raw.keys()):
        pieces[i].columns = [p, f"{p}_open"]
    df_all = pd.concat(pieces, axis=1, sort=True).ffill().bfill()
    times = pd.to_datetime(df_all.index)

    close_mat = df_all[[p for p in MACRO_PAIRS]].values
    open_mat = df_all[[f"{p}_open" for p in MACRO_PAIRS]].values
    hours = times.hour.values
    minutes = times.minute.values
    weekdays = times.weekday.values

    print("\n" + "="*85)
    print("US MACRO POST-SPIKE FADE (13:00 UTC) — 5-BROKER TRANSACTION COST AUDIT")
    print("="*85)

    hold_bars = 6 # 30-min hold
    broker_rows = []

    for b in BROKERS:
        sim = ExecutionSimulator(b)
        n_bars = len(df_all)
        in_pos = [False] * len(MACRO_PAIRS)
        exit_bar = [0] * len(MACRO_PAIRS)
        entry_pr = [0.0] * len(MACRO_PAIRS)
        direction = [0] * len(MACRO_PAIRS)
        
        pnl_list = []

        for t in range(25, n_bars):
            # Check exits
            for p_i in range(len(MACRO_PAIRS)):
                if in_pos[p_i] and t >= exit_bar[p_i]:
                    c_exit = close_mat[t, p_i]
                    c_entry = entry_pr[p_i]
                    p_name = MACRO_PAIRS[p_i]
                    dir_i = direction[p_i]
                    gross_pnl = (c_exit - c_entry) / c_entry * 50000.0 * dir_i
                    comm = sim.profile.commission_per_lot * 0.5
                    net_pnl = gross_pnl - comm
                    pnl_list.append(net_pnl)
                    in_pos[p_i] = False

            # Check macro news window (13:00 UTC bar, 30-min post-release)
            # Release at 12:30 UTC -> spike measured from 12:30 to 13:00 UTC (6 bars)
            if hours[t] == 13 and minutes[t] in [0, 5]:
                for p_i in range(len(MACRO_PAIRS)):
                    c_curr = close_mat[t, p_i]
                    c_prev = close_mat[t - 6, p_i]
                    if c_curr <= 0 or c_prev <= 0:
                        continue
                    ret_spike = (c_curr - c_prev) / c_prev
                    
                    # If spike exceeds +/-0.25% (~25-35 pips), fade it
                    if abs(ret_spike) >= 0.0025 and not in_pos[p_i]:
                        in_pos[p_i] = True
                        direction[p_i] = -1 if ret_spike > 0 else 1 # Fade direction
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
