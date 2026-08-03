#!/usr/bin/env python3
"""NY H21 (21:00 UTC WM Fix Fade) — 5-Broker Honest Backtest."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd
import numpy as np
from proxima_honest_backtest.strategies.sunday_h22.sweep import load_and_align, BROKERS
from proxima_honest_backtest.execution.execution_simulator import ExecutionSimulator

NY_PAIRS = ["EURJPY", "GBPJPY"]

def main():
    t0 = time.time()
    print("Loading M5 dataset for NY H21 Honest Backtest Audit...")
    raw, pre_align = load_and_align()
    pieces = [df.set_index("time")[["close","open"]] for df in raw.values()]
    for i, p in enumerate(raw.keys()):
        pieces[i].columns = [p, f"{p}_open"]
    df_all = pd.concat(pieces, axis=1, sort=True).ffill().bfill()
    times = pd.to_datetime(df_all.index)

    close_mat = df_all[[p for p in NY_PAIRS]].values
    open_mat = df_all[[f"{p}_open" for p in NY_PAIRS]].values
    hours = times.hour.values
    minutes = times.minute.values

    # Pre-compute 18-pair 60-min return matrix for market decline filter
    all_pairs_list = list(raw.keys())
    close_all = df_all[[p for p in all_pairs_list]].values
    ret_all = np.zeros_like(close_all)
    ret_all[12:] = np.log(close_all[12:] / close_all[:-12])

    print("\n" + "="*85)
    print("NY H21 (21:00 UTC / 02:30 AM IST) — 5-BROKER TRANSACTION AUDIT")
    print("="*85)

    hold_bars = 12 # 60-min hold
    broker_rows = []

    for b in BROKERS:
        sim = ExecutionSimulator(b)
        n_bars = len(df_all)
        in_pos = [False] * len(NY_PAIRS)
        exit_bar = [0] * len(NY_PAIRS)
        entry_pr = [0.0] * len(NY_PAIRS)
        max_favorable = [0.0] * len(NY_PAIRS)
        trailed = [False] * len(NY_PAIRS)
        
        pnl_list = []

        for t in range(25, n_bars):
            # Check exits & trailing profit lock (+15 pips trigger -> lock +10 pips)
            for p_i in range(len(NY_PAIRS)):
                if in_pos[p_i]:
                    c_curr = close_mat[t, p_i]
                    c_entry = entry_pr[p_i]
                    p_name = NY_PAIRS[p_i]
                    gain_pips = (c_curr - c_entry) / 0.01

                    # Trailing trigger (+15 pips)
                    if gain_pips >= 15.0:
                        trailed[p_i] = True

                    # Trailing exit check (retrace below +10 pips)
                    if trailed[p_i] and gain_pips <= 10.0:
                        gross_pnl = (c_curr - c_entry) / c_entry * 20000.0 # 0.20 Lot
                        comm = sim.profile.commission_per_lot * 0.20
                        net_pnl = gross_pnl - comm
                        pnl_list.append(net_pnl)
                        in_pos[p_i] = False
                        continue

                    # Time-based expiry exit
                    if t >= exit_bar[p_i]:
                        gross_pnl = (c_curr - c_entry) / c_entry * 20000.0 # 0.20 Lot
                        comm = sim.profile.commission_per_lot * 0.20
                        net_pnl = gross_pnl - comm
                        pnl_list.append(net_pnl)
                        in_pos[p_i] = False

            # Trigger condition: 21:00 UTC (02:30 AM IST)
            if hours[t] == 21 and minutes[t] in [0, 5]:
                # 18-pair market decline gate: at least 8 pairs declining
                decl_cnt = np.sum(ret_all[t] < 0)
                if decl_cnt < 8:
                    continue

                for p_i in range(len(NY_PAIRS)):
                    c_curr = close_mat[t, p_i]
                    c_prev = close_mat[t - 12, p_i]
                    if c_curr <= 0 or c_prev <= 0:
                        continue
                    ret_1h = (c_curr - c_prev) / c_prev
                    
                    if ret_1h < -0.0001 and not in_pos[p_i]:
                        in_pos[p_i] = True
                        exit_bar[p_i] = t + hold_bars
                        entry_pr[p_i] = open_mat[t, p_i]
                        trailed[p_i] = False

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
