#!/usr/bin/env python3
"""Correlation Breakdown (Laggard Catch-Up Pairs Trading) — 5-Broker Honest Backtest."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd
import numpy as np
from proxima_honest_backtest.strategies.sunday_h22.sweep import load_and_align, BROKERS
from proxima_honest_backtest.execution.execution_simulator import ExecutionSimulator

def main():
    t0 = time.time()
    print("Loading M5 dataset for Correlation Breakdown Audit...")
    raw, pre_align = load_and_align()
    pieces = [df.set_index("time")[["close","open"]] for df in raw.values()]
    for i, p in enumerate(raw.keys()):
        pieces[i].columns = [p, f"{p}_open"]
    df_all = pd.concat(pieces, axis=1, sort=True).ffill().bfill()
    times = pd.to_datetime(df_all.index)

    # Test pair 1: AUDUSD (leader) vs NZDUSD (laggard)
    # Test pair 2: EURJPY (leader) vs GBPJPY (laggard)
    pair_tuples = [
        ("AUDUSD", "NZDUSD"),
        ("EURJPY", "GBPJPY")
    ]

    print("\n" + "="*85)
    print("#8 CORRELATION BREAKDOWN (LAGGARD CATCH-UP) — 5-BROKER TRANSACTION AUDIT")
    print("="*85)

    hold_bars = 6 # 30-min hold
    broker_rows = []

    for b in BROKERS:
        sim = ExecutionSimulator(b)
        n_bars = len(df_all)
        pnl_list = []

        for lead_p, lag_p in pair_tuples:
            c_lead = df_all[lead_p].values
            c_lag = df_all[lag_p].values
            o_lag = df_all[f"{lag_p}_open"].values

            ret_lead_15m = np.zeros(n_bars)
            ret_lag_15m = np.zeros(n_bars)
            ret_lead_15m[3:] = (c_lead[3:] - c_lead[:-3]) / c_lead[:-3]
            ret_lag_15m[3:] = (c_lag[3:] - c_lag[:-3]) / c_lag[:-3]

            in_pos = False
            exit_bar = 0
            entry_pr = 0.0
            direction = 0

            for t in range(205, n_bars):
                if in_pos and t >= exit_bar:
                    c_exit = c_lag[t]
                    gross_pnl = (c_exit - entry_pr) / entry_pr * 50000.0 * direction
                    comm = sim.profile.commission_per_lot * 0.5
                    net_pnl = gross_pnl - comm
                    pnl_list.append(net_pnl)
                    in_pos = False

                if not in_pos:
                    # Divergence: Leader moved >= +0.15% while Laggard moved <= +0.03%
                    if ret_lead_15m[t] >= 0.0015 and ret_lag_15m[t] <= 0.0003:
                        in_pos = True
                        direction = 1 # BUY laggard (catch-up)
                        exit_bar = t + hold_bars
                        entry_pr = o_lag[t]
                    elif ret_lead_15m[t] <= -0.0015 and ret_lag_15m[t] >= -0.0003:
                        in_pos = True
                        direction = -1 # SELL laggard (catch-down)
                        exit_bar = t + hold_bars
                        entry_pr = o_lag[t]

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
