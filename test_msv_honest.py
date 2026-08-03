#!/usr/bin/env python3
"""MSV Asian FX Network Dispersion Exhaustion (Strategy #5) Honest Backtest."""
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
    print("Loading & pre-aligning 18-pair M5 dataset for MSV Asian Exhaustion...")
    raw, pre_align = load_and_align()
    print(f"  Loaded {len(raw)} pairs, {len(pre_align):,} bars ({time.time()-t0:.1f}s)")

    # Build dataframe matrix
    pieces = [df.set_index("time")[["close","open"]] for df in raw.values()]
    for i, p in enumerate(raw.keys()):
        pieces[i].columns = [p, f"{p}_open"]
    df_all = pd.concat(pieces, axis=1, sort=True).ffill().bfill()
    times = pd.to_datetime(df_all.index)

    close_mat = df_all[[p for p in ALL_PAIRS]].values
    open_mat = df_all[[f"{p}_open" for p in ALL_PAIRS]].values
    hours = times.hour.values

    # Pre-compute 60-min basket returns (12 bars)
    ret_60m = np.zeros_like(close_mat)
    ret_60m[12:] = np.log(close_mat[12:] / close_mat[:-12])

    print("\n" + "="*85)
    print("STRATEGY #5: MSV ASIAN FX NETWORK DISPERSION EXHAUSTION (5-BROKER AUDIT)")
    print("="*85)

    hold_bars = 12 # 60 minutes
    broker_rows = []

    for b in BROKERS:
        sim = ExecutionSimulator(b)
        n_bars = len(df_all)
        in_pos = [False] * len(ALL_PAIRS)
        exit_bar = [0] * len(ALL_PAIRS)
        entry_pr = [0.0] * len(ALL_PAIRS)
        
        pnl_list = []

        for t in range(13, n_bars):
            # Exit checks
            for p_i in range(len(ALL_PAIRS)):
                if in_pos[p_i] and t >= exit_bar[p_i]:
                    c_exit = close_mat[t, p_i]
                    c_entry = entry_pr[p_i]
                    p_name = ALL_PAIRS[p_i]
                    # 0.5 lot PnL
                    gross_pnl = (c_exit - c_entry) / c_entry * 50000.0
                    comm = sim.profile.commission_per_lot * 0.5
                    net_pnl = gross_pnl - comm
                    pnl_list.append(net_pnl)
                    in_pos[p_i] = False

            # Entry check (Asian hours 00:00 to 07:00 UTC)
            if hours[t] < 0 or hours[t] > 6:
                continue

            r_t = ret_60m[t]
            mean_r = np.mean(r_t)

            # Condition 1: Basket in decline <= -0.02%
            if mean_r > -0.0002:
                continue

            # Condition 2: Network Dispersion >= 0.0012
            dispersion = np.std(r_t, ddof=0)
            if dispersion < 0.0012:
                continue

            # Enter LONG on declined pairs
            for p_i in range(len(ALL_PAIRS)):
                if r_t[p_i] < mean_r and not in_pos[p_i]:
                    in_pos[p_i] = True
                    exit_bar[p_i] = t + hold_bars
                    entry_pr[p_i] = open_mat[t, p_i]

        wins = [p for p in pnl_list if p > 0]
        losses = [p for p in pnl_list if p < 0]
        gw = sum(wins)
        gl = abs(sum(losses))
        net = sum(pnl_list)
        wr = len(wins) / len(pnl_list) * 100 if pnl_list else 0.0
        pf = gw / gl if gl > 0 else 0.0
        avg_win = net / len(pnl_list) if pnl_list else 0.0

        broker_rows.append({
            "Broker": b.upper(),
            "Trades": len(pnl_list),
            "Win Rate": f"{wr:.1f}%",
            "Net PnL": f"+${net:.2f}" if net > 0 else f"-${abs(net):.2f}",
            "PF": round(pf, 2),
            "Avg$/Trade": f"+${avg_win:.2f}",
            "Status": "PASS" if net > 0 and pf > 1.0 else "FAIL"
        })

    print(pd.DataFrame(broker_rows).to_string(index=False))

if __name__ == "__main__":
    main()
