#!/usr/bin/env python3
"""Direct Vectorized CZR Evaluator."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd
import numpy as np
from proxima_honest_backtest.strategies.sunday_h22.sweep import MT5Provider
from proxima_honest_backtest.execution.execution_simulator import ExecutionSimulator

ALL_PAIRS = [
    "AUDJPY", "AUDNZD", "AUDUSD", "EURAUD", "EURCHF", "EURGBP",
    "EURJPY", "EURNZD", "EURUSD", "GBPAUD", "GBPCAD", "GBPJPY",
    "GBPNZD", "GBPUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY",
]
MONTHS = [(2026,m) for m in range(1, 8)]

def main():
    t0 = time.time()
    provider = MT5Provider()
    raw = {}
    for p in ALL_PAIRS:
        frames = [f for f in [provider.load_rates(p, y, m, "m5") for y,m in MONTHS] if not f.empty]
        if frames:
            d = pd.concat(frames, ignore_index=True)
            d.sort_values("time", inplace=True); d.reset_index(drop=True, inplace=True)
            raw[p] = d
    
    # Pre-align prices
    pieces = [df.set_index("time")[["close","open"]] for df in raw.values()]
    for i, p in enumerate(raw.keys()):
        pieces[i].columns = [p, f"{p}_open"]
    df_all = pd.concat(pieces, axis=1, sort=True).ffill().bfill()
    times = df_all.index.values

    # Pre-compute Z-scores for all pairs
    z_df = pd.DataFrame(index=df_all.index)
    for p in ALL_PAIRS:
        s_c = df_all[p]
        s_r = np.log(s_c / s_c.shift(3))
        s_m = s_r.shift(1).rolling(200).mean()
        s_s = s_r.shift(1).rolling(200).std(ddof=0)
        z_df[p] = (s_r - s_m) / s_s
    
    z_mat = z_df.values
    close_mat = df_all[[p for p in ALL_PAIRS]].values
    open_mat = df_all[[f"{p}_open" for p in ALL_PAIRS]].values
    pair_names = ALL_PAIRS

    print(f"Data Loaded: {len(df_all):,} bars ({time.time()-t0:.1f}s)")
    print("\n" + "="*85)
    print("CZR STRATEGY — DIRECT VECTORIZED EVALUATION (5-BROKER SURVIVAL)")
    print("="*85)

    configs = [(4.0, 9), (4.0, 12), (3.0, 9), (3.0, 12)]
    brokers = ["exness", "ftmo", "fundednext", "fusionmarkets", "dukascopy"]

    for z_thresh, hold_bars in configs:
        print(f"\n--- CONFIG: z>={z_thresh} | hold={hold_bars*5}min ({hold_bars} bars) ---")
        broker_rows = []
        for b in brokers:
            sim = ExecutionSimulator(b)
            n_bars = len(df_all)
            in_pos = [False] * len(ALL_PAIRS)
            exit_bar = [0] * len(ALL_PAIRS)
            entry_pr = [0.0] * len(ALL_PAIRS)
            
            pnl_list = []
            
            for t in range(205, n_bars):
                # Check exits
                for p_i in range(len(ALL_PAIRS)):
                    if in_pos[p_i] and t >= exit_bar[p_i]:
                        c_exit = close_mat[t, p_i]
                        c_entry = entry_pr[p_i]
                        p_name = pair_names[p_i]
                        # 0.5 lot size PnL
                        gross_pnl = (c_exit - c_entry) / c_entry * 50000.0
                        comm = sim.profile.commission_per_lot * 0.5
                        net_pnl = gross_pnl - comm
                        pnl_list.append(net_pnl)
                        in_pos[p_i] = False

                # Entry check
                z_vals = z_mat[t]
                min_i = np.argmin(z_vals)
                min_z = z_vals[min_i]
                
                if min_z <= -z_thresh and not in_pos[min_i]:
                    in_pos[min_i] = True
                    exit_bar[min_i] = t + hold_bars
                    entry_pr[min_i] = open_mat[t, min_i]

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
