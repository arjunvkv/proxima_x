#!/usr/bin/env python3
"""Quantitative Buffing Audit for Session Momentum Relay (SMR #2)."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd
import numpy as np

from proxima_honest_backtest.strategies.sunday_h22.sweep import load_and_align
from proxima_honest_backtest.execution.execution_simulator import ExecutionSimulator

PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "EURAUD", "GBPAUD", "EURJPY", "GBPJPY", "EURNZD", "GBPNZD"]

def run_smr_simulation(df_all, close_mat, open_mat, hours, minutes, pairs_subset, thresh_pct=0.0035, entry_hour=13, entry_min=30, hold_bars=18):
    sim = ExecutionSimulator("fundednext")
    n_bars = len(df_all)
    pair_indices = [PAIRS.index(p) for p in pairs_subset]

    in_pos = [False] * len(PAIRS)
    exit_bar = [0] * len(PAIRS)
    entry_pr = [0.0] * len(PAIRS)
    direction = [0] * len(PAIRS)
    
    trades = []

    for t in range(75, n_bars):
        # Exits
        for p_i in pair_indices:
            if in_pos[p_i] and t >= exit_bar[p_i]:
                c_exit = close_mat[t, p_i]
                c_entry = entry_pr[p_i]
                dir_i = direction[p_i]
                gross_pnl = (c_exit - c_entry) / c_entry * 50000.0 * dir_i
                comm = sim.profile.commission_per_lot * 0.5
                net_pnl = gross_pnl - comm
                trades.append({"time": pd.to_datetime(df_all.index[t]), "pair": PAIRS[p_i], "net_pnl": net_pnl, "win": net_pnl > 0})
                in_pos[p_i] = False

        # Entry Trigger
        if hours[t] == entry_hour and minutes[t] == entry_min:
            for p_i in pair_indices:
                c_curr = close_mat[t, p_i]
                c_london_start = close_mat[t-72, p_i]
                if c_curr <= 0 or c_london_start <= 0: continue
                ret_london = (c_curr - c_london_start) / c_london_start
                
                if ret_london >= thresh_pct and not in_pos[p_i]:
                    in_pos[p_i] = True
                    direction[p_i] = 1 # BUY trend relay
                    exit_bar[p_i] = t + hold_bars
                    entry_pr[p_i] = open_mat[t, p_i]
                elif ret_london <= -thresh_pct and not in_pos[p_i]:
                    in_pos[p_i] = True
                    direction[p_i] = -1 # SELL trend relay
                    exit_bar[p_i] = t + hold_bars
                    entry_pr[p_i] = open_mat[t, p_i]

    return pd.DataFrame(trades)

def main():
    t0 = time.time()
    print("Loading M5 dataset for SMR Buffing Audit...")
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
    print("SESSION MOMENTUM RELAY (SMR #2) — QUANTITATIVE BUFFING AUDIT")
    print("="*85)

    # 1. Per-Pair Breakdown (Threshold = 0.25%, Entry = 13:00, Hold = 90m)
    print("\n1. PER-PAIR BREAKDOWN (Baseline Config):")
    df_base = run_smr_simulation(df_all, close_mat, open_mat, hours, minutes, PAIRS, 0.0025, 13, 0, 18)
    pair_stats = []
    for p in PAIRS:
        df_p = df_base[df_base["pair"] == p]
        if df_p.empty: continue
        pnls = df_p["net_pnl"].values
        wins = sum(1 for x in pnls if x > 0)
        gw = sum(x for x in pnls if x > 0)
        gl = abs(sum(x for x in pnls if x < 0))
        net = sum(pnls)
        wr = wins / len(pnls) * 100.0
        pf = gw / gl if gl > 0 else 0.0
        pair_stats.append({"Pair": p, "Trades": len(pnls), "Win Rate": f"{wr:.1f}%", "Net PnL": f"+${net:.2f}" if net > 0 else f"-${abs(net):.2f}", "PF": round(pf, 2)})
    print(pd.DataFrame(pair_stats).to_string(index=False))

    # 2. Parameter Sweep Matrix (Threshold x Entry Hour/Min x Hold Bars)
    print("\n2. PARAMETER BUFFING SWEEP MATRIX (Cross Pairs & JPY Crosses):")
    top_pairs = ["EURAUD", "GBPAUD", "EURJPY", "GBPJPY", "GBPNZD"]
    sweep_rows = []
    for th in [0.0025, 0.0035, 0.0045]:
        for entry_h, entry_m in [(13, 0), (13, 30), (14, 0)]:
            for hold in [12, 18, 24]:
                df_sw = run_smr_simulation(df_all, close_mat, open_mat, hours, minutes, top_pairs, th, entry_h, entry_m, hold)
                if df_sw.empty: continue
                pnls = df_sw["net_pnl"].values
                n_t = len(pnls)
                wins = sum(1 for x in pnls if x > 0)
                gw = sum(x for x in pnls if x > 0)
                gl = abs(sum(x for x in pnls if x < 0))
                net = sum(pnls)
                wr = wins / n_t * 100.0
                pf = gw / gl if gl > 0 else 0.0
                avg_w = net / n_t
                sweep_rows.append({
                    "Thresh": f"{th*100:.2f}%",
                    "Time": f"{entry_h:02d}:{entry_m:02d} UTC",
                    "Hold": f"{hold*5}m",
                    "Trades": n_t,
                    "Win Rate": f"{wr:.1f}%",
                    "Net PnL": f"+${net:.2f}" if net > 0 else f"-${abs(net):.2f}",
                    "PF": round(pf, 2),
                    "Avg$/Trade": f"+${avg_w:.2f}",
                    "Status": "PASS (BUFFED)" if wr >= 60.0 and pf >= 2.0 else "NORMAL"
                })
    df_sweep = pd.DataFrame(sweep_rows)
    print(df_sweep.sort_values(by="PF", ascending=False).head(10).to_string(index=False))

if __name__ == "__main__":
    main()
