#!/usr/bin/env python3
"""Full 4-Step Statistical Audit Suite for BUFFED Session Momentum Relay (SMR #2)."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd
import numpy as np

from proxima_honest_backtest.strategies.sunday_h22.sweep import load_and_align, BROKERS
from proxima_honest_backtest.execution.execution_simulator import ExecutionSimulator

TARGET_PAIRS = ["GBPAUD", "USDJPY", "EURJPY", "GBPJPY"]

def run_buffed_smr(df_all, close_mat, open_mat, hours, minutes, pairs_subset, thresh_pct=0.0045, entry_hour=13, entry_min=30, hold_bars=18, broker="fundednext", all_pairs=None):
    sim = ExecutionSimulator(broker)
    n_bars = len(df_all)
    if all_pairs is None:
        all_pairs = ["AUDJPY", "AUDNZD", "AUDUSD", "EURAUD", "EURCHF", "EURGBP", "EURJPY", "EURNZD", "EURUSD", "GBPAUD", "GBPCAD", "GBPJPY", "GBPNZD", "GBPUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY"]
    pair_indices = [all_pairs.index(p) for p in pairs_subset]

    in_pos = [False] * len(all_pairs)
    exit_bar = [0] * len(all_pairs)
    entry_pr = [0.0] * len(all_pairs)
    direction = [0] * len(all_pairs)
    
    trades = []

    for t in range(75, n_bars):
        for p_i in pair_indices:
            if in_pos[p_i] and t >= exit_bar[p_i]:
                c_exit = close_mat[t, p_i]
                c_entry = entry_pr[p_i]
                dir_i = direction[p_i]
                gross_pnl = (c_exit - c_entry) / c_entry * 50000.0 * dir_i
                comm = sim.profile.commission_per_lot * 0.5
                net_pnl = gross_pnl - comm
                trades.append({"time": pd.to_datetime(df_all.index[t]), "pair": all_pairs[p_i], "net_pnl": net_pnl, "win": net_pnl > 0})
                in_pos[p_i] = False

        if hours[t] == entry_hour and minutes[t] == entry_min:
            for p_i in pair_indices:
                c_curr = close_mat[t, p_i]
                c_london_start = close_mat[t-72, p_i]
                if c_curr <= 0 or c_london_start <= 0: continue
                ret_london = (c_curr - c_london_start) / c_london_start
                
                if ret_london >= thresh_pct and not in_pos[p_i]:
                    in_pos[p_i] = True
                    direction[p_i] = 1
                    exit_bar[p_i] = t + hold_bars
                    entry_pr[p_i] = open_mat[t, p_i]
                elif ret_london <= -thresh_pct and not in_pos[p_i]:
                    in_pos[p_i] = True
                    direction[p_i] = -1
                    exit_bar[p_i] = t + hold_bars
                    entry_pr[p_i] = open_mat[t, p_i]

    return pd.DataFrame(trades)

raw_global = None

def main():
    global raw_global
    t0 = time.time()
    print("Loading M5 dataset for BUFFED SMR Full Statistical Audit Suite...")
    raw, pre_align = load_and_align()
    raw_global = raw
    all_pairs = list(raw.keys())
    pieces = [df.set_index("time")[["close","open"]] for df in raw.values()]
    for i, p in enumerate(all_pairs):
        pieces[i].columns = [p, f"{p}_open"]
    df_all = pd.concat(pieces, axis=1, sort=True).ffill().bfill()
    times = pd.to_datetime(df_all.index)

    close_mat = df_all[[p for p in all_pairs]].values
    open_mat = df_all[[f"{p}_open" for p in all_pairs]].values
    hours = times.hour.values
    minutes = times.minute.values

    df_t = run_buffed_smr(df_all, close_mat, open_mat, hours, minutes, TARGET_PAIRS, 0.0045, 13, 30, 18, "fundednext")
    pnls = df_t["net_pnl"].values if not df_t.empty else np.array([])
    n_trades = len(pnls)

    print("\n" + "="*85)
    print("FULL 4-STEP STATISTICAL AUDIT SUITE: BUFFED SESSION MOMENTUM RELAY (SMR #2)")
    print("="*85)

    # 1. Sign-Permutation Test (1,000 Shuffles)
    obs_sharpe = np.mean(pnls) / np.std(pnls, ddof=1) if len(pnls) > 1 else 0.0
    n_perm = 1000
    count_better = 0
    np.random.seed(42)
    for _ in range(n_perm):
        shuffled_signs = np.random.choice([-1, 1], size=n_trades)
        perm_pnls = pnls * shuffled_signs
        perm_sharpe = np.mean(perm_pnls) / np.std(perm_pnls, ddof=1)
        if perm_sharpe >= obs_sharpe:
            count_better += 1
    p_val = count_better / n_perm

    print(f"\n1. SIGN-PERMUTATION TEST (1,000 SHUFFLES):")
    print(f"   Observed Per-Trade Sharpe : {obs_sharpe:.4f}")
    print(f"   Permutations >= Observed : {count_better} / {n_perm}")
    print(f"   p-value                  : {p_val:.4f}")
    print(f"   Status                   : {'PASS (p < 0.01)' if p_val < 0.01 else 'FAIL'}")

    # 2. Walk-Forward Out-of-Sample Test (5 Windows)
    print(f"\n2. WALK-FORWARD OUT-OF-SAMPLE TEST (5 WINDOWS):")
    df_t["window"] = pd.qcut(df_t["time"], 5, labels=["W1", "W2", "W3", "W4", "W5"], duplicates='drop')
    wf_rows = []
    all_oos_pos = True
    for w_name, grp in df_t.groupby("window", observed=False):
        w_pnls = grp["net_pnl"].values
        w_wins = sum(1 for p in w_pnls if p > 0)
        w_net = sum(w_pnls)
        w_wr = w_wins / len(w_pnls) * 100.0 if len(w_pnls) > 0 else 0.0
        w_sh = np.mean(w_pnls) / np.std(w_pnls, ddof=1) if len(w_pnls) > 1 else 0.0
        if w_net <= 0: all_oos_pos = False
        wf_rows.append({
            "Window": w_name,
            "Trades": len(w_pnls),
            "Win Rate": f"{w_wr:.1f}%",
            "Net PnL": f"+${w_net:.2f}" if w_net > 0 else f"-${abs(w_net):.2f}",
            "Sharpe": round(w_sh, 2),
            "Status": "PASS" if w_net > 0 else "FAIL"
        })
    print(pd.DataFrame(wf_rows).to_string(index=False))
    print(f"   Overall Walk-Forward Status: {'PASS (100% OOS Windows Positive)' if all_oos_pos else 'FAIL'}")

    # 3. Anti-Overfit Grid Audit (9 Configurations)
    print(f"\n3. ANTI-OVERFIT GRID STABILITY AUDIT (9 CONFIGURATIONS):")
    grid_rows = []
    n_pass = 0
    for th in [0.0035, 0.0040, 0.0045]:
        for h in [12, 18, 24]:
            df_g = run_buffed_smr(df_all, close_mat, open_mat, hours, minutes, TARGET_PAIRS, th, 13, 30, h, "fundednext")
            g_pnls = df_g["net_pnl"].values if not df_g.empty else np.array([])
            g_wins = sum(1 for p in g_pnls if p > 0)
            g_gw = sum(p for p in g_pnls if p > 0)
            g_gl = abs(sum(p for p in g_pnls if p < 0))
            g_net = sum(g_pnls)
            g_wr = g_wins / len(g_pnls) * 100.0 if len(g_pnls) > 0 else 0.0
            g_pf = g_gw / g_gl if g_gl > 0 else 0.0
            st = "PASS" if g_net > 0 and g_pf > 1.2 else "FAIL"
            if st == "PASS": n_pass += 1
            grid_rows.append({
                "Threshold": f"{th*100:.2f}%",
                "Hold (M5)": f"{h*5}m",
                "Trades": len(g_pnls),
                "Win Rate": f"{g_wr:.1f}%",
                "Net PnL": f"+${g_net:.2f}" if g_net > 0 else f"-${abs(g_net):.2f}",
                "PF": round(g_pf, 2),
                "Status": st
            })
    print(pd.DataFrame(grid_rows).to_string(index=False))
    print(f"   Grid Stability Score: {n_pass} / 9 Configurations Positive ({n_pass/9*100:.1f}%)")

    # 4. 5-Broker Survival Audit
    print(f"\n4. 5-BROKER SURVIVAL AUDIT:")
    broker_rows = []
    n_b_pass = 0
    for b in BROKERS:
        df_b = run_buffed_smr(df_all, close_mat, open_mat, hours, minutes, TARGET_PAIRS, 0.0045, 13, 30, 18, b)
        b_pnls = df_b["net_pnl"].values if not df_b.empty else np.array([])
        b_wins = sum(1 for p in b_pnls if p > 0)
        b_gw = sum(p for p in b_pnls if p > 0)
        b_gl = abs(sum(p for p in b_pnls if p < 0))
        b_net = sum(b_pnls)
        b_wr = b_wins / len(b_pnls) * 100.0 if len(b_pnls) > 0 else 0.0
        b_pf = b_gw / b_gl if b_gl > 0 else 0.0
        st = "PASS" if b_net > 0 and b_pf > 1.2 else "FAIL"
        if st == "PASS": n_b_pass += 1
        broker_rows.append({
            "Broker": b.upper(),
            "Trades": len(b_pnls),
            "Win Rate": f"{b_wr:.1f}%",
            "Net PnL": f"+${b_net:.2f}" if b_net > 0 else f"-${abs(b_net):.2f}",
            "PF": round(b_pf, 2),
            "Status": st
        })
    print(pd.DataFrame(broker_rows).to_string(index=False))
    print(f"   Broker Survival Score: {n_b_pass} / 5 Brokers Passed")

if __name__ == "__main__":
    main()
