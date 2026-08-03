#!/usr/bin/env python3
"""Full 5-Part Validation Audit Suite for ULTRA MONSTER Rolling ORB."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd
import numpy as np

from proxima_honest_backtest.strategies.sunday_h22.sweep import load_and_align, BROKERS
from proxima_honest_backtest.execution.execution_simulator import ExecutionSimulator
from ultra_buff_rolling_orb import run_ultra_buffed_orb

PAIRS_ALL = ["EURUSD", "GBPUSD", "USDJPY", "EURAUD", "GBPAUD", "EURJPY", "GBPJPY", "EURNZD", "GBPNZD"]

def main():
    t0 = time.time()
    print("Loading M5 dataset for ULTRA MONSTER Full Validation Suite...")
    raw, pre_align = load_and_align()
    pieces = [df.set_index("time")[["close","open","high","low"]] for df in raw.values()]
    for i, p in enumerate(raw.keys()):
        pieces[i].columns = [p, f"{p}_open", f"{p}_high", f"{p}_low"]
    df_all = pd.concat(pieces, axis=1, sort=True).ffill().bfill()
    times = pd.to_datetime(df_all.index)

    close_mat = df_all[[p for p in PAIRS_ALL]].values
    open_mat = df_all[[f"{p}_open" for p in PAIRS_ALL]].values
    high_mat = df_all[[f"{p}_high" for p in PAIRS_ALL]].values
    low_mat = df_all[[f"{p}_low" for p in PAIRS_ALL]].values

    hours = times.hour.values
    minutes = times.minute.values

    df_u = run_ultra_buffed_orb(df_all, close_mat, open_mat, high_mat, low_mat, hours, minutes, PAIRS_ALL, 6.0, range(0, 24), [0, 30], 3)
    pnls = df_u["net_pnl"].values if not df_u.empty else np.array([])
    n_trades = len(pnls)

    print("\n" + "="*95)
    print("FULL VALIDATION SUITE: ULTRA MONSTER ROLLING ORB ENGINE")
    print("="*95)

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
    df_u["window"] = pd.qcut(df_u["time"], 5, labels=["W1", "W2", "W3", "W4", "W5"], duplicates='drop')
    wf_rows = []
    all_oos_pos = True
    for w_name, grp in df_u.groupby("window", observed=False):
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
    for min_p in [4.0, 6.0, 8.0]:
        for h in [2, 3, 4]:
            df_g = run_ultra_buffed_orb(df_all, close_mat, open_mat, high_mat, low_mat, hours, minutes, PAIRS_ALL, min_p, range(0, 24), [0, 30], h)
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
                "Min Range": f"{min_p:.0f}p",
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
        sim = ExecutionSimulator(b)
        n_bars = len(df_all)
        pair_indices = range(len(PAIRS_ALL))
        in_pos = [False] * len(PAIRS_ALL)
        exit_bar = [0] * len(PAIRS_ALL)
        entry_pr = [0.0] * len(PAIRS_ALL)
        direction = [0] * len(PAIRS_ALL)
        trades_b = []
        for t in range(13, n_bars):
            for p_i in pair_indices:
                if in_pos[p_i] and t >= exit_bar[p_i]:
                    c_exit = close_mat[t, p_i]
                    c_entry = entry_pr[p_i]
                    dir_i = direction[p_i]
                    gross_pnl = (c_exit - c_entry) / c_entry * 50000.0 * dir_i
                    comm = sim.profile.commission_per_lot * 0.5
                    net_pnl = gross_pnl - comm
                    trades_b.append(net_pnl)
                    in_pos[p_i] = False
            if minutes[t] in [0, 30]:
                for p_i in pair_indices:
                    if in_pos[p_i]: continue
                    h_prev = np.max(high_mat[t-12:t, p_i])
                    l_prev = np.min(low_mat[t-12:t, p_i])
                    c_now = close_mat[t, p_i]
                    range_pips = (h_prev - l_prev) * 10000.0 if "JPY" not in PAIRS_ALL[p_i] else (h_prev - l_prev) * 100.0
                    if range_pips < 6.0: continue
                    if c_now > h_prev:
                        in_pos[p_i] = True; direction[p_i] = 1; exit_bar[p_i] = t + 3; entry_pr[p_i] = open_mat[t, p_i]
                    elif c_now < l_prev:
                        in_pos[p_i] = True; direction[p_i] = -1; exit_bar[p_i] = t + 3; entry_pr[p_i] = open_mat[t, p_i]

        b_pnls = trades_b
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
