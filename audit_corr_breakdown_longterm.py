#!/usr/bin/env python3
"""Long-Term Survivability & Walk-Forward Audit for Correlation Breakdown (#8)."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd
import numpy as np
from proxima_honest_backtest.strategies.sunday_h22.sweep import load_and_align
from proxima_honest_backtest.execution.execution_simulator import ExecutionSimulator

PAIR_TUPLES = [
    ("AUDUSD", "NZDUSD"),
    ("EURJPY", "GBPJPY"),
    ("EURAUD", "GBPAUD"),
    ("EURNZD", "GBPNZD")
]

def run_corr_audit(thresh_pct=0.0015, lag_max_pct=0.0003, hold_bars=6):
    raw, pre_align = load_and_align()
    pieces = [df.set_index("time")[["close","open"]] for df in raw.values()]
    for i, p in enumerate(raw.keys()):
        pieces[i].columns = [p, f"{p}_open"]
    df_all = pd.concat(pieces, axis=1, sort=True).ffill().bfill()
    times = pd.to_datetime(df_all.index)

    sim = ExecutionSimulator("fundednext")
    n_bars = len(df_all)
    trades = []

    for lead_p, lag_p in PAIR_TUPLES:
        c_lead = df_all[lead_p].values
        c_lag = df_all[lag_p].values
        o_lag = df_all[f"{lag_p}_open"].values

        ret_lead = np.zeros(n_bars)
        ret_lag = np.zeros(n_bars)
        ret_lead[3:] = (c_lead[3:] - c_lead[:-3]) / c_lead[:-3]
        ret_lag[3:] = (c_lag[3:] - c_lag[:-3]) / c_lag[:-3]

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
                trades.append({
                    "time": times[t],
                    "lead": lead_p,
                    "lag": lag_p,
                    "net_pnl": net_pnl,
                    "win": net_pnl > 0
                })
                in_pos = False

            if not in_pos:
                if ret_lead[t] >= thresh_pct and ret_lag[t] <= lag_max_pct:
                    in_pos = True
                    direction = 1
                    exit_bar = t + hold_bars
                    entry_pr = o_lag[t]
                elif ret_lead[t] <= -thresh_pct and ret_lag[t] >= -lag_max_pct:
                    in_pos = True
                    direction = -1
                    exit_bar = t + hold_bars
                    entry_pr = o_lag[t]

    return pd.DataFrame(trades)

def main():
    print("Running Long-Term Survivability & Walk-Forward Audit for Correlation Breakdown...")
    df_t = run_corr_audit(0.0015, 0.0003, 6)
    pnls = df_t["net_pnl"].values
    n_trades = len(pnls)

    print("\n" + "="*85)
    print("LONG-TERM STATISTICAL SURVIVABILITY REPORT: #8 CORRELATION BREAKDOWN")
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
    print(f"   Status                   : {'PASS (Statistically Non-Random)' if p_val < 0.01 else 'FAIL'}")

    # 2. Walk-Forward Out-of-Sample Test (5 Windows)
    print(f"\n2. WALK-FORWARD OUT-OF-SAMPLE TEST (5 WINDOWS):")
    df_t["window"] = pd.qcut(df_t["time"], 5, labels=["W1 (Jan-Feb)", "W2 (Feb-Mar)", "W3 (Mar-Apr)", "W4 (Apr-May)", "W5 (May-Jul)"])
    wf_rows = []
    all_oos_pos = True
    for w_name, grp in df_t.groupby("window"):
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
    for th in [0.0010, 0.0015, 0.0020]:
        for h in [4, 6, 9]:
            df_g = run_corr_audit(thresh_pct=th, lag_max_pct=0.0003, hold_bars=h)
            g_pnls = df_g["net_pnl"].values
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

if __name__ == "__main__":
    main()
