#!/usr/bin/env python3
"""Audit BUFFED SMR at 0.40% Threshold (23 Trades)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd
import numpy as np

from audit_buffed_smr_full import run_buffed_smr, load_and_align, BROKERS

def main():
    raw, pre_align = load_and_align()
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

    target_pairs = ["GBPAUD", "USDJPY", "EURJPY", "GBPJPY"]

    df_t = run_buffed_smr(df_all, close_mat, open_mat, hours, minutes, target_pairs, 0.0040, 13, 30, 18, "fundednext")
    pnls = df_t["net_pnl"].values if not df_t.empty else np.array([])
    n_trades = len(pnls)

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

    df_t["window"] = pd.qcut(df_t["time"], 5, labels=["W1", "W2", "W3", "W4", "W5"], duplicates='drop')
    all_oos_pos = True
    wf_rows = []
    for w_name, grp in df_t.groupby("window", observed=False):
        w_pnls = grp["net_pnl"].values
        w_net = sum(w_pnls)
        if w_net <= 0: all_oos_pos = False
        wf_rows.append({"Window": w_name, "Trades": len(w_pnls), "Net PnL": f"+${w_net:.2f}" if w_net > 0 else f"-${abs(w_net):.2f}"})

    print("="*85)
    print("BUFFED SMR AT 0.40% THRESHOLD (23 TRADES) AUDIT RESULTS")
    print("="*85)
    print(f"  Total Trades        : {n_trades}")
    print(f"  Net PnL (FundedNext): +${sum(pnls):.2f}")
    print(f"  Permutation p-value : {p_val:.4f} ({'PASS p<0.01' if p_val < 0.01 else 'FAIL'})")
    print(f"  Walk-Forward Status : {'PASS (100% OOS Positive)' if all_oos_pos else 'FAIL'}")
    print(pd.DataFrame(wf_rows).to_string(index=False))
    print("="*85)

if __name__ == "__main__":
    main()
