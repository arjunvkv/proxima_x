#!/usr/bin/env python3
"""Sweep Hold Durations for CPMC_Z Mean Reversion Engine across 7-Month Audited Dataset."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd
import numpy as np

from proxima_honest_backtest.strategies.sunday_h22.sweep import load_and_align

PAIRS_CPMC = ["GBPAUD", "GBPNZD"]

def run_cpmc_fade_sim(df_all, close_mat, pairs, z_thresh=4.5, hold_bars=9):
    trades = []
    n_bars = len(df_all)

    for p_idx, pair in enumerate(pairs):
        closes = close_mat[:, p_idx]
        pip_mult = 100.0 if "JPY" in pair else 10000.0
        
        # 3-bar returns (15m)
        ret3 = (closes[3:] - closes[:-3]) * pip_mult
        
        # Rolling 200-bar std and mean
        for t in range(200, len(ret3) - hold_bars):
            window = ret3[t-200:t]
            mean_w = np.mean(window)
            std_w = np.std(window)
            if std_w <= 0:
                continue
            
            z = (ret3[t] - mean_w) / std_w

            # Mean Reversion (FADE): z >= 4.5 -> SELL, z <= -4.5 -> BUY
            if z >= z_thresh:
                entry_p = closes[t+3]
                exit_p = closes[t+3+hold_bars]
                pnl_pip = (entry_p - exit_p) * pip_mult
                pnl_usd = pnl_pip * 10.0 * 1.20
                trades.append(pnl_usd)
            elif z <= -z_thresh:
                entry_p = closes[t+3]
                exit_p = closes[t+3+hold_bars]
                pnl_pip = (exit_p - entry_p) * pip_mult
                pnl_usd = pnl_pip * 10.0 * 1.20
                trades.append(pnl_usd)

    return trades

def main():
    print("Sweeping All Hold Durations for CPMC_Z (Mean Reversion Mode)...")
    raw, pre_align = load_and_align()
    pieces = [df.set_index("time")[["close","open","high","low"]] for df in raw.values()]
    for i, p in enumerate(raw.keys()):
        pieces[i].columns = [p, f"{p}_open", f"{p}_high", f"{p}_low"]
    df_all = pd.concat(pieces, axis=1, sort=True).ffill().bfill()

    close_mat = df_all[[p for p in PAIRS_CPMC]].values

    holds = [6, 9, 12, 18, 24]  # 30m, 45m, 60m, 90m, 120m
    results = []

    for hb in holds:
        pnls = run_cpmc_fade_sim(df_all, close_mat, PAIRS_CPMC, 4.5, hb)
        n_t = len(pnls)
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        wr = len(wins) / n_t * 100.0 if n_t > 0 else 0
        tot_pnl = sum(pnls)
        pf = sum(wins) / max(1, abs(sum(losses)))
        avg_w = np.mean(wins) if wins else 0
        avg_l = np.mean(losses) if losses else 0

        results.append({
            "Hold Bars": hb,
            "Hold Duration": f"{hb * 5} Mins",
            "Total Trades": n_t,
            "Wins 🟢": len(wins),
            "Losses 🔴": len(losses),
            "Net Win Rate (%)": f"{wr:.1f}%",
            "Profit Factor": f"{pf:.2f}",
            "Average Win (1.20L)": f"+${avg_w:.2f}",
            "Average Loss (1.20L)": f"-${abs(avg_l):.2f}",
            "Cumulative Net Profit": f"+${tot_pnl:,.2f}"
        })

    df_r = pd.DataFrame(results)
    print("="*115)
    print("CPMC_Z (MEAN REVERSION FADE MODE) HOLD DURATION OPTIMIZATION MATRIX")
    print("="*115)
    print(df_r.to_string(index=False))
    print("="*115)

if __name__ == "__main__":
    main()
