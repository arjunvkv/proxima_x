#!/usr/bin/env python3
"""Buff CPMC_Z_MT5 Win Rate while Preserving High Event Frequency."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd
import numpy as np

from proxima_honest_backtest.strategies.sunday_h22.sweep import load_and_align

PAIRS_CPMC = ["GBPAUD", "GBPNZD"]

def run_cpmc_buff_sim(df_all, close_mat, high_mat, low_mat, pairs, z_thresh=5.0, min_range_pips=8.0, hold_bars=18):
    trades = []
    n_bars = len(df_all)

    for p_idx, pair in enumerate(pairs):
        closes = close_mat[:, p_idx]
        highs = high_mat[:, p_idx]
        lows = low_mat[:, p_idx]
        pip_mult = 100.0 if "JPY" in pair else 10000.0
        
        # 3-bar returns (15m)
        ret3 = (closes[3:] - closes[:-3]) * pip_mult
        
        for t in range(200, len(ret3) - hold_bars):
            window = ret3[t-200:t]
            mean_w = np.mean(window)
            std_w = np.std(window)
            if std_w <= 0:
                continue
            
            z = (ret3[t] - mean_w) / std_w

            # Volatility range check
            bar_range = (np.max(highs[t:t+3]) - np.min(lows[t:t+3])) * pip_mult
            if bar_range < min_range_pips:
                continue

            # FADE Mean Reversion (z >= z_thresh -> SELL, z <= -z_thresh -> BUY)
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
    print("Buffing CPMC_Z_MT5 Win Rate across 7-Month Dataset...")
    raw, pre_align = load_and_align()
    pieces = [df.set_index("time")[["close","open","high","low"]] for df in raw.values()]
    for i, p in enumerate(raw.keys()):
        pieces[i].columns = [p, f"{p}_open", f"{p}_high", f"{p}_low"]
    df_all = pd.concat(pieces, axis=1, sort=True).ffill().bfill()

    close_mat = df_all[[p for p in PAIRS_CPMC]].values
    high_mat = df_all[[f"{p}_high" for p in PAIRS_CPMC]].values
    low_mat = df_all[[f"{p}_low" for p in PAIRS_CPMC]].values

    tot_days = 148

    configs = [
        {"name": "v1.00 Baseline (z=4.5, range=0p)", "z": 4.5, "range": 0.0, "hold": 18},
        {"name": "Option A: Volatility Gate (z=4.5, range=8.0p)", "z": 4.5, "range": 8.0, "hold": 18},
        {"name": "Option B: Moderate Z-Buff (z=5.0, range=8.0p)", "z": 5.0, "range": 8.0, "hold": 18},
        {"name": "Option C: High-Conviction Z-Buff (z=5.5, range=8.0p)", "z": 5.5, "range": 8.0, "hold": 18},
        {"name": "Option D: Extreme 6-Sigma (z=6.0, range=8.0p)", "z": 6.0, "range": 8.0, "hold": 18},
    ]

    results = []

    for cfg in configs:
        pnls = run_cpmc_buff_sim(df_all, close_mat, high_mat, low_mat, PAIRS_CPMC, cfg["z"], cfg["range"], cfg["hold"])
        n_t = len(pnls)
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        wr = len(wins) / n_t * 100.0 if n_t > 0 else 0
        tot_pnl = sum(pnls)
        pf = sum(wins) / max(1, abs(sum(losses)))
        avg_w = np.mean(wins) if wins else 0
        avg_l = np.mean(losses) if losses else 0
        t_per_day = n_t / tot_days

        results.append({
            "Configuration": cfg["name"],
            "Total Trades": n_t,
            "Trades / Day": f"{t_per_day:.2f} / day",
            "Wins 🟢": len(wins),
            "Losses 🔴": len(losses),
            "Net Win Rate (%)": f"{wr:.1f}%",
            "Profit Factor": f"{pf:.2f}",
            "Average Win (1.20L)": f"+${avg_w:.2f}",
            "Cumulative Net Profit": f"+${tot_pnl:,.2f}"
        })

    df_res = pd.DataFrame(results)
    print("="*115)
    print("CPMC_Z_MT5 WIN RATE ENHANCEMENT MATRIX (1.20 LOT SIZING / 7-MONTH DATASET)")
    print("="*115)
    print(df_res.to_string(index=False))
    print("="*115)

if __name__ == "__main__":
    main()
