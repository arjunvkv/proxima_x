#!/usr/bin/env python3
"""Audit Ultra_Monster_MT5 Hour-by-Hour Performance Pattern across 24 Hours."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd
import numpy as np

from proxima_honest_backtest.strategies.sunday_h22.sweep import load_and_align
from ultra_buff_rolling_orb import run_ultra_buffed_orb

PAIRS_ALL = ["EURUSD", "GBPUSD", "USDJPY", "EURAUD", "GBPAUD", "EURJPY", "GBPJPY", "EURNZD", "GBPNZD"]

def main():
    print("Analyzing Hour-by-Hour Performance Pattern for Ultra_Monster_MT5...")
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
    
    df_u["dt"] = pd.to_datetime(df_u["time"])
    df_u["hour_utc"] = df_u["dt"].dt.hour
    df_u["pnl_1lot"] = df_u["net_pnl"] / 0.15 * 1.00

    hourly_stats = []

    for h_utc in range(24):
        sub = df_u[df_u["hour_utc"] == h_utc]
        n_t = len(sub)
        if n_t == 0:
            continue
        pnls = sub["pnl_1lot"].values
        wins = sum(1 for p in pnls if p > 0)
        wr = wins / n_t * 100.0
        tot_pnl = sum(pnls)
        avg_pnl = np.mean(pnls)

        # Convert UTC to IST (+05:30)
        h_ist = (h_utc + 5) % 24
        ist_str = f"{h_ist:02d}:30 IST"

        hourly_stats.append({
            "UTC Hour": f"{h_utc:02d}:00 UTC",
            "IST Hour": ist_str,
            "Trades": n_t,
            "Wins": wins,
            "Losses": n_t - wins,
            "Win Rate": f"{wr:.1f}%",
            "Total PnL (1.00L)": f"+${tot_pnl:,.2f}" if tot_pnl >= 0 else f"-${abs(tot_pnl):,.2f}",
            "Avg PnL / Trade": f"+${avg_pnl:.2f}" if avg_pnl >= 0 else f"-${abs(avg_pnl):.2f}"
        })

    df_h = pd.DataFrame(hourly_stats)
    print("="*105)
    print("24-HOUR HOURLY PERFORMANCE PATTERN REPORT: ULTRA_MONSTER_MT5 (10,068 TRADES)")
    print("="*105)
    print(df_h.to_string(index=False))
    print("="*105)

if __name__ == "__main__":
    main()
