#!/usr/bin/env python3
"""Run MT5 execution test specifically for Ultra_Monster_MT5 on July 30, 2026."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd
import numpy as np

from proxima_honest_backtest.strategies.sunday_h22.sweep import load_and_align
from ultra_buff_rolling_orb import run_ultra_buffed_orb

PAIRS_ALL = ["EURUSD", "GBPUSD", "USDJPY", "EURAUD", "GBPAUD", "EURJPY", "GBPJPY", "EURNZD", "GBPNZD"]

def main():
    print("Loading M5 dataset for Ultra_Monster_MT5 July 30 Specific Audit...")
    raw, pre_align = load_and_align()
    pieces = [df.set_index("time")[["close","open","high","low"]] for df in raw.values()]
    for i, p in enumerate(raw.keys()):
        pieces[i].columns = [p, f"{p}_open", f"{p}_high", f"{p}_low"]
    df_all = pd.concat(pieces, axis=1, sort=True).ffill().bfill()
    times = pd.to_datetime(df_all.index)

    # Filter for July 30, 2026
    mask_jul30 = (times >= "2026-07-30 00:00:00") & (times <= "2026-07-30 23:59:59")
    df_jul30 = df_all.loc[mask_jul30]
    times_j = pd.to_datetime(df_jul30.index)

    close_mat = df_jul30[[p for p in PAIRS_ALL]].values
    open_mat = df_jul30[[f"{p}_open" for p in PAIRS_ALL]].values
    high_mat = df_jul30[[f"{p}_high" for p in PAIRS_ALL]].values
    low_mat = df_jul30[[f"{p}_low" for p in PAIRS_ALL]].values

    hours = times_j.hour.values
    minutes = times_j.minute.values

    # Audit different Min Range thresholds for July 30
    print("="*95)
    print("SPECIFIC MT5 AUDIT: ULTRA_MONSTER_MT5 ON JULY 30, 2026")
    print("="*95)

    for min_r in [3.0, 4.0, 5.0, 6.0]:
        df_u = run_ultra_buffed_orb(df_jul30, close_mat, open_mat, high_mat, low_mat, hours, minutes, PAIRS_ALL, min_r, range(0, 24), [0, 30], 3)
        if not df_u.empty:
            pnls = df_u["net_pnl"].values
            wins = sum(1 for p in pnls if p > 0)
            n_t = len(pnls)
            wr = wins / n_t * 100.0
            tot_pnl = sum(pnls) / 0.15 * 1.00
            print(f"  Min Range >= {min_r:.1f}p ──► {n_t:2d} Trades | {wins:2d} Wins ({wr:.1f}% WR) | Net PnL: +${tot_pnl:,.2f}")
        else:
            print(f"  Min Range >= {min_r:.1f}p ──►  0 Trades")

    print("="*95)

if __name__ == "__main__":
    main()
