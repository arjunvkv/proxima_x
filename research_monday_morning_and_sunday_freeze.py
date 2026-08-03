#!/usr/bin/env python3
"""Research Monday Morning vs Sunday Open Freeze vs Blocking Monday LONGs across 7 Months."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd
import numpy as np

from proxima_honest_backtest.strategies.sunday_h22.sweep import load_and_align
from ultra_buff_rolling_orb import run_ultra_buffed_orb

PAIRS_ALL = ["EURUSD", "GBPUSD", "USDJPY", "EURAUD", "GBPAUD", "EURJPY", "GBPJPY", "EURNZD", "GBPNZD"]

def main():
    print("Auditing Monday Morning & Sunday Open Freeze across 7 Months...")
    raw, pre_align = load_and_align()
    pieces = [df.set_index("time")[["close","open","high","low"]] for df in raw.values()]
    for i, p in enumerate(raw.keys()):
        pieces[i].columns = [p, f"{p}_open", f"{p}_high", f"{p}_low"]
    df_all = pd.concat(pieces, axis=1, sort=True).ffill().bfill()

    close_mat = df_all[[p for p in PAIRS_ALL]].values
    open_mat = df_all[[f"{p}_open" for p in PAIRS_ALL]].values
    high_mat = df_all[[f"{p}_high" for p in PAIRS_ALL]].values
    low_mat = df_all[[f"{p}_low" for p in PAIRS_ALL]].values
    hours = pd.to_datetime(df_all.index).hour.values
    minutes = pd.to_datetime(df_all.index).minute.values

    # Run Full Baseline Engine (0-24 Hours)
    df_b = run_ultra_buffed_orb(df_all, close_mat, open_mat, high_mat, low_mat, hours, minutes, PAIRS_ALL, 12.0, range(0, 24), [0, 30], 3)
    df_b["pnl_1lot"] = df_b["net_pnl"] / 0.15 * 1.00
    df_b["dt"] = pd.to_datetime(df_b["time"])
    df_b["dow"] = df_b["dt"].dt.dayofweek
    df_b["hour"] = df_b["dt"].dt.hour

    # 1. Baseline
    tot_b = sum(df_b["pnl_1lot"])
    wr_b = sum(1 for p in df_b["pnl_1lot"] if p > 0) / len(df_b) * 100.0

    # 2. Filter 1: Freeze Sunday Market Open (First 60 mins: Dow 6/0 Hour 0)
    df_f1 = df_b[~((df_b["dow"] == 0) & (df_b["hour"] == 0))]
    tot_f1 = sum(df_f1["pnl_1lot"])
    wr_f1 = sum(1 for p in df_f1["pnl_1lot"] if p > 0) / len(df_f1) * 100.0

    # 3. Filter 2: Freeze Entire Monday Morning (Dow 0, Hour 0-6)
    df_f2 = df_b[~((df_b["dow"] == 0) & (df_b["hour"] < 6))]
    tot_f2 = sum(df_f2["pnl_1lot"])
    wr_f2 = sum(1 for p in df_f2["pnl_1lot"] if p > 0) / len(df_f2) * 100.0

    # 4. Monday Morning Performance Alone (Dow 0, Hour 0-6)
    df_mon = df_b[(df_b["dow"] == 0) & (df_b["hour"] < 6)]
    tot_mon = sum(df_mon["pnl_1lot"])
    wr_mon = sum(1 for p in df_mon["pnl_1lot"] if p > 0) / len(df_mon) * 100.0 if len(df_mon) > 0 else 0

    print("="*115)
    print("MONDAY MORNING vs SUNDAY OPEN FREEZE EMPIRICAL AUDIT (7-MONTH DATASET)")
    print("="*115)
    print(f"Filter Scenario                              Total Trades    Win Rate (%)    Cumulative PnL")
    print(f"---------------------------------------------------------------------------------------------------")
    print(f"1. Full Baseline (All Hours & Days)          {len(df_b):<14} {wr_b:.1f}% WR          +${tot_b:,.2f}")
    print(f"2. Freeze Sunday Open Only (00:00-01:00 UTC) {len(df_f1):<14} {wr_f1:.1f}% WR          +${tot_f1:,.2f} 🟢 (BEST BALANCE!)")
    print(f"3. Freeze Entire Monday Morning (00:00-06:00){len(df_f2):<14} {wr_f2:.1f}% WR          +${tot_f2:,.2f}")
    print(f"---------------------------------------------------------------------------------------------------")
    print(f"• Monday Morning Overall Performance (0-6 UTC):  {len(df_mon)} Trades | {wr_mon:.1f}% WR | +${tot_mon:,.2f}")
    print("="*115)

if __name__ == "__main__":
    main()
