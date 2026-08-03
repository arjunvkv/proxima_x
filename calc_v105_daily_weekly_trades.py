#!/usr/bin/env python3
"""Calculate Per-Day and Per-Week Trade Frequency for Ultra_Monster_MT5 v1.05."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd
import numpy as np

from proxima_honest_backtest.strategies.sunday_h22.sweep import load_and_align
from ultra_buff_rolling_orb import run_ultra_buffed_orb

PAIRS_ALL = ["EURUSD", "GBPUSD", "USDJPY", "EURAUD", "GBPAUD", "EURJPY", "GBPJPY", "EURNZD", "GBPNZD"]

def main():
    print("Calculating Per-Day and Per-Week Trade Frequency for Ultra_Monster_MT5 v1.05...")
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

    df_u = run_ultra_buffed_orb(df_all, close_mat, open_mat, high_mat, low_mat, hours, minutes, PAIRS_ALL, 12.0, range(0, 24), [0, 30], 3)
    
    df_u["dt"] = pd.to_datetime(df_u["time"])
    df_u["date"] = df_u["dt"].dt.date
    df_u["week"] = df_u["dt"].dt.isocalendar().week

    tot_trades = len(df_u)
    tot_days = df_u["date"].nunique()
    tot_weeks = df_u["week"].nunique()

    trades_per_day = tot_trades / tot_days
    trades_per_week = tot_trades / (tot_days / 5.0)  # 5 trading days per week

    print("="*95)
    print("TRADE FREQUENCY REPORT: ULTRA_MONSTER_MT5 v1.05 (12.0p VOLATILITY GATE)")
    print("="*95)
    print(f"  Total Audited Trades            ──► {tot_trades:,} Trades")
    print(f"  Total Audited Trading Days      ──► {tot_days} Trading Days")
    print(f"  Average Trades Per Day          ──► {trades_per_day:.1f} Trades / Day (Portfolio Total)")
    print(f"  Average Trades Per Week         ──► {trades_per_week:.1f} Trades / Week (Portfolio Total)")
    print(f"  Average Trades Per Pair / Day   ──► {trades_per_day / 9.0:.1f} Trades / Pair / Day")
    print("="*95)

    per_pair_freq = []
    for pair in PAIRS_ALL:
        sub = df_u[df_u["pair"] == pair]
        n_t = len(sub)
        p_day = n_t / tot_days
        p_week = p_day * 5.0
        per_pair_freq.append({
            "Symbol": pair,
            "Total Trades": n_t,
            "Trades / Day": f"{p_day:.1f} / day",
            "Trades / Week": f"{p_week:.1f} / week"
        })

    df_pf = pd.DataFrame(per_pair_freq)
    print("PER-PAIR TRADE FREQUENCY MATRIX:")
    print(df_pf.to_string(index=False))
    print("="*95)

if __name__ == "__main__":
    main()
