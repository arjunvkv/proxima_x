#!/usr/bin/env python3
"""Print Early Day vs NY Session PnL for ALL Days in July 2026."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd
import numpy as np

from proxima_honest_backtest.strategies.sunday_h22.sweep import load_and_align
from ultra_buff_rolling_orb import run_ultra_buffed_orb

PAIRS_ALL = ["EURUSD", "GBPUSD", "USDJPY", "EURAUD", "GBPAUD", "EURJPY", "GBPJPY", "EURNZD", "GBPNZD"]

def main():
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
    df_u["date"] = df_u["dt"].dt.date
    df_u["h_ist"] = (df_u["dt"].dt.hour + 5) % 24
    df_u["pnl_1lot"] = df_u["net_pnl"] / 0.15 * 1.00

    july_dates = sorted(list(set(df_u[df_u["date"] >= pd.to_datetime("2026-07-01").date()]["date"])))

    july_matrix = []

    for d in july_dates:
        sub = df_u[df_u["date"] == d]
        
        early_trades = sub[sub["h_ist"] <= 16]
        ny_trades = sub[sub["h_ist"] > 16]

        e_pnl = early_trades["pnl_1lot"].sum()
        ny_pnl = ny_trades["pnl_1lot"].sum()
        total_pnl = e_pnl + ny_pnl

        july_matrix.append({
            "Date": d.strftime("%Y-%m-%d (%a)"),
            "Early Day PnL (07:00 AM - 04:30 PM IST)": f"+${e_pnl:,.2f}" if e_pnl >= 0 else f"-${abs(e_pnl):,.2f}",
            "NY Session PnL (06:00 PM - 11:30 PM IST)": f"+${ny_pnl:,.2f}" if ny_pnl >= 0 else f"-${abs(ny_pnl):,.2f}",
            "Final Day-End Net PnL": f"+${total_pnl:,.2f}" if total_pnl >= 0 else f"-${abs(total_pnl):,.2f}",
            "NY Surge Share": f"{(ny_pnl/total_pnl)*100.0:.1f}%" if total_pnl > 0 and ny_pnl > 0 else "N/A"
        })

    df_m = pd.DataFrame(july_matrix)
    print("="*115)
    print("ALL 20 TRADING DAYS IN JULY 2026: EARLY DAY vs NY SESSION ACCELERATION MATRIX")
    print("="*115)
    print(df_m.to_string(index=False))
    print("="*115)

if __name__ == "__main__":
    main()
