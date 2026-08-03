#!/usr/bin/env python3
"""Print Morning vs Afternoon/Evening PnL Breakdown for ALL Days in July 2026."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd
import numpy as np

from proxima_honest_backtest.strategies.sunday_h22.sweep import load_and_align
from ultra_buff_rolling_orb import run_ultra_buffed_orb

PAIRS_ALL = ["EURUSD", "GBPUSD", "USDJPY", "EURAUD", "GBPAUD", "EURJPY", "GBPJPY", "EURNZD", "GBPNZD"]

def main():
    print("Extracting Morning vs Afternoon/Evening Breakdown for ALL July 2026 Days...")
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
    df_u["h"] = df_u["dt"].dt.hour
    df_u["pnl_1lot"] = df_u["net_pnl"] / 0.15 * 1.00

    july_dates = sorted(list(set(df_u[df_u["date"] >= pd.to_datetime("2026-07-01").date()]["date"])))

    all_july_breakdown = []

    for d in july_dates:
        sub = df_u[df_u["date"] == d]
        
        m_trades = sub[sub["h"] <= 6]
        a_trades = sub[sub["h"] > 6]

        m_pnl = m_trades["pnl_1lot"].sum()
        a_pnl = a_trades["pnl_1lot"].sum()
        total_pnl = m_pnl + a_pnl

        all_july_breakdown.append({
            "Date": d.strftime("%Y-%m-%d (%a)"),
            "Morning PnL (01:00-11:30 AM IST)": f"+${m_pnl:,.2f}" if m_pnl >= 0 else f"-${abs(m_pnl):,.2f}",
            "Afternoon PnL (12:30 PM+ IST)": f"+${a_pnl:,.2f}" if a_pnl >= 0 else f"-${abs(a_pnl):,.2f}",
            "Final Day-End PnL": f"+${total_pnl:,.2f}" if total_pnl >= 0 else f"-${abs(total_pnl):,.2f}",
            "Afternoon Gain Multiple": f"{a_pnl/max(1, abs(m_pnl)):.1f}x" if m_pnl < 0 else "Strong Surge"
        })

    df_full = pd.DataFrame(all_july_breakdown)
    print("="*115)
    print("JULY 2026 FULL MONTH: MORNING (01:00-11:30 AM IST) vs AFTERNOON/EVENING (12:30 PM+ IST) COMPARISON")
    print("="*115)
    print(df_full.to_string(index=False))
    print("="*115)

if __name__ == "__main__":
    main()
