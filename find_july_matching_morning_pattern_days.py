#!/usr/bin/env python3
"""Find Historical Days in July 2026 matching Today's Morning Drawdown Pattern."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd
import numpy as np

from proxima_honest_backtest.strategies.sunday_h22.sweep import load_and_align
from ultra_buff_rolling_orb import run_ultra_buffed_orb

PAIRS_ALL = ["EURUSD", "GBPUSD", "USDJPY", "EURAUD", "GBPAUD", "EURJPY", "GBPJPY", "EURNZD", "GBPNZD"]

def main():
    print("Searching July 2026 dataset for days matching Today's Morning Pattern...")
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

    matching_days = []

    for d in july_dates:
        sub = df_u[df_u["date"] == d]
        
        morning_trades = sub[sub["h"] <= 6]
        afternoon_trades = sub[sub["h"] > 6]

        m_pnl = morning_trades["pnl_1lot"].sum()
        a_pnl = afternoon_trades["pnl_1lot"].sum()
        total_day_pnl = m_pnl + a_pnl

        # Search for days with negative morning PnL between -$10 and -$300
        if -300.0 <= m_pnl < 0.0:
            matching_days.append({
                "Date": d.strftime("%Y-%m-%d (%a)"),
                "Morning Trades": len(morning_trades),
                "Morning PnL (01:00-11:30 AM IST)": f"-${abs(m_pnl):,.2f}",
                "Afternoon/Evening PnL (12:30 PM+ IST)": f"+${a_pnl:,.2f}" if a_pnl >= 0 else f"-${abs(a_pnl):,.2f}",
                "Final Day-End Net PnL": f"+${total_day_pnl:,.2f}" if total_day_pnl >= 0 else f"-${abs(total_day_pnl):,.2f}",
                "Day Outcome": "🟢 RECOVERED TO WIN" if total_day_pnl > 0 else "🔴 CLOSED LOSS"
            })

    df_match = pd.DataFrame(matching_days)
    print("="*115)
    print("MATCHING JULY 2026 DAYS (DAYS WITH SAME MORNING DRAWDOWN PATTERN AS TODAY)")
    print("="*115)
    print(df_match.to_string(index=False))
    print("="*115)

if __name__ == "__main__":
    main()
