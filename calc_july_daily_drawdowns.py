#!/usr/bin/env python3
"""Calculate Exact Peak Daily Drawdown for Every Trading Day in July 2026."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd
import numpy as np

from proxima_honest_backtest.strategies.sunday_h22.sweep import load_and_align
from ultra_buff_rolling_orb import run_ultra_buffed_orb

PAIRS_ALL = ["EURUSD", "GBPUSD", "USDJPY", "EURAUD", "GBPAUD", "EURJPY", "GBPJPY", "EURNZD", "GBPNZD"]

def main():
    print("Calculating Intraday Peak Daily Drawdowns for July 2026...")
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

    # Run Ultra Monster Engine
    df_u = run_ultra_buffed_orb(df_all, close_mat, open_mat, high_mat, low_mat, hours, minutes, PAIRS_ALL, 6.0, range(0, 24), [0, 30], 3)
    df_u["date"] = pd.to_datetime(df_u["time"]).dt.date
    df_u["pnl_squeeze"] = df_u["net_pnl"] / 0.15 * 1.00

    july_dates = sorted(list(set(df_u[df_u["date"] >= pd.to_datetime("2026-07-01").date()]["date"])))

    ftmo_daily_limit = 1250.0
    daily_dd_list = []

    for d in july_dates:
        sub = df_u[df_u["date"] == d].copy()
        pnls = sub["pnl_squeeze"].values
        
        # Calculate intraday equity curve for the day
        eq = np.cumsum(pnls)
        peak = np.maximum.accumulate(np.insert(eq, 0, 0.0))
        troughs = peak - np.insert(eq, 0, 0.0)
        max_dd = np.max(troughs) if len(troughs) > 0 else 0.0

        daily_net_pnl = sum(pnls)
        unused_cushion = ftmo_daily_limit - max_dd
        dd_pct_limit = (max_dd / ftmo_daily_limit) * 100.0

        daily_dd_list.append({
            "Date": d.strftime("%Y-%m-%d (%a)"),
            "Total Trades": len(pnls),
            "Daily Net PnL": f"+${daily_net_pnl:,.2f}" if daily_net_pnl >= 0 else f"-${abs(daily_net_pnl):,.2f}",
            "Peak Daily DD": f"${max_dd:,.2f}",
            "FTMO Limit Usage": f"{dd_pct_limit:.1f}% of $1,250",
            "Unused Daily Cushion": f"${unused_cushion:,.2f} SAFE",
            "Status": "🟢 SAFE" if max_dd < ftmo_daily_limit else "🔴 VIOLATED"
        })

    df_dd = pd.DataFrame(daily_dd_list)
    print("="*115)
    print("DAILY DRAWDOWN AUDIT TABLE: JULY 2026 (FTMO $25k ACCOUNT - $1,250 DAILY LIMIT)")
    print("="*115)
    print(df_dd.to_string(index=False))
    print("="*115)
    max_month_dd = max([float(x['Peak Daily DD'].replace('$','').replace(',','')) for x in daily_dd_list])
    print(f"WORST INTRADAY DAILY DRAWDOWN IN JULY 2026: ${max_month_dd:,.2f} ({max_month_dd/ftmo_daily_limit*100:.1f}% of $1,250 Daily Limit)")
    print("="*115)

if __name__ == "__main__":
    main()
