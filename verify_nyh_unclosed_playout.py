#!/usr/bin/env python3
"""Audit exact price playout for NYH trades if left unclosed until 45m and 60m expiry."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd
import numpy as np

from proxima_honest_backtest.strategies.sunday_h22.sweep import load_and_align

def main():
    print("Auditing Direct MT5 Price Playout for NYH Trades at 45m and 60m Expiry...")
    raw, pre_align = load_and_align()
    pieces = [df.set_index("time")[["close","open","high","low"]] for df in raw.values()]
    for i, p in enumerate(raw.keys()):
        pieces[i].columns = [p, f"{p}_open", f"{p}_high", f"{p}_low"]
    df_all = pd.concat(pieces, axis=1, sort=True).ffill().bfill()
    times = pd.to_datetime(df_all.index)

    # 1. EURJPY BUY entered at 18:00 UTC at 182.820
    # 2. GBPJPY BUY entered at 18:00 UTC at 213.737

    # Find bar indices for 18:00, 18:18 (manual), 18:45 (45m), 19:00 (60m)
    idx_1800 = [i for i, t in enumerate(times) if t.hour == 18 and t.minute == 0][-1]
    idx_1845 = [i for i, t in enumerate(times) if t.hour == 18 and t.minute == 45][-1]
    idx_1900 = [i for i, t in enumerate(times) if t.hour == 19 and t.minute == 0][-1]

    eurjpy_entry = 182.820
    gbpjpy_entry = 213.737

    eurjpy_45m = df_all["EURJPY"].iloc[idx_1845]
    gbpjpy_45m = df_all["GBPJPY"].iloc[idx_1845]

    eurjpy_60m = df_all["EURJPY"].iloc[idx_1900]
    gbpjpy_60m = df_all["GBPJPY"].iloc[idx_1900]

    # Calculate exact PnL
    pnl_eurjpy_manual = (182.983 - eurjpy_entry) * 100.0 * 1.6
    pnl_gbpjpy_manual = (213.882 - gbpjpy_entry) * 100.0 * 1.6

    pnl_eurjpy_45m = (eurjpy_45m - eurjpy_entry) * 100.0 * 1.6
    pnl_gbpjpy_45m = (gbpjpy_45m - gbpjpy_entry) * 100.0 * 1.6

    pnl_eurjpy_60m = (eurjpy_60m - eurjpy_entry) * 100.0 * 1.6
    pnl_gbpjpy_60m = (gbpjpy_60m - gbpjpy_entry) * 100.0 * 1.6

    print("="*105)
    print("DIRECT MT5 BACKTEST PLAYOUT AUDIT: MANUAL CLOSE vs 45M BUFF vs 60M BASELINE")
    print("="*105)
    print(f"Scenario                             EURJPY PnL (1.6L)   GBPJPY PnL (1.6L)   Combined Net Profit")
    print(f"---------------------------------------------------------------------------------------------------")
    print(f"1. Manual Close (18m / 11:48 PM IST)  +${pnl_eurjpy_manual:.2f}         +${pnl_gbpjpy_manual:.2f}         +${pnl_eurjpy_manual + pnl_gbpjpy_manual:,.2f} 🟢")
    print(f"2. Buffed 45m Exit (12:15 AM IST)     +${pnl_eurjpy_45m:.2f}         +${pnl_gbpjpy_45m:.2f}         +${pnl_eurjpy_45m + pnl_gbpjpy_45m:,.2f} 🟢")
    print(f"3. Baseline 60m Exit (12:30 AM IST)   +${pnl_eurjpy_60m:.2f}         +${pnl_gbpjpy_60m:.2f}         +${pnl_eurjpy_60m + pnl_gbpjpy_60m:,.2f} 🟢")
    print("="*105)

if __name__ == "__main__":
    main()
