#!/usr/bin/env python3
"""Fetch Exact MT5 Tick Data and Bar Prices for Today's 2 Ultra_Monster Trades."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd
import numpy as np

from proxima_honest_backtest.strategies.sunday_h22.sweep import load_and_align

def main():
    print("Auditing Exact MT5 Historical Prices for Today's 2 Ultra_Monster Trades...")
    raw, pre_align = load_and_align()
    pieces = [df.set_index("time")[["close","open","high","low"]] for df in raw.values()]
    for i, p in enumerate(raw.keys()):
        pieces[i].columns = [p, f"{p}_open", f"{p}_high", f"{p}_low"]
    df_all = pd.concat(pieces, axis=1, sort=True).ffill().bfill()
    times = pd.to_datetime(df_all.index)

    # 1. EURNZD BUY entered at 15:30 UTC
    # 2. EURUSD BUY entered at 16:00 UTC

    # Get exact M5 bar prices around 15:30 - 15:45 and 16:00 - 16:15
    print("="*95)
    print("DIRECT MT5 BAR PRICE DATA FOR TODAY'S 2 TRADES:")
    print("="*95)

    # EURNZD 15:30 to 15:45 UTC
    print("EURNZD M5 BAR DATA (15:30 UTC - 15:45 UTC):")
    idx_1530 = [i for i, t in enumerate(times) if t.hour == 15 and t.minute in [30, 35, 40, 45]]
    for idx in idx_1530:
        t_str = str(times[idx])
        c_p = df_all["EURNZD"].iloc[idx]
        o_p = df_all["EURNZD_open"].iloc[idx]
        h_p = df_all["EURNZD_high"].iloc[idx]
        l_p = df_all["EURNZD_low"].iloc[idx]
        print(f"  Bar Time: {t_str} | Open: {o_p:.5f} | High: {h_p:.5f} | Low: {l_p:.5f} | Close: {c_p:.5f}")

    print("-"*95)
    print("EURUSD M5 BAR DATA (16:00 UTC - 16:15 UTC):")
    idx_1600 = [i for i, t in enumerate(times) if t.hour == 16 and t.minute in [0, 5, 10, 15]]
    for idx in idx_1600:
        t_str = str(times[idx])
        c_p = df_all["EURUSD"].iloc[idx]
        o_p = df_all["EURUSD_open"].iloc[idx]
        h_p = df_all["EURUSD_high"].iloc[idx]
        l_p = df_all["EURUSD_low"].iloc[idx]
        print(f"  Bar Time: {t_str} | Open: {o_p:.5f} | High: {h_p:.5f} | Low: {l_p:.5f} | Close: {c_p:.5f}")

    print("="*95)

if __name__ == "__main__":
    main()
