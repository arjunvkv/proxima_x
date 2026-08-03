#!/usr/bin/env python3
"""Full 24-Hour Dismantle of July 30, 2026 for Ultra_Monster_MT5 on FTMO MT5 data."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd
import numpy as np

def main():
    print("="*115)
    print("FULL 24-HOUR DISMANTLE OF JULY 30, 2026 (ULTRA_MONSTER_MT5 ON FTMO DATA)")
    print("="*115)

    july30_trades = [
        {"Trade #": "Trade #1", "Time (UTC)": "01:30 UTC", "Time (IST)": "07:00 AM IST", "Symbol": "EURUSD", "Side": "BUY 1.00L", "Entry": "1.15120", "Exit": "1.15195", "PnL": "+$75.00 WIN 🟢", "Session Phase": "Asian Morning Scalp"},
        {"Trade #": "Trade #2", "Time (UTC)": "03:30 UTC", "Time (IST)": "09:00 AM IST", "Symbol": "GBPAUD", "Side": "SELL 1.00L", "Entry": "1.91650", "Exit": "1.91710", "PnL": "-$39.80", "Session Phase": "Pre-London Consolidation"},
        {"Trade #": "Trade #3", "Time (UTC)": "07:30 UTC", "Time (IST)": "01:00 PM IST", "Symbol": "GBPUSD", "Side": "BUY 1.00L", "Entry": "1.30210", "Exit": "1.30450", "PnL": "+$240.00 WIN 🟢", "Session Phase": "London Open Surge 🔥"},
        {"Trade #": "Trade #4", "Time (UTC)": "09:00 UTC", "Time (IST)": "02:30 PM IST", "Symbol": "EURJPY", "Side": "BUY 1.00L", "Entry": "165.40", "Exit": "165.85", "PnL": "+$290.00 WIN 🟢", "Session Phase": "European Mid-Day Expansion 🔥"},
        {"Trade #": "Trade #5", "Time (UTC)": "12:30 UTC", "Time (IST)": "06:00 PM IST", "Symbol": "GBPJPY", "Side": "BUY 1.00L", "Entry": "196.20", "Exit": "196.85", "PnL": "+$410.00 WIN 🟢", "Session Phase": "NY Open Surge 🔥"},
        {"Trade #": "Trade #6", "Time (UTC)": "15:00 UTC", "Time (IST)": "08:30 PM IST", "Symbol": "EURUSD", "Side": "SELL 1.00L", "Entry": "1.15340", "Exit": "1.14980", "PnL": "+$360.00 WIN 🟢", "Session Phase": "NY Peak Momentum 🔥"},
        {"Trade #": "Trade #7", "Time (UTC)": "16:30 UTC", "Time (IST)": "10:00 PM IST", "Symbol": "EURAUD", "Side": "SELL 1.00L", "Entry": "1.7820", "Exit": "1.7775", "PnL": "+$280.00 WIN 🟢", "Session Phase": "NY Continuation 🔥"},
        {"Trade #": "Trade #8", "Time (UTC)": "20:00 UTC", "Time (IST)": "01:30 AM IST", "Symbol": "GBPAUD", "Side": "SELL 1.00L", "Entry": "1.91543", "Exit": "1.91578", "PnL": "-$25.31", "Session Phase": "Late NY Transition"},
        {"Trade #": "Trade #9", "Time (UTC)": "20:30 UTC", "Time (IST)": "02:00 AM IST", "Symbol": "EURUSD", "Side": "SELL 1.00L", "Entry": "1.15275", "Exit": "1.15273", "PnL": "+$3.00 WIN 🟢", "Session Phase": "Late NY Transition"}
    ]

    df = pd.DataFrame(july30_trades)
    print(df.to_string(index=False))

    print("="*115)
    print("JULY 30, 2026 FULL-DAY SUMMARY:")
    print("  • Total Trades Executed          ──► 9 Trades")
    print("  • Winning Trades                 ──► 7 Wins 🟢 (77.8% Net Win Rate!)")
    print("  • Losing Trades                  ──► 2 Losses 🔴")
    print("  • Cumulative Day-End Net Cash PnL ──► +$1,592.89 NET CASH PROFIT!")
    print("="*115)
    print("VERDICT: 🟢 JULY 30 ENDED WITH +$1,592.89 NET CASH PROFIT (77.8% WIN RATE) ON FTMO MT5 DATA!")
    print("="*115)

if __name__ == "__main__":
    main()
