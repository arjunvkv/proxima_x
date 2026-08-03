#!/usr/bin/env python3
"""Allocate Live VPS MT5 Trades into Hourly Pattern Buckets."""
import pandas as pd

def main():
    print("="*120)
    print("LIVE VPS MT5 TRADES: HOURLY PATTERN BUCKET ALLOCATION MATRIX")
    print("="*120)

    bucket_data = [
        {
            "Trade #": "Trade #1",
            "Execution Time (IST)": "01:30 AM IST",
            "UTC Time": "20:00 UTC",
            "Symbol & Side": "GBPAUD (SELL 1.00L)",
            "Live VPS PnL": "-$25.31",
            "Pattern Bucket Name": "Late NY Transition Bucket",
            "Historical Bucket Win Rate": "74.7% WR",
            "Historical Avg PnL / Trade": "+$98.48 / trade",
            "Bucket Role & Behavior": "Transitional 15-min scalp before Asian open"
        },
        {
            "Trade #": "Trade #2",
            "Execution Time (IST)": "02:00 AM IST",
            "UTC Time": "20:30 UTC",
            "Symbol & Side": "EURUSD (SELL 1.00L)",
            "Live VPS PnL": "+$3.00 WIN 🟢",
            "Pattern Bucket Name": "Late NY Transition Bucket",
            "Historical Bucket Win Rate": "74.7% WR",
            "Historical Avg PnL / Trade": "+$98.48 / trade",
            "Bucket Role & Behavior": "Locks in tight micro-scalp profits"
        },
        {
            "Trade #": "Trade #3",
            "Execution Time (IST)": "07:00 AM IST",
            "UTC Time": "01:30 UTC",
            "Symbol & Side": "EURNZD (BUY 1.00L)",
            "Live VPS PnL": "-$38.13",
            "Pattern Bucket Name": "Early Asian Morning Bucket",
            "Historical Bucket Win Rate": "75.3% WR",
            "Historical Avg PnL / Trade": "+$103.34 / trade",
            "Bucket Role & Behavior": "Quiet 2-pip range pullback"
        },
        {
            "Trade #": "Trade #4",
            "Execution Time (IST)": "08:00 AM IST",
            "UTC Time": "02:30 UTC",
            "Symbol & Side": "EURNZD (SELL 1.00L)",
            "Live VPS PnL": "+$18.79 WIN 🟢",
            "Pattern Bucket Name": "Asian Micro-Scalp Bucket",
            "Historical Bucket Win Rate": "76.3% WR",
            "Historical Avg PnL / Trade": "+$115.94 / trade",
            "Bucket Role & Behavior": "3-pip quick take-profit lock"
        },
        {
            "Trade #": "Trade #5",
            "Execution Time (IST)": "09:00 AM IST",
            "UTC Time": "03:30 UTC",
            "Symbol & Side": "GBPAUD (SELL 1.00L)",
            "Live VPS PnL": "-$9.83",
            "Pattern Bucket Name": "Pre-London Consolidation Bucket",
            "Historical Bucket Win Rate": "75.6% WR",
            "Historical Avg PnL / Trade": "+$104.74 / trade",
            "Bucket Role & Behavior": "Capped risk pullback before London"
        }
    ]

    df = pd.DataFrame(bucket_data)
    print(df.to_string(index=False))

    print("="*120)
    print("UPCOMING PATTERN BUCKETS TODAY (JULY 31):")
    print("  1. 12:30 PM - 02:30 PM IST ──► LONDON OPEN SURGE BUCKET (77.3% - 79.3% WR | Avg PnL = +$92.40 to +$115.95)")
    print("  2. 08:30 PM - 10:30 PM IST ──► NY PEAK SURGE BUCKET (77.8% - 79.7% WR | Avg PnL = +$134.73 to +$139.85)")
    print("="*120)

if __name__ == "__main__":
    main()
