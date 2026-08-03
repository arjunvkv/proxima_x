#!/usr/bin/env python3
"""Test Half-Hourly Candle Close ORB Breakouts vs Continuous Ticks."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd
import numpy as np

from proxima_honest_backtest.strategies.sunday_h22.sweep import load_and_align

PAIRS_ALL = ["EURUSD", "GBPUSD", "USDJPY", "EURAUD", "GBPAUD", "EURJPY", "GBPJPY", "EURNZD", "GBPNZD"]

def run_half_hourly_orb_sim(df_all, close_mat, high_mat, low_mat, hours, minutes, pairs):
    trades = []
    n_bars = len(df_all)

    # Scans only at half-hour candle closes (minutes 0 and 30) between 07:00 UTC and 09:30 UTC
    for t_idx in range(30, n_bars - 3):
        h = hours[t_idx]
        m = minutes[t_idx]

        if h in [7, 8, 9] and m in [0, 30]:
            for p_idx, pair in enumerate(pairs):
                orb_high = np.max(high_mat[t_idx-6:t_idx, p_idx])
                orb_low = np.min(low_mat[t_idx-6:t_idx, p_idx])
                c_now = close_mat[t_idx, p_idx]
                c_entry = close_mat[t_idx+1, p_idx]
                c_exit = close_mat[t_idx+1+3, p_idx]

                if c_now > orb_high:
                    pnl_pip = (c_exit - c_entry) * 10000 if "JPY" not in pair else (c_exit - c_entry) * 100
                    pnl_usd = pnl_pip * 10.0 * 1.00
                    trades.append(pnl_usd)
                elif c_now < orb_low:
                    pnl_pip = (c_entry - c_exit) * 10000 if "JPY" not in pair else (c_entry - c_exit) * 100
                    pnl_usd = pnl_pip * 10.0 * 1.00
                    trades.append(pnl_usd)

    return trades

def main():
    print("Testing Half-Hourly Candle Close ORB Breakout Strategy...")
    raw, pre_align = load_and_align()
    pieces = [df.set_index("time")[["close","open","high","low"]] for df in raw.values()]
    for i, p in enumerate(raw.keys()):
        pieces[i].columns = [p, f"{p}_open", f"{p}_high", f"{p}_low"]
    df_all = pd.concat(pieces, axis=1, sort=True).ffill().bfill()
    times = pd.to_datetime(df_all.index)

    close_mat = df_all[[p for p in PAIRS_ALL]].values
    high_mat = df_all[[f"{p}_high" for p in PAIRS_ALL]].values
    low_mat = df_all[[f"{p}_low" for p in PAIRS_ALL]].values

    hours = times.hour.values
    minutes = times.minute.values

    trades = run_half_hourly_orb_sim(df_all, close_mat, high_mat, low_mat, hours, minutes, PAIRS_ALL)

    n_t = len(trades)
    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]
    wr = len(wins) / n_t * 100.0 if n_t > 0 else 0
    tot_pnl = sum(trades)
    pf = sum(wins) / max(1, abs(sum(losses)))

    print("="*95)
    print("HALF-HOURLY CANDLE CLOSE ORB BREAKOUT REPORT")
    print("="*95)
    print(f"  Total Trades Fired           ──► {n_t:,} Trades")
    print(f"  Net Win Rate (%)             ──► {wr:.1f}% Win Rate")
    print(f"  Profit Factor                ──► {pf:.2f} Profit Factor")
    print(f"  Cumulative Net PnL (1.00L)   ──► +${tot_pnl:,.2f} Net Cash Profit")
    print("="*95)

if __name__ == "__main__":
    main()
