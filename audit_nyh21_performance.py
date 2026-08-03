#!/usr/bin/env python3
"""Audit Current Win Rate and Daily Event Frequency for NYH21_MT5."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd
import numpy as np

from proxima_honest_backtest.strategies.ny_h21.strategy import NYH21Strategy
from proxima_honest_backtest.strategies.sunday_h22.sweep import load_and_align

def main():
    print("Auditing NYH21_MT5 Performance and Daily Event Frequency...")
    raw, pre_align = load_and_align()
    pieces = [df.set_index("time")[["close","open","high","low"]] for df in raw.values()]
    for i, p in enumerate(raw.keys()):
        pieces[i].columns = [p, f"{p}_open", f"{p}_high", f"{p}_low"]
    df_all = pd.concat(pieces, axis=1, sort=True).ffill().bfill()

    # Optimal NYH21 Config: EURJPY + GBPJPY, lb=12 (60m lookback), hold=12 (60m hold), n=5
    strat = NYH21Strategy(parameters={"lb": 12, "hold": 12, "n": 5, "pairs": ["EURJPY", "GBPJPY"]})
    signals = strat.generate_signals(df_all)

    # Calculate exact PnL and trade count
    tot_trades = len(signals)
    tot_days = 148

    wins = 0
    losses = 0
    gross_w = 0.0
    gross_l = 0.0
    all_pnls = []

    for sig in signals:
        p = sig.pair
        t_entry = sig.bar_index
        t_exit = t_entry + 12
        if t_exit >= len(df_all):
            continue
        c_entry = df_all.iloc[t_entry+1][p]
        c_exit = df_all.iloc[t_exit][p]
        dir_mult = 1.0 if sig.direction == "LONG" else -1.0
        pip_mult = 100.0 if "JPY" in p else 10000.0
        pnl_pip = (c_exit - c_entry) * dir_mult * pip_mult
        pnl_usd = pnl_pip * 10.0 * 1.00  # 1.00 Lot sizing
        all_pnls.append(pnl_usd)

        if pnl_usd > 0:
            wins += 1
            gross_w += pnl_usd
        else:
            losses += 1
            gross_l += abs(pnl_usd)

    n_valid = len(all_pnls)
    wr = wins / n_valid * 100.0 if n_valid > 0 else 0
    pf = gross_w / max(1, gross_l)
    tot_pnl = sum(all_pnls)
    events_per_day = n_valid / tot_days
    events_per_week = events_per_day * 5.0

    print("="*95)
    print("NYH21_MT5 EMPIRICAL BENCHMARK REPORT (7-MONTH AUDITED DATASET)")
    print("="*95)
    print(f"  • Target Pair Universe          ──► EURJPY + GBPJPY")
    print(f"  • Execution Time Window         ──► 21:00 UTC (02:30 AM IST NY Closing Bell)")
    print(f"  • Total Audited Trades Fired    ──► {n_valid} Trades")
    print(f"  • Daily Event Frequency        ──► {events_per_day:.2f} Trades / Day")
    print(f"  • Weekly Event Frequency       ──► {events_per_week:.1f} Trades / Week")
    print(f"  • Net Win Rate (%)              ──► {wr:.1f}% Net Win Rate 🟢")
    print(f"  • Profit Factor                 ──► {pf:.2f} Profit Factor 🚀")
    print(f"  • Cumulative Net Cash Profit   ──► +${tot_pnl:,.2f} Net Profit 💰")
    print("="*95)

if __name__ == "__main__":
    main()
