#!/usr/bin/env python3
"""Test Ultra_Monster_MT5 v1.05 with Full MSV Regime Filter."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd
import numpy as np

from proxima_honest_backtest.strategies.sunday_h22.sweep import load_and_align
from ultra_buff_rolling_orb import run_ultra_buffed_orb

PAIRS_ALL = ["EURUSD", "GBPUSD", "USDJPY", "EURAUD", "GBPAUD", "EURJPY", "GBPJPY", "EURNZD", "GBPNZD"]

def main():
    print("Auditing Full MSV Regime Filter + 12.0p Volatility Gate for Ultra_Monster_MT5...")
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

    # Full MSV Regime Filter + 12.0p Volatility Gate
    df_u = run_ultra_buffed_orb(df_all, close_mat, open_mat, high_mat, low_mat, hours, minutes, PAIRS_ALL, 12.0, range(0, 24), [0, 30], 3)
    df_u["pnl_1lot"] = df_u["net_pnl"] / 0.15 * 1.00

    pnls = df_u["pnl_1lot"].values
    n_t = len(pnls)
    wins = sum(1 for p in pnls if p > 0)
    wr = wins / n_t * 100.0
    tot_pnl = sum(pnls)
    gross_w = sum(p for p in pnls if p > 0)
    gross_l = abs(sum(p for p in pnls if p <= 0))
    pf = gross_w / max(1, gross_l)

    print("="*105)
    print("FULL MSV REGIME FILTER + 12.0p VOLATILITY GATE PERFORMANCE REPORT")
    print("="*105)
    print(f"  Total Trades Executed       ──► {n_t:,} Trades")
    print(f"  Net Portfolio Win Rate (%)   ──► {wr:.1f}% Win Rate 🟢")
    print(f"  Profit Factor               ──► {pf:.2f} Profit Factor 🚀")
    print(f"  Cumulative Net Cash Profit  ──► +${tot_pnl:,.2f} Net Profit 💰")
    print("="*105)

if __name__ == "__main__":
    main()
