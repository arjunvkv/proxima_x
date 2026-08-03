#!/usr/bin/env python3
"""Dismantle July 22, 2026 Trade-by-Trade Execution Audit."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd
import numpy as np

from proxima_honest_backtest.strategies.sunday_h22.sweep import load_and_align
from ultra_buff_rolling_orb import run_ultra_buffed_orb

PAIRS_ALL = ["EURUSD", "GBPUSD", "USDJPY", "EURAUD", "GBPAUD", "EURJPY", "GBPJPY", "EURNZD", "GBPNZD"]

def main():
    print("Dismantling July 22, 2026 Trade-by-Trade Dataset...")
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

    # Filter July 22, 2026
    df_jul22 = df_u[df_u["date"] == pd.to_datetime("2026-07-22").date()].copy()
    df_jul22["pnl_1lot"] = df_jul22["net_pnl"] / 0.15 * 1.00
    df_jul22["outcome"] = df_jul22["pnl_1lot"].apply(lambda x: "🟢 WIN" if x > 0 else "🔴 LOSS")

    print("="*115)
    print("TRADE-BY-TRADE DISMANTLE: WEDNESDAY, JULY 22, 2026 (ALL 35 TRADES)")
    print("="*115)

    display_cols = ["time", "pair", "side", "entry_price", "exit_price", "pnl_1lot", "outcome"]
    if all(c in df_jul22.columns for c in display_cols):
        print(df_jul22[display_cols].to_string(index=False))
    else:
        print(df_jul22[["time", "pair", "net_pnl", "pnl_1lot", "outcome"]].to_string(index=False))

    tot_wins = sum(1 for p in df_jul22["pnl_1lot"] if p > 0)
    tot_losses = sum(1 for p in df_jul22["pnl_1lot"] if p <= 0)
    tot_pnl = sum(df_jul22["pnl_1lot"])
    avg_win = np.mean([p for p in df_jul22["pnl_1lot"] if p > 0])
    avg_loss = abs(np.mean([p for p in df_jul22["pnl_1lot"] if p <= 0]))

    print("="*115)
    print(f"JULY 22 AUDIT SUMMARY:")
    print(f"  Total Trades Fired ──► {len(df_jul22)} Trades")
    print(f"  Winning Trades     ──► {tot_wins} WINS 🟢 ({tot_wins/len(df_jul22)*100:.1f}% WR)")
    print(f"  Losing Trades      ──► {tot_losses} LOSSES 🔴")
    print(f"  Average Win (1.00L)──► +${avg_win:.2f}")
    print(f"  Average Loss (1.00L)──► -${avg_loss:.2f}")
    print(f"  Payoff Ratio       ──► {avg_win/avg_loss:.2f}x")
    print(f"  Net Daily Cash PnL ──► +${tot_pnl:,.2f} NET PROFIT")
    print("="*115)

if __name__ == "__main__":
    main()
