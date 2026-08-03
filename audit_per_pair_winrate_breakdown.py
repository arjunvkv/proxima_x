#!/usr/bin/env python3
"""Audit Per-Pair Win Rate Breakdown for Ultra_Monster_MT5 across 9 Pairs."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd
import numpy as np

from proxima_honest_backtest.strategies.sunday_h22.sweep import load_and_align
from ultra_buff_rolling_orb import run_ultra_buffed_orb

PAIRS_ALL = ["EURUSD", "GBPUSD", "USDJPY", "EURAUD", "GBPAUD", "EURJPY", "GBPJPY", "EURNZD", "GBPNZD"]

def main():
    print("Auditing Per-Pair Win Rate Breakdown for Ultra_Monster_MT5...")
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
    df_u["pnl_1lot"] = df_u["net_pnl"] / 0.15 * 1.00

    per_pair_stats = []

    for pair in PAIRS_ALL:
        sub = df_u[df_u["pair"] == pair]
        n_t = len(sub)
        if n_t == 0:
            continue
        pnls = sub["pnl_1lot"].values
        wins = sum(1 for p in pnls if p > 0)
        wr = wins / n_t * 100.0
        tot_pnl = sum(pnls)
        gross_w = sum(p for p in pnls if p > 0)
        gross_l = abs(sum(p for p in pnls if p <= 0))
        pf = gross_w / max(1, gross_l)

        per_pair_stats.append({
            "Symbol": pair,
            "Trades Executed": n_t,
            "Wins 🟢": wins,
            "Losses 🔴": n_t - wins,
            "Net Win Rate (%)": f"{wr:.1f}%",
            "Profit Factor": f"{pf:.2f}",
            "Cumulative Net Profit": f"+${tot_pnl:,.2f}"
        })

    df_pairs = pd.DataFrame(per_pair_stats)
    print("="*105)
    print("PER-PAIR WIN RATE BREAKDOWN REPORT: ULTRA_MONSTER_MT5 (10,068 TRADES)")
    print("="*105)
    print(df_pairs.to_string(index=False))

    tot_trades = len(df_u)
    tot_wins = sum(1 for p in df_u["pnl_1lot"] if p > 0)
    overall_wr = tot_wins / tot_trades * 100.0
    tot_pnl_all = sum(df_u["pnl_1lot"])
    tot_gw = sum(p for p in df_u["pnl_1lot"] if p > 0)
    tot_gl = abs(sum(p for p in df_u["pnl_1lot"] if p <= 0))
    overall_pf = tot_gw / max(1, tot_gl)

    print("="*105)
    print(f"FULL 9-PAIR PORTFOLIO COMBINED TOTALS:")
    print(f"  • Total Trades Fired           ──► {tot_trades:,} Trades")
    print(f"  • Combined Portfolio Win Rate  ──► {overall_wr:.1f}% Net Win Rate 🟢")
    print(f"  • Combined Portfolio PF        ──► {overall_pf:.2f} Profit Factor 🚀")
    print(f"  • Combined Cumulative Profit   ──► +${tot_pnl_all:,.2f} Net Cash Profit 💰")
    print("="*105)

if __name__ == "__main__":
    main()
