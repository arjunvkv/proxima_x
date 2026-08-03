#!/usr/bin/env python3
"""Audit Full 9-Pair Portfolio Performance for Ultra_Monster_MT5 v1.05."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd
import numpy as np

from proxima_honest_backtest.strategies.sunday_h22.sweep import load_and_align

PAIRS_ALL = ["EURUSD", "GBPUSD", "USDJPY", "EURAUD", "GBPAUD", "EURJPY", "GBPJPY", "EURNZD", "GBPNZD"]

def run_v105_portfolio_sim(df_all, close_mat, high_mat, low_mat, hours, minutes, pairs):
    trades_by_pair = {p: [] for p in pairs}
    n_bars = len(df_all)

    for t_idx in range(30, n_bars - 3):
        h = hours[t_idx]
        m = minutes[t_idx]

        if m not in [0, 30]:
            continue

        for p_idx, pair in enumerate(pairs):
            orb_high = np.max(high_mat[t_idx-6:t_idx, p_idx])
            orb_low = np.min(low_mat[t_idx-6:t_idx, p_idx])
            
            pip_mult = 100.0 if "JPY" in pair else 10000.0
            range_pips = (orb_high - orb_low) * pip_mult

            # v1.05: 12.0 Pip Gate
            if range_pips < 12.0:
                continue

            c_now = close_mat[t_idx, p_idx]
            c_entry = close_mat[t_idx+1, p_idx]
            c_exit = close_mat[t_idx+1+3, p_idx]

            # v1.05: 1.0 Pip Breakout Buffer
            buf = 1.0 / pip_mult

            if c_now > (orb_high + buf):
                pnl_pip = (c_exit - c_entry) * pip_mult
                pnl_usd = pnl_pip * 10.0 * 1.00
                trades_by_pair[pair].append(pnl_usd)
            elif c_now < (orb_low - buf):
                pnl_pip = (c_entry - c_exit) * pip_mult
                pnl_usd = pnl_pip * 10.0 * 1.00
                trades_by_pair[pair].append(pnl_usd)

    return trades_by_pair

def main():
    print("Auditing Full 9-Pair Portfolio Performance for Ultra_Monster_MT5 v1.05...")
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

    trades_map = run_v105_portfolio_sim(df_all, close_mat, high_mat, low_mat, hours, minutes, PAIRS_ALL)

    pair_rows = []
    all_trades = []

    for pair in PAIRS_ALL:
        t_list = trades_map[pair]
        all_trades.extend(t_list)
        n = len(t_list)
        w = [t for t in t_list if t > 0]
        l = [t for t in t_list if t <= 0]
        wr = len(w) / n * 100.0 if n > 0 else 0
        pf = sum(w) / max(1, abs(sum(l)))
        tot = sum(t_list)

        pair_rows.append({
            "Symbol": pair,
            "Trades Executed": n,
            "Wins 🟢": len(w),
            "Losses 🔴": len(l),
            "Net Win Rate (%)": f"{wr:.1f}%",
            "Profit Factor": f"{pf:.2f}",
            "Cumulative Net Profit": f"+${tot:,.2f}"
        })

    df_p = pd.DataFrame(pair_rows)
    print("="*105)
    print("PER-PAIR PERFORMANCE REPORT: ULTRA_MONSTER_MT5 v1.05 (12.0p GATE + 1.0p BUFFER)")
    print("="*105)
    print(df_p.to_string(index=False))

    tot_n = len(all_trades)
    tot_w = [t for t in all_trades if t > 0]
    tot_l = [t for t in all_trades if t <= 0]
    overall_wr = len(tot_w) / tot_n * 100.0 if tot_n > 0 else 0
    overall_pf = sum(tot_w) / max(1, abs(sum(tot_l)))
    tot_pnl_all = sum(all_trades)

    print("="*105)
    print("FULL 9-PAIR PORTFOLIO COMBINED TOTALS (v1.05 ENHANCED):")
    print(f"  • Total Trades Fired           ──► {tot_n:,} Trades")
    print(f"  • Combined Portfolio Win Rate  ──► {overall_wr:.1f}% Net Win Rate 🟢")
    print(f"  • Combined Portfolio PF        ──► {overall_pf:.2f} Profit Factor 🚀")
    print(f"  • Combined Cumulative Profit   ──► +${tot_pnl_all:,.2f} Net Cash Profit 💰")
    print("="*105)

if __name__ == "__main__":
    main()
