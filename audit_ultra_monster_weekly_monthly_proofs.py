#!/usr/bin/env python3
"""Audit Ultra Monster (Conditioned Rolling ORB) Monthly and Weekly Performance Breakdown."""

import sys, time
from pathlib import Path
import pandas as pd
import numpy as np

from proxima_honest_backtest.strategies.sunday_h22.sweep import load_and_align
from proxima_honest_backtest.execution.execution_simulator import ExecutionSimulator

PAIRS_ALL = ["EURUSD", "GBPUSD", "USDJPY", "EURAUD", "GBPAUD", "EURJPY", "GBPJPY", "EURNZD", "GBPNZD"]

def run_ultra_monster_backtest(df_all, close_mat, open_mat, high_mat, low_mat, hours, minutes, min_range_pips=6.0, trigger_mins=[0, 30], hold_bars=3):
    sim = ExecutionSimulator("fundednext")
    n_bars = len(df_all)
    pair_indices = list(range(len(PAIRS_ALL)))

    in_pos = [False] * len(PAIRS_ALL)
    exit_bar = [0] * len(PAIRS_ALL)
    entry_pr = [0.0] * len(PAIRS_ALL)
    direction = [0] * len(PAIRS_ALL)
    
    trades = []

    for t in range(13, n_bars):
        # Exits
        for p_i in pair_indices:
            if in_pos[p_i] and t >= exit_bar[p_i]:
                c_exit = close_mat[t, p_i]
                c_entry = entry_pr[p_i]
                dir_i = direction[p_i]
                gross_pnl = (c_exit - c_entry) / c_entry * 50000.0 * dir_i
                comm = sim.profile.commission_per_lot * 0.15 * 2 # $3/lot on 0.15 lots
                net_pnl = gross_pnl - comm
                
                trades.append({
                    "time": pd.to_datetime(df_all.index[t]),
                    "pair": PAIRS_ALL[p_i],
                    "dir": "BUY" if dir_i == 1 else "SELL",
                    "entry_price": c_entry,
                    "exit_price": c_exit,
                    "gross_pnl": gross_pnl,
                    "net_pnl": net_pnl,
                    "win": net_pnl > 0
                })
                in_pos[p_i] = False

        # Entry Trigger
        if minutes[t] in trigger_mins:
            for p_i in pair_indices:
                if in_pos[p_i]: continue
                
                h_prev = np.max(high_mat[t-12:t, p_i])
                l_prev = np.min(low_mat[t-12:t, p_i])
                c_now = close_mat[t, p_i]
                
                mult = 100.0 if "JPY" in PAIRS_ALL[p_i] else 10000.0
                range_pips = (h_prev - l_prev) * mult
                if range_pips < min_range_pips:
                    continue

                if c_now > h_prev:
                    in_pos[p_i] = True
                    direction[p_i] = 1
                    exit_bar[p_i] = t + hold_bars
                    entry_pr[p_i] = open_mat[t, p_i]
                elif c_now < l_prev:
                    in_pos[p_i] = True
                    direction[p_i] = -1
                    exit_bar[p_i] = t + hold_bars
                    entry_pr[p_i] = open_mat[t, p_i]

    return pd.DataFrame(trades)

def main():
    print("Loading M5 dataset for Ultra Monster Local Proofs...")
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

    print("\nExecuting Ultra Monster Local Backtest...")
    df_trades = run_ultra_monster_backtest(df_all, close_mat, open_mat, high_mat, low_mat, hours, minutes, min_range_pips=6.0, trigger_mins=[0, 30], hold_bars=3)

    df_trades["year_month"] = df_trades["time"].dt.strftime("%Y-%m")
    df_trades["week"] = df_trades["time"].dt.strftime("%Y-W%U")

    # Overall Summary
    total_trades = len(df_trades)
    wins = (df_trades["net_pnl"] > 0).sum()
    losses = (df_trades["net_pnl"] <= 0).sum()
    win_rate = (wins / total_trades) * 100.0 if total_trades > 0 else 0
    net_pnl = df_trades["net_pnl"].sum()
    gross_win = df_trades[df_trades["net_pnl"] > 0]["net_pnl"].sum()
    gross_loss = abs(df_trades[df_trades["net_pnl"] < 0]["net_pnl"].sum())
    pf = gross_win / gross_loss if gross_loss > 0 else 0

    print("="*115)
    print("ULTRA MONSTER STRATEGY - LOCAL PROOF OVERALL SUMMARY")
    print("="*115)
    print(f"Total Trades Taken  : {total_trades}")
    print(f"Winning Trades      : {wins}")
    print(f"Losing Trades       : {losses}")
    print(f"Overall Win Rate    : {win_rate:.2f}%")
    print(f"Gross Profit        : +${gross_win:.2f}")
    print(f"Gross Loss          : -${gross_loss:.2f}")
    print(f"Net Realized PnL    : +${net_pnl:.2f}")
    print(f"Profit Factor (PF)  : {pf:.2f}")
    print("="*115)

    # Monthly Breakdown
    monthly_rows = []
    for ym, group in df_trades.groupby("year_month"):
        n_t = len(group)
        w = (group["net_pnl"] > 0).sum()
        l = (group["net_pnl"] <= 0).sum()
        wr = (w / n_t) * 100.0 if n_t > 0 else 0
        pnl = group["net_pnl"].sum()
        gw = group[group["net_pnl"] > 0]["net_pnl"].sum()
        gl = abs(group[group["net_pnl"] < 0]["net_pnl"].sum())
        m_pf = gw / gl if gl > 0 else 0
        monthly_rows.append({
            "Month": ym,
            "Trades": n_t,
            "Wins": w,
            "Losses": l,
            "Win Rate": f"{wr:.1f}%",
            "Net PnL": f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}",
            "PF": f"{m_pf:.2f}"
        })

    df_monthly = pd.DataFrame(monthly_rows)
    print("\nMONTH-BY-MONTH PERFORMANCE BREAKDOWN:")
    print("="*85)
    print(df_monthly.to_string(index=False))
    print("="*85)

    # Last 6 Weeks Breakdown
    weekly_rows = []
    unique_weeks = sorted(df_trades["week"].unique())[-8:]
    for wk in unique_weeks:
        group = df_trades[df_trades["week"] == wk]
        n_t = len(group)
        w = (group["net_pnl"] > 0).sum()
        l = (group["net_pnl"] <= 0).sum()
        wr = (w / n_t) * 100.0 if n_t > 0 else 0
        pnl = group["net_pnl"].sum()
        gw = group[group["net_pnl"] > 0]["net_pnl"].sum()
        gl = abs(group[group["net_pnl"] < 0]["net_pnl"].sum())
        w_pf = gw / gl if gl > 0 else 0
        weekly_rows.append({
            "Week": wk,
            "Trades": n_t,
            "Wins": w,
            "Losses": l,
            "Win Rate": f"{wr:.1f}%",
            "Net PnL": f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}",
            "PF": f"{w_pf:.2f}"
        })

    df_weekly = pd.DataFrame(weekly_rows)
    print("\nRECENT WEEKS PERFORMANCE BREAKDOWN:")
    print("="*85)
    print(df_weekly.to_string(index=False))
    print("="*85)

    # Per Pair Performance Breakdown
    pair_rows = []
    for pair, group in df_trades.groupby("pair"):
        n_t = len(group)
        w = (group["net_pnl"] > 0).sum()
        l = (group["net_pnl"] <= 0).sum()
        wr = (w / n_t) * 100.0 if n_t > 0 else 0
        pnl = group["net_pnl"].sum()
        gw = group[group["net_pnl"] > 0]["net_pnl"].sum()
        gl = abs(group[group["net_pnl"] < 0]["net_pnl"].sum())
        p_pf = gw / gl if gl > 0 else 0
        pair_rows.append({
            "Symbol": pair,
            "Trades": n_t,
            "Wins": w,
            "Losses": l,
            "Win Rate": f"{wr:.1f}%",
            "Net PnL": f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}",
            "PF": f"{p_pf:.2f}"
        })

    df_pairs = pd.DataFrame(pair_rows)
    print("\nPER-PAIR PERFORMANCE BREAKDOWN:")
    print("="*85)
    print(df_pairs.to_string(index=False))
    print("="*85)

if __name__ == "__main__":
    main()
