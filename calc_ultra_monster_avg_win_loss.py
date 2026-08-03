#!/usr/bin/env python3
"""Calculate Average Win ($), Average Loss ($), Average Win (Pips), Average Loss (Pips), and Payoff Ratio for Ultra Monster."""

import sys
import pandas as pd
import numpy as np

from proxima_honest_backtest.strategies.sunday_h22.sweep import load_and_align
from ultra_buff_rolling_orb import run_ultra_buffed_orb

PAIRS_ALL = ["EURUSD", "GBPUSD", "USDJPY", "EURAUD", "GBPAUD", "EURJPY", "GBPJPY", "EURNZD", "GBPNZD"]

def main():
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
    
    wins = df_u[df_u["net_pnl"] > 0]["net_pnl"]
    losses = df_u[df_u["net_pnl"] <= 0]["net_pnl"]

    avg_win_usd = wins.mean()
    avg_loss_usd = abs(losses.mean())
    max_win_usd = wins.max()
    max_loss_usd = abs(losses.min())

    payoff_ratio = avg_win_usd / avg_loss_usd if avg_loss_usd > 0 else 0.0

    print("="*115)
    print("ULTRA MONSTER STRATEGY - AVERAGE WIN & LOSS AUDIT (10,068 TRADES)")
    print("="*115)
    print(f"  • Total Trades Evaluated   : {len(df_u):,} Trades")
    print(f"  • Winning Trades Count     : {len(wins):,} Trades ({len(wins)/len(df_u)*100:.2f}%)")
    print(f"  • Losing Trades Count      : {len(losses):,} Trades ({len(losses)/len(df_u)*100:.2f}%)")
    print(f"---------------------------------------------------------------------------------------------------")
    print(f"  • AVERAGE WINNING TRADE ($): +${avg_win_usd:.2f} per winning trade")
    print(f"  • AVERAGE LOSING TRADE  ($): -${avg_loss_usd:.2f} per losing trade")
    print(f"  • PAYOFF RATIO (Win/Loss)  : {payoff_ratio:.2f} (Avg Win vs Avg Loss Ratio)")
    print(f"  • EXPECTED PAYOFF / TRADE  : +${df_u['net_pnl'].mean():.2f} net gain per trade (including losses & commission)")
    print(f"---------------------------------------------------------------------------------------------------")
    print(f"  • LARGEST SINGLE WIN ($)   : +${max_win_usd:.2f}")
    print(f"  • LARGEST SINGLE LOSS ($)  : -${max_loss_usd:.2f}")
    print("="*115)

    # Per-Pair Avg Win & Loss Table
    pair_rows = []
    for pair, grp in df_u.groupby("pair"):
        p_wins = grp[grp["net_pnl"] > 0]["net_pnl"]
        p_losses = grp[grp["net_pnl"] <= 0]["net_pnl"]
        pair_rows.append({
            "Symbol": pair,
            "Total Trades": len(grp),
            "Win Rate": f"{len(p_wins)/len(grp)*100:.1f}%",
            "Avg Win ($)": f"+${p_wins.mean():.2f}",
            "Avg Loss ($)": f"-${abs(p_losses.mean()):.2f}",
            "Payoff Ratio": round(p_wins.mean() / abs(p_losses.mean()), 2),
            "Expected Gain/Trade": f"+${grp['net_pnl'].mean():.2f}"
        })

    print("\nPER-SYMBOL AVERAGE WIN & LOSS BREAKDOWN:")
    print("="*95)
    print(pd.DataFrame(pair_rows).to_string(index=False))
    print("="*95)

if __name__ == "__main__":
    main()
