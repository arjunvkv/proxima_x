#!/usr/bin/env python3
"""Calculate Losing Streak Probabilities and PnL Win/Loss Factors for Ultra_Monster_MT5."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd
import numpy as np

from proxima_honest_backtest.strategies.sunday_h22.sweep import load_and_align
from ultra_buff_rolling_orb import run_ultra_buffed_orb

PAIRS_ALL = ["EURUSD", "GBPUSD", "USDJPY", "EURAUD", "GBPAUD", "EURJPY", "GBPJPY", "EURNZD", "GBPNZD"]

def main():
    print("Loading M5 dataset for Ultra_Monster_MT5 Streak & Payoff Audit...")
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
    pnls = df_u["net_pnl"].values

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    avg_win_015 = np.mean(wins) if len(wins) > 0 else 0.0
    avg_loss_015 = abs(np.mean(losses)) if len(losses) > 0 else 0.0

    avg_win_squeeze = avg_win_015 / 0.15 * 1.00
    avg_loss_squeeze = avg_loss_015 / 0.15 * 1.00

    payoff_ratio = avg_win_015 / avg_loss_015 if avg_loss_015 > 0 else 0.0
    profit_factor = sum(wins) / abs(sum(losses)) if sum(losses) != 0 else 0.0

    # Calculate Max Losing Streak
    curr_streak = 0
    max_losing_streak = 0
    for p in pnls:
        if p < 0:
            curr_streak += 1
            if curr_streak > max_losing_streak:
                max_losing_streak = curr_streak
        else:
            curr_streak = 0

    p_loss = len(losses) / len(pnls)

    print("="*95)
    print("EMPIRICAL RISK & PAYOFF REPORT: ULTRA_MONSTER_MT5 (10,068 TRADES)")
    print("="*95)
    print("1. WIN & LOSS FACTOR IN PNL (PAYOFF RATIO):")
    print(f"   Average Winning Trade (0.15 Lot) ──► +${avg_win_015:.2f}")
    print(f"   Average Losing Trade  (0.15 Lot) ──► -${avg_loss_015:.2f}")
    print(f"   Average Winning Trade (1.00 Lot) ──► +${avg_win_squeeze:.2f}")
    print(f"   Average Losing Trade  (1.00 Lot) ──► -${avg_loss_squeeze:.2f}")
    print(f"   Win/Loss Payoff Ratio           ──► {payoff_ratio:.2f}x (Avg Win is {payoff_ratio:.2f}x larger than Avg Loss!)")
    print(f"   Overall Profit Factor           ──► {profit_factor:.2f}")

    print("\n2. CONSECUTIVE LOSING STREAK STATISTICS:")
    print(f"   Maximum Observed Loss Streak (in 10,068 Trades) ──► {max_losing_streak} Consecutive Losses")
    print(f"   Probability of 2 Consecutive Losses  ──► {p_loss**2*100:.2f}%  (1 in {1/(p_loss**2):.0f} trade sequences)")
    print(f"   Probability of 3 Consecutive Losses  ──► {p_loss**3*100:.2f}%  (1 in {1/(p_loss**3):.0f} trade sequences)")
    print(f"   Probability of 4 Consecutive Losses  ──► {p_loss**4*100:.2f}%  (1 in {1/(p_loss**4):.0f} trade sequences)")
    print(f"   Probability of 5 Consecutive Losses  ──► {p_loss**5*100:.2f}%  (1 in {1/(p_loss**5):.0f} trade sequences)")
    print("="*95)

if __name__ == "__main__":
    main()
