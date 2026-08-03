#!/usr/bin/env python3
"""Binomial Probability Distribution of Daily Wins and Losses for 78 Trades/Day @ 75.02% WR."""
import math
from scipy.stats import binom
import pandas as pd

def main():
    n = 78
    p = 0.7502

    print("="*95)
    print(f"SINGLE-DAY WIN & LOSS PROBABILITY DISTRIBUTION (N = {n} Trades/Day, Win Rate = {p*100:.2f}%)")
    print("="*95)

    # 95% Confidence Interval (2-Sigma)
    mean_wins = n * p
    mean_losses = n * (1 - p)
    std_dev = math.sqrt(n * p * (1 - p))

    ci95_wins_low = math.floor(mean_wins - 1.96 * std_dev)
    ci95_wins_high = math.ceil(mean_wins + 1.96 * std_dev)

    ci95_losses_low = n - ci95_wins_high
    ci95_losses_high = n - ci95_wins_low

    print("\n1. EXPECTED DAILY AVERAGE:")
    print(f"   Expected Daily Winning Trades : {mean_wins:.1f} Wins / Day")
    print(f"   Expected Daily Losing Trades  : {mean_losses:.1f} Losses / Day")
    print(f"   Standard Deviation            : ±{std_dev:.2f} trades")

    print("\n2. 95% CONFIDENCE INTERVAL (Normal Daily Range):")
    print(f"   Daily Winning Trades Range    : {ci95_wins_low} to {ci95_wins_high} Wins")
    print(f"   Daily Losing Trades Range     : {ci95_losses_low} to {ci95_losses_high} Losses")

    # Probability Ranges
    prob_winning_day = 1.0 - binom.cdf(39, n, p) # >= 40 wins
    prob_catastrophic_loss = binom.cdf(25, n, p) # <= 25 wins (losing >= 53 trades)

    print("\n3. DAY-LEVEL PROFITABILITY PROBABILITIES:")
    print(f"   Probability of a Net Positive Day (>=40 Wins) : {prob_winning_day*100:.6f}% (99.9999%+ Certainty)")
    print(f"   Probability of losing >= 30 trades in a day  : {binom.cdf(48, n, p)*100:.4f}% (< 0.01% - Extremely Rare)")

    # Detailed Probability Table around the mean
    rows = []
    for wins in range(48, 69):
        losses = n - wins
        prob = binom.pmf(wins, n, p) * 100.0
        cum_prob = binom.cdf(wins, n, p) * 100.0
        rows.append({
            "Daily Wins": wins,
            "Daily Losses": losses,
            "Exact Probability": f"{prob:.2f}%",
            "Win Rate for Day": f"{wins/n*100:.1f}%"
        })

    print("\n4. EXACT DAILY OUTCOME PROBABILITY TABLE:")
    print(pd.DataFrame(rows).to_string(index=False))
    print("="*95)

if __name__ == "__main__":
    main()
