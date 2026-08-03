#!/usr/bin/env python3
"""Calculate Binomial Distribution for 3-Trade Sample at 75% Win Rate."""
import math

def main():
    p = 0.75  # Win Rate
    q = 0.25  # Loss Rate
    n = 3     # Sample size

    # Probability of 3 Wins: (3 C 3) * (0.75)^3 * (0.25)^0
    p_3w = (p**3) * 100.0

    # Probability of 2 Wins, 1 Loss: (3 C 2) * (0.75)^2 * (0.25)^1
    p_2w = 3 * (p**2) * q * 100.0

    # Probability of 1 Win, 2 Losses: (3 C 1) * (0.75)^1 * (0.25)^2
    p_1w = 3 * p * (q**2) * 100.0

    # Probability of 0 Wins, 3 Losses: (3 C 0) * (0.75)^0 * (0.25)^3
    p_0w = (q**3) * 100.0

    print("="*95)
    print("BINOMIAL PROBABILITY DISTRIBUTION: 3-TRADE MICRO SAMPLE AT 75% WIN RATE")
    print("="*95)
    print(f"  Probability of 3 Wins / 0 Losses ──► {p_3w:.1f}%")
    print(f"  Probability of 2 Wins / 1 Loss   ──► {p_2w:.1f}%")
    print(f"  Probability of 1 Win / 2 Losses  ──► {p_1w:.1f}% (1 in every 7 3-trade sequences!)")
    print(f"  Probability of 0 Wins / 3 Losses ──► {p_0w:.1f}%")
    print("="*95)
    print("VERDICT: 🟢 TAKING 2 LOSSES IN A 3-TRADE SAMPLE IS 100% MATHEMATICALLY NORMAL (14.1% PROBABILITY)!")
    print("="*95)

if __name__ == "__main__":
    main()
