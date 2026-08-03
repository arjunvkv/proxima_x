#!/usr/bin/env python3
"""Calculate Trade-Weighted Portfolio Win Rate Breakdown."""
import pandas as pd

def main():
    engines = [
        {"Engine": "Ultra_Monster_MT5", "Trades/Day": 65.4, "WR": 74.5, "Weight_Pct": 84.2, "Weighted_Wins": 48.72},
        {"Engine": "TokyoH0_MT5", "Trades/Day": 5.0, "WR": 95.3, "Weight_Pct": 6.4, "Weighted_Wins": 4.77},
        {"Engine": "ORB_Ride_MT5", "Trades/Day": 2.4, "WR": 61.6, "Weight_Pct": 3.1, "Weighted_Wins": 1.48},
        {"Engine": "NY_H21_MT5", "Engine": "NY_H21_MT5", "Trades/Day": 2.0, "WR": 64.3, "Weight_Pct": 2.6, "Weighted_Wins": 1.29},
        {"Engine": "CPMC_Z_MT5", "Trades/Day": 1.7, "WR": 61.5, "Weight_Pct": 2.2, "Weighted_Wins": 1.05},
        {"Engine": "Sunday_H22_MT5", "Trades/Day": 0.7, "WR": 84.3, "Weight_Pct": 0.9, "Weighted_Wins": 0.59},
        {"Engine": "MSV_Asian_Exhaustion", "Trades/Day": 0.3, "WR": 76.5, "Weight_Pct": 0.4, "Weighted_Wins": 0.23},
        {"Engine": "CPPF_Z_MT5", "Trades/Day": 0.2, "WR": 85.2, "Weight_Pct": 0.2, "Weighted_Wins": 0.17},
    ]

    total_trades = sum(e["Trades/Day"] for e in engines)
    total_winning_trades = sum(e["Trades/Day"] * (e["WR"]/100.0) for e in engines)
    portfolio_wr = (total_winning_trades / total_trades) * 100.0

    print("="*95)
    print("EXACT TRADE-WEIGHTED PORTFOLIO WIN RATE BREAKDOWN")
    print("="*95)
    df_e = pd.DataFrame(engines)
    print(df_e.to_string(index=False))

    print("\nMATHEMATICAL PROOF:")
    print(f"  Total Trades / Day          : {total_trades:.1f}")
    print(f"  Total Winning Trades / Day  : {total_winning_trades:.2f}")
    print(f"  Calculated Portfolio Win Rate: {total_winning_trades:.2f} / {total_trades:.1f} = {portfolio_wr:.2f}% (~75.0%)")
    print("="*95)

if __name__ == "__main__":
    main()
