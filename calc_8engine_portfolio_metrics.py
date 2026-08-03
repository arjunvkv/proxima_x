#!/usr/bin/env python3
"""Calculate Portfolio Metrics for 8 Active VPS Engines."""
import pandas as pd

def main():
    engines = [
        {"Name": "1. TokyoH0_MT5", "Trades/Day": 5.0, "WR": 95.3, "Lot_6k": 0.15, "Lot_25k": 0.60, "Max_DD_6k": 18.50},
        {"Name": "2. Sunday_H22_MT5", "Trades/Day": 0.7, "WR": 84.3, "Lot_6k": 0.15, "Lot_25k": 0.60, "Max_DD_6k": 12.20},
        {"Name": "3. CPPF_Z_MT5", "Trades/Day": 0.2, "WR": 85.2, "Lot_6k": 0.15, "Lot_25k": 0.60, "Max_DD_6k": 15.40},
        {"Name": "4. MSV_Asian_Exhaustion", "Trades/Day": 0.3, "WR": 76.5, "Lot_6k": 0.02, "Lot_25k": 0.08, "Max_DD_6k": 6.80},
        {"Name": "5. NY_H21_MT5", "Trades/Day": 2.0, "WR": 64.3, "Lot_6k": 0.20, "Lot_25k": 0.80, "Max_DD_6k": 14.50},
        {"Name": "6. CPMC_Z_MT5", "Trades/Day": 1.7, "WR": 61.5, "Lot_6k": 0.15, "Lot_25k": 0.60, "Max_DD_6k": 16.20},
        {"Name": "7. ORB_Ride_MT5", "Trades/Day": 2.4, "WR": 61.6, "Lot_6k": 0.15, "Lot_25k": 0.60, "Max_DD_6k": 14.80},
        {"Name": "8. Ultra_Monster_MT5", "Trades/Day": 65.4, "WR": 74.5, "Lot_6k": 0.15, "Lot_25k": 0.60, "Max_DD_6k": 42.50},
    ]

    total_trades_day = sum(e["Trades/Day"] for e in engines)
    avg_wr = sum(e["WR"] * e["Trades/Day"] for e in engines) / total_trades_day
    total_dd_6k = sum(e["Max_DD_6k"] for e in engines)
    total_dd_25k = total_dd_6k * (25000.0 / 6000.0)

    print("="*95)
    print("COMBINED PORTFOLIO METRICS: ACTIVE 8 VPS ENGINES")
    print("="*95)
    df_e = pd.DataFrame(engines)
    print(df_e.to_string(index=False))

    print("\nPORTFOLIO TOTALS:")
    print(f"  Total Trades / Day : {total_trades_day:.1f} Trades/Day (~78 trades/day)")
    print(f"  Weighted Win Rate  : {avg_wr:.1f}%")
    print(f"  Max Daily DD ($6k) : ${total_dd_6k:.2f} ({total_dd_6k/300.0*100:.1f}% of $300 Daily Limit)")
    print(f"  Max Daily DD ($25k): ${total_dd_25k:.2f} ({total_dd_25k/1250.0*100:.1f}% of $1,250 Daily Limit)")
    print("="*95)

if __name__ == "__main__":
    main()
