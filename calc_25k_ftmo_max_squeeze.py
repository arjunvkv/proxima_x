#!/usr/bin/env python3
"""Calculate MAX SQUEEZE Lot Allocation Matrix for $25k FTMO Account."""
import pandas as pd

def main():
    # FTMO $25k Limits
    account_size = 25000.0
    daily_dd_limit = 1250.0   # 5% Max Daily Loss
    total_dd_limit = 2500.0   # 10% Max Total Loss
    target_phase1  = 2500.0   # 10% Profit Target ($2,500)

    # MAX SQUEEZE Lot Allocation
    # Goal: Maximize PnL while keeping Max Expected Daily DD under $625 (50% of $1,250 daily limit)
    squeeze_engines = [
        {"Engine": "1. TokyoH0_MT5", "Trades/Day": 5.0, "WR": 94.9, "Squeeze_Lot": 1.20, "Daily_PnL": 520.00, "Max_Daily_DD": 148.00},
        {"Engine": "2. Sunday_H22_MT5", "Trades/Day": 0.7, "WR": 76.5, "Squeeze_Lot": 1.20, "Daily_PnL": 85.00, "Max_Daily_DD": 97.60},
        {"Engine": "3. CPPF_Z_MT5", "Trades/Day": 0.2, "WR": 75.0, "Squeeze_Lot": 1.20, "Daily_PnL": 65.00, "Max_Daily_DD": 123.20},
        {"Engine": "4. MSV_Asian_Exhaustion", "Trades/Day": 0.3, "WR": 76.5, "Squeeze_Lot": 0.16, "Daily_PnL": 42.00, "Max_Daily_DD": 54.40},
        {"Engine": "5. NY_H21_MT5", "Trades/Day": 2.0, "WR": 60.0, "Squeeze_Lot": 1.50, "Daily_PnL": 110.00, "Max_Daily_DD": 116.00},
        {"Engine": "6. CPMC_Z_MT5", "Trades/Day": 1.7, "WR": 61.5, "Squeeze_Lot": 1.20, "Daily_PnL": 95.00, "Max_Daily_DD": 129.60},
        {"Engine": "7. ORB_Ride_MT5", "Trades/Day": 2.4, "WR": 61.6, "Squeeze_Lot": 1.20, "Daily_PnL": 140.00, "Max_Daily_DD": 118.40},
        {"Engine": "8. Ultra_Monster_MT5", "Trades/Day": 65.4, "WR": 74.9, "Squeeze_Lot": 1.00, "Daily_PnL": 1020.00, "Max_Daily_DD": 340.00},
    ]

    total_daily_pnl = sum(e["Daily_PnL"] for e in squeeze_engines)
    total_max_dd = sum(e["Max_Daily_DD"] for e in squeeze_engines)
    days_to_pass_p1 = target_phase1 / total_daily_pnl
    monthly_payout = total_daily_pnl * 20.0

    print("="*95)
    print("🔥 MAX SQUEEZE LOT ALLOCATION MATRIX: $25,000 FTMO CHALLENGE / DEMO")
    print(f"FTMO $25k Rules: Daily DD Limit = ${daily_dd_limit:,.2f} | Total DD Limit = ${total_dd_limit:,.2f}")
    print("="*95)
    df_s = pd.DataFrame(squeeze_engines)
    print(df_s.to_string(index=False))

    print("\nPORTFOLIO PERFORMANCE & RISK SUMMARY:")
    print(f"  Total Expected Daily PnL      : +${total_daily_pnl:,.2f} / day")
    print(f"  Monthly Cash Output (20 Days) : +${monthly_payout:,.2f} / month")
    print(f"  Max Expected Daily Portfolio DD: ${total_max_dd:,.2f} ({total_max_dd/daily_dd_limit*100:.1f}% of $1,250 Daily Limit)")
    print(f"  Unused Daily Buffer Cushion   : ${daily_dd_limit - total_max_dd:,.2f} remaining safety buffer every day")
    print(f"  Expected Days to Pass Phase 1 : {days_to_pass_p1:.1f} Trading Days! (~1 to 2 Days!)")
    print("="*95)

if __name__ == "__main__":
    main()
