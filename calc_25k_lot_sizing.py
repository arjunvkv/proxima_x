#!/usr/bin/env python3
"""Calculate Optimal Lot Allocations & Expected Returns for a $25,000 Account."""
import pandas as pd

def main():
    account_size = 25000.0
    daily_limit = account_size * 0.05   # $1,250.00
    total_dd_limit = account_size * 0.10 # $2,500.00

    print("="*95)
    print(f"PROXIMA ENGINE LOT ALLOCATION MATRIX: ${account_size:,.0f} FUNDEDNEXT ACCOUNT")
    print(f"Daily Limit: ${daily_limit:,.2f} (5.0%) | Total DD Limit: ${total_dd_limit:,.2f} (10.0%)")
    print("="*95)

    tiers = [
        {"Tier": "Tier 1: Conservative", "tokyo": 0.45, "sunday": 0.45, "cppf": 0.45, "msv": 0.06, "ny": 0.60, "cpmc": 0.45, "orb": 0.45, "dd_pct": 22.4, "max_dd": 280.0, "pnl": 6800.0},
        {"Tier": "Tier 2: Max Profit (RECOMMENDED)", "tokyo": 0.60, "sunday": 0.60, "cppf": 0.60, "msv": 0.08, "ny": 0.80, "cpmc": 0.60, "orb": 0.60, "dd_pct": 30.4, "max_dd": 380.0, "pnl": 9400.0},
        {"Tier": "Tier 3: Aggressive", "tokyo": 0.75, "sunday": 0.75, "cppf": 0.75, "msv": 0.10, "ny": 1.00, "cpmc": 0.75, "orb": 0.75, "dd_pct": 39.2, "max_dd": 490.0, "pnl": 11800.0},
    ]

    rows = []
    for t in tiers:
        bank_payout = t["pnl"] * 0.80
        rows.append({
            "Risk Tier": t["Tier"],
            "TokyoH0": f"{t['tokyo']:.2f}",
            "SundayH22": f"{t['sunday']:.2f}",
            "CPPF_Z": f"{t['cppf']:.2f}",
            "MSV_Asian": f"{t['msv']:.2f}",
            "NY_H21": f"{t['ny']:.2f}",
            "CPMC_Z": f"{t['cpmc']:.2f}",
            "ORB_Ride": f"{t['orb']:.2f}",
            "Max Daily DD": f"${t['max_dd']:.0f} ({t['dd_pct']}%)",
            "Monthly Gross": f"+${t['pnl']:,.0f}",
            "Net Cash / Mo (80%)": f"+${bank_payout:,.0f}"
        })

    print(pd.DataFrame(rows).to_string(index=False))
    print("="*95)

if __name__ == "__main__":
    main()
