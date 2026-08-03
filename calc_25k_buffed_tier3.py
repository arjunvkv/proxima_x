#!/usr/bin/env python3
"""Calculate Tier 3+ Max Headroom Scaling for $25,000 Account."""
import pandas as pd

def main():
    account_size = 25000.0
    daily_limit = 1250.0

    print("="*95)
    print("PROXIMA MAX HEADROOM BUFFED LOT MATRIX: $25,000 FUNDEDNEXT ACCOUNT")
    print("="*95)

    tiers = [
        {"Tier": "Tier 2: Recommended Max Profit", "tokyo": 0.60, "sunday": 0.60, "cppf": 0.60, "msv": 0.08, "ny": 0.80, "cpmc": 0.60, "orb": 0.60, "max_dd": 380.0, "dd_pct": 30.4, "pnl": 9400.0},
        {"Tier": "Tier 3: Aggressive Buffed", "tokyo": 0.80, "sunday": 0.80, "cppf": 0.80, "msv": 0.10, "ny": 1.00, "cpmc": 0.80, "orb": 0.80, "max_dd": 500.0, "dd_pct": 40.0, "pnl": 12500.0},
        {"Tier": "Tier 3+: MAX HEADROOM CHAMPION", "tokyo": 1.00, "sunday": 1.00, "cppf": 1.00, "msv": 0.12, "ny": 1.25, "cpmc": 1.00, "orb": 1.00, "max_dd": 620.0, "dd_pct": 49.6, "pnl": 15600.0},
    ]

    rows = []
    for t in tiers:
        payout = t["pnl"] * 0.80
        buffer_val = daily_limit - t["max_dd"]
        rows.append({
            "Tier Level": t["Tier"],
            "1.0 Lot Base Pairs": f"{t['tokyo']:.2f} Lot",
            "NY_H21 Lot": f"{t['ny']:.2f} Lot",
            "MSV Lot": f"{t['msv']:.2f} Lot/pr",
            "Max Daily DD": f"${t['max_dd']:.0f} ({t['dd_pct']}%)",
            "Unused Daily Buffer": f"${buffer_val:.0f} ({100-t['dd_pct']:.1f}%)",
            "Monthly Gross": f"+${t['pnl']:,.0f}",
            "Net Cash Payout (80%)": f"+${payout:,.0f}"
        })

    print(pd.DataFrame(rows).to_string(index=False))
    print("="*95)

if __name__ == "__main__":
    main()
