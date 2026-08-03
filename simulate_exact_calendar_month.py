#!/usr/bin/env python3
"""Exact Calendar Month Simulation (22 Trading Days + 8 Weekend No-Trade Days) for 6 Engines."""
import numpy as np
import pandas as pd

np.random.seed(42)

def main():
    # 30-Day Month Structure (Days 1 to 30)
    # Weekends (No trade except Sunday 22:00 UTC): Days 6, 7, 13, 14, 20, 21, 27, 28
    weekend_days = [6, 7, 13, 14, 20, 21, 27, 28]
    sunday_open_days = [7, 14, 21, 28] # Sunday H22 fires on Sunday open
    trading_days = [d for d in range(1, 31) if d not in weekend_days]

    starting_balance = 6000.0
    balance = starting_balance

    # Engines definition: (Name, Lot, Daily_Frequency, Win_Rate, Avg_Win, Avg_Loss)
    engines_def = {
        "TokyoH0":   {"lot": 0.15, "freq": 5,  "wr": 0.953, "win": 16.60, "loss": 12.30, "fixed": True},
        "SundayH22": {"lot": 0.15, "freq": 5,  "wr": 0.843, "win": 18.40, "loss": 14.50, "sunday": True},
        "NYH21":     {"lot": 0.20, "freq": 2,  "wr": 0.643, "win": 15.20, "loss": 11.10, "fixed": True},
        "CPPF_Z":    {"lot": 0.15, "freq": 1,  "wr": 0.852, "win": 22.80, "loss": 15.20, "dynamic": True, "prob": 0.25},
        "MSV_Asian": {"lot": 0.02, "freq": 1,  "wr": 0.765, "win": 19.10, "loss": 13.80, "dynamic": True, "prob": 0.20},
        "CPMC_Z":    {"lot": 0.15, "freq": 1,  "wr": 0.615, "win": 19.51, "loss": 11.15, "dynamic": True, "prob": 0.25},
    }

    trades_history = []
    
    for day in range(1, 31):
        is_weekend = day in weekend_days
        is_sunday = day in sunday_open_days

        # 1. TokyoH0 (Fires every weekday)
        if not is_weekend:
            for _ in range(5):
                is_win = np.random.rand() < engines_def["TokyoH0"]["wr"]
                pnl = np.random.normal(16.60, 2.0) if is_win else -np.random.normal(12.30, 2.0)
                trades_history.append({"day": day, "engine": "TokyoH0", "pnl": pnl, "win": is_win})

        # 2. SundayH22 (Fires on Sunday 22:00 UTC)
        if is_sunday:
            for _ in range(5):
                is_win = np.random.rand() < engines_def["SundayH22"]["wr"]
                pnl = np.random.normal(18.40, 2.5) if is_win else -np.random.normal(14.50, 2.5)
                trades_history.append({"day": day, "engine": "SundayH22", "pnl": pnl, "win": is_win})

        # 3. NYH21 (Fires every weekday)
        if not is_weekend:
            for _ in range(2):
                is_win = np.random.rand() < engines_def["NYH21"]["wr"]
                pnl = np.random.normal(15.20, 2.0) if is_win else -np.random.normal(11.10, 2.0)
                trades_history.append({"day": day, "engine": "NYH21", "pnl": pnl, "win": is_win})

        # 4. CPPF_Z (Dynamic 24/7 shocks)
        if not is_weekend and np.random.rand() < engines_def["CPPF_Z"]["prob"]:
            is_win = np.random.rand() < engines_def["CPPF_Z"]["wr"]
            pnl = np.random.normal(22.80, 3.0) if is_win else -np.random.normal(15.20, 3.0)
            trades_history.append({"day": day, "engine": "CPPF_Z", "pnl": pnl, "win": is_win})

        # 5. MSV_Asian (Dynamic Asian dispersion)
        if not is_weekend and np.random.rand() < engines_def["MSV_Asian"]["prob"]:
            is_win = np.random.rand() < engines_def["MSV_Asian"]["wr"]
            pnl = np.random.normal(19.10, 2.5) if is_win else -np.random.normal(13.80, 2.5)
            trades_history.append({"day": day, "engine": "MSV_Asian", "pnl": pnl, "win": is_win})

        # 6. CPMC_Z (Dynamic 24/7 momentum shocks)
        if not is_weekend and np.random.rand() < engines_def["CPMC_Z"]["prob"]:
            is_win = np.random.rand() < engines_def["CPMC_Z"]["wr"]
            pnl = np.random.normal(19.51, 2.5) if is_win else -np.random.normal(11.15, 2.5)
            trades_history.append({"day": day, "engine": "CPMC_Z", "pnl": pnl, "win": is_win})

    df_tr = pd.DataFrame(trades_history)

    print("="*95)
    print("REALISTIC CALENDAR MONTH LIFE-CYCLE SIMULATION (22 TRADING DAYS + 8 WEEKEND NO-TRADE DAYS)")
    print("="*95)
    print(f"Day  | Status | Trades | Daily Net PnL | Account Balance | Engine Contributions")
    print("-" * 95)

    cum_pnl = 0.0
    payout_1 = 0.0
    payout_2 = 0.0

    for day in range(1, 31):
        is_wk = day in weekend_days
        t_d = df_tr[df_tr["day"] == day]
        n_t = len(t_d)
        d_pnl = t_d["pnl"].sum() if n_t > 0 else 0.0
        balance += d_pnl
        cum_pnl += d_pnl

        status_str = "TRADING" if not is_wk else ("SUNDAY OPEN" if day in sunday_open_days else "NO TRADES (SAT)")
        
        # Engine details
        eng_counts = t_d["engine"].value_counts().to_dict() if n_t > 0 else {}
        eng_str = ", ".join([f"{k}:{v}" for k,v in eng_counts.items()]) if eng_counts else "None (Market Closed)"

        payout_event = ""
        if day == 15:
            prof = balance - starting_balance
            if prof > 0:
                payout_1 = prof * 0.80
                balance -= prof
                payout_event = f" 💰 PAYOUT #1: +${payout_1:.2f} WITHDRAWN!"
        elif day == 30:
            prof = balance - starting_balance
            if prof > 0:
                payout_2 = prof * 0.80
                balance -= prof
                payout_event = f" 💰 PAYOUT #2: +${payout_2:.2f} WITHDRAWN!"

        print(f"Day {day:<2} | {status_str:<11} | {n_t:<6} | ${d_pnl:>+8.2f} | ${balance:>14.2f} | {eng_str}{payout_event}")

    tot_payout = payout_1 + payout_2

    print("\n" + "="*95)
    print("ENGINE-BY-ENGINE SEPARATE MONTHLY BREAKDOWN MATRIX")
    print("="*95)
    
    engine_summary = []
    for eng in ["TokyoH0", "SundayH22", "NYH21", "CPPF_Z", "MSV_Asian", "CPMC_Z"]:
        df_e = df_tr[df_tr["engine"] == eng]
        t_cnt = len(df_e)
        w_cnt = df_e["win"].sum()
        wr_pct = (w_cnt / t_cnt * 100.0) if t_cnt > 0 else 0.0
        net_e = df_e["pnl"].sum()
        bank_share = net_e * 0.80 if net_e > 0 else 0.0
        engine_summary.append({
            "Engine Name": eng,
            "Lot Size": engines_def[eng]["lot"],
            "Monthly Trades": t_cnt,
            "Wins": w_cnt,
            "Losses": t_cnt - w_cnt,
            "Win Rate": f"{wr_pct:.1f}%",
            "Gross PnL": f"+${net_e:.2f}" if net_e > 0 else f"-${abs(net_e):.2f}",
            "Bank Net Share (80%)": f"+${bank_share:.2f}"
        })

    df_eng = pd.DataFrame(engine_summary)
    print(df_eng.to_string(index=False))

    print("="*95)
    print("FINAL MONTHLY CASH-IN-POCKET SUMMARY")
    print("="*95)
    print(f"  Starting Balance         : $6,000.00")
    print(f"  Active Trading Days      : 22 Days")
    print(f"  Weekend No-Trade Days    : 8 Days")
    print(f"  Total Trades Executed    : {len(df_tr)} Trades")
    print(f"  Gross Portfolio Return   : +${cum_pnl:.2f} (+{cum_pnl/6000.0*100:.1f}%)")
    print("-" * 95)
    print(f"  Bi-Weekly Payout #1 (Day 15) : +${payout_1:.2f} Cash Withdrawn")
    print(f"  Bi-Weekly Payout #2 (Day 30) : +${payout_2:.2f} Cash Withdrawn")
    print("-" * 95)
    print(f"  💵 TOTAL CASH WITHDRAWN TO BANK ACCOUNT (80% SHARE) : +${tot_payout:.2f} IN POCKET!")
    print("="*95)

if __name__ == "__main__":
    main()
