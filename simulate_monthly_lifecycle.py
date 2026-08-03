#!/usr/bin/env python3
"""30-Day Monthly Portfolio Life-Cycle Simulation ($6,000 FundedNext Account)."""
import numpy as np
import pandas as pd

np.random.seed(2026)

def simulate_monthly_cycle():
    starting_balance = 6000.0
    balance = starting_balance
    daily_logs = []
    
    # Engine parameters: (Name, lot, trades_per_month, win_rate, avg_win, avg_loss)
    engines = [
        ("TokyoH0", 0.15, 100, 0.953, 16.60, 12.30),
        ("SundayH22", 0.15, 20, 0.843, 18.40, 14.50),
        ("NYH21", 0.20, 40, 0.643, 15.20, 11.10),
        ("CPPF_Z", 0.15, 5, 0.852, 22.80, 15.20),
        ("MSV_Asian", 0.02, 4, 0.765, 19.10, 13.80),
        ("CPMC_Z", 0.15, 5, 0.615, 19.51, 11.15)
    ]
    
    # Generate randomized schedule of 174 trades over 30 days
    all_trades = []
    for eng_name, lot, n_tr, wr, avg_w, avg_l in engines:
        # Determine wins and losses
        n_wins = int(round(n_tr * wr))
        n_losses = n_tr - n_wins
        
        wins = np.random.normal(avg_w, avg_w * 0.15, n_wins)
        losses = -np.random.normal(avg_l, avg_l * 0.15, n_losses)
        pnls = np.concatenate([wins, losses])
        np.random.shuffle(pnls)
        
        # Assign random days (1 to 30)
        days = np.random.randint(1, 31, size=n_tr)
        for d, pnl in zip(days, pnls):
            all_trades.append({"day": d, "engine": eng_name, "pnl": pnl})
            
    df_trades = pd.DataFrame(all_trades).sort_values(["day", "engine"])
    
    # Day-by-day simulation
    cum_pnl = 0.0
    withdrawn_payout_1 = 0.0
    withdrawn_payout_2 = 0.0
    
    peak_balance = balance
    max_dd = 0.0
    
    print("="*85)
    print("30-DAY PORTFOLIO LIFE-CYCLE SIMULATION ($6,000 FUNDEDNEXT ACCOUNT)")
    print("="*85)
    print(f"Day  | Trades | Daily PnL  | Account Balance | Max Daily DD | Status / Event")
    print("-" * 85)
    
    for day in range(1, 31):
        t_day = df_trades[df_trades["day"] == day]
        n_t = len(t_day)
        if n_t > 0:
            daily_pnl = t_day["pnl"].sum()
            worst_draw = t_day["pnl"].clip(upper=0).sum() # Worst intra-day loss sum
        else:
            daily_pnl = 0.0
            worst_draw = 0.0
            
        balance += daily_pnl
        cum_pnl += daily_pnl
        
        if balance > peak_balance:
            peak_balance = balance
        dd = peak_balance - balance
        if dd > max_dd:
            max_dd = dd
            
        event_str = "Normal Trading"
        
        # Day 15: Bi-weekly Payout #1 (80% profit split withdrawn)
        if day == 15:
            profit_tier = balance - starting_balance
            if profit_tier > 0:
                withdrawn_payout_1 = profit_tier * 0.80 # 80% trader share
                prop_share_1 = profit_tier * 0.20
                balance -= profit_tier # Reset balance back to $6,000 buffer
                event_str = f"💰 PAYOUT #1: +${withdrawn_payout_1:.2f} Withdrawn to Bank! (Prop: ${prop_share_1:.2f})"
                
        # Day 30: Bi-weekly Payout #2 (80% profit split withdrawn)
        if day == 30:
            profit_tier = balance - starting_balance
            if profit_tier > 0:
                withdrawn_payout_2 = profit_tier * 0.80 # 80% trader share
                prop_share_2 = profit_tier * 0.20
                balance -= profit_tier # Reset balance back to $6,000 buffer
                event_str = f"💰 PAYOUT #2: +${withdrawn_payout_2:.2f} Withdrawn to Bank! (Prop: ${prop_share_2:.2f})"
                
        print(f"Day {day:<2} | {n_t:<6} | {daily_pnl:>+9.2f} | ${balance:>14.2f} | {worst_draw:>11.2f} | {event_str}")

    total_withdrawn = withdrawn_payout_1 + withdrawn_payout_2
    
    print("="*85)
    print("MONTHLY CASH-IN-POCKET FINANCIAL SUMMARY")
    print("="*85)
    print(f"  Starting Capital Baseline  : $6,000.00")
    print(f"  Total Trades Executed      : 174 Trades")
    print(f"  Total Gross Strategy PnL   : +${cum_pnl:.2f} (+{cum_pnl/6000.0*100:.1f}% Gross Account Return)")
    print(f"  Maximum Peak-to-Trough DD  : -${max_dd:.2f} (Only {max_dd/6000.0*100:.2f}% vs $600 Limit)")
    print("-" * 85)
    print(f"  Bi-Weekly Payout #1 (Day 15): +${withdrawn_payout_1:.2f}")
    print(f"  Bi-Weekly Payout #2 (Day 30): +${withdrawn_payout_2:.2f}")
    print("-" * 85)
    print(f"  💵 TOTAL CASH WITHDRAWN TO BANK (80% SHARE) : +${total_withdrawn:.2f} NET PROFIT IN POCKET!")
    print("="*85)

if __name__ == "__main__":
    simulate_monthly_cycle()
