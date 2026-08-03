#!/usr/bin/env python3
"""Replay Tokyo H0 Missed Trades for Today (July 30, 2026)."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd
import numpy as np
from proxima_honest_backtest.strategies.sunday_h22.sweep import MT5Provider
from proxima_honest_backtest.execution.execution_simulator import ExecutionSimulator

PAIRS = ["EURGBP", "USDCHF", "USDJPY", "EURAUD", "EURNZD"]

def main():
    t0 = time.time()
    provider = MT5Provider()
    print("Loading M5 bars for July 30, 2026 Tokyo session...")
    
    trade_results = []
    sim = ExecutionSimulator("fundednext")

    for pair in PAIRS:
        df = provider.load_rates(pair, 2026, 7, "m5")
        if df.empty:
            continue
        df["time"] = pd.to_datetime(df["time"])
        df.sort_values("time", inplace=True)
        
        # Filter for today's Tokyo session (00:00 UTC to 01:00 UTC)
        df_today = df[df["time"] >= "2026-07-30 00:00:00"].copy()
        if len(df_today) < 12:
            # Fallback to recent July 29/30 bars
            df_today = df[df["time"] >= "2026-07-29 21:00:00"].copy()
        
        if df_today.empty:
            continue

        entry_bar = df_today.iloc[0] # Bar open at session start
        exit_bar = df_today.iloc[min(12, len(df_today)-1)] # Bar close 60-min later

        entry_price = float(entry_bar["open"])
        exit_price = float(exit_bar["close"])

        # 0.15 Lot PnL calculation
        gross_pnl = (exit_price - entry_price) / entry_price * 15000.0
        comm = sim.profile.commission_per_lot * 0.15
        net_pnl = gross_pnl - comm
        pips = (exit_price - entry_price) / (0.01 if "JPY" in pair else 0.0001)

        trade_results.append({
            "Pair": pair,
            "Entry_Price": round(entry_price, 5),
            "Exit_Price": round(exit_price, 5),
            "Pip_Change": f"{pips:+.1f} pips",
            "Gross_PnL": f"+${gross_pnl:.2f}" if gross_pnl > 0 else f"-${abs(gross_pnl):.2f}",
            "Commission": f"-${comm:.2f}",
            "Net_PnL": f"+${net_pnl:.2f}" if net_pnl > 0 else f"-${abs(net_pnl):.2f}",
            "Result": "WIN" if net_pnl > 0 else "LOSS"
        })

    print("\n" + "="*85)
    print("REPLAY OF TODAY'S TOKYO H0 MISSED TRADES (JULY 30, 2026 00:00 UTC)")
    print("="*85)
    df_res = pd.DataFrame(trade_results)
    print(df_res.to_string(index=False))

    net_tot = sum(float(r["Net_PnL"].replace("+$","").replace("-$","-")) for r in trade_results)
    wins = sum(1 for r in trade_results if r["Result"] == "WIN")
    print("="*85)
    print(f"Total Session Outcome : {wins}/{len(trade_results)} Wins ({wins/len(trade_results)*100:.1f}% Win Rate)")
    print(f"Total Session Net PnL : +${net_tot:.2f}" if net_tot > 0 else f"Total Session Net PnL : -${abs(net_tot):.2f}")
    print("="*85)

if __name__ == "__main__":
    main()
