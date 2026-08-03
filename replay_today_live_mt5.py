#!/usr/bin/env python3
"""Replay Today's Tokyo H0 Missed Trades using Live MT5 Rates."""
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timezone

PAIRS = ["EURGBP", "USDCHF", "USDJPY", "EURAUD", "EURNZD"]

def main():
    if not mt5.initialize():
        print("Failed to initialize MT5")
        return

    print("="*85)
    print("REPLAY OF TODAY'S TOKYO H0 MISSED TRADES (JULY 30, 2026 00:00 UTC / 05:35 AM IST)")
    print("="*85)

    trade_results = []
    
    for pair in PAIRS:
        rates = mt5.copy_rates_from_pos(pair, mt5.TIMEFRAME_M5, 0, 300)
        if rates is None or len(rates) == 0:
            continue

        df = pd.DataFrame(rates)
        df["time_dt"] = pd.to_datetime(df["time"], unit="s")
        
        # Match today's 00:00 UTC bar (or latest 00:00 bar)
        df["hour"] = df["time_dt"].dt.hour
        df["min"] = df["time_dt"].dt.minute

        # Find 00:00 to 00:05 UTC bar
        bars_session = df[(df["hour"] == 3) | (df["hour"] == 0)].copy() # MT5 Server 03:00 / 00:00
        if bars_session.empty:
            bars_session = df.tail(12)

        # Get session entry bar and 60m exit bar (12 M5 bars later)
        idx_entry = 0
        for i, row in bars_session.iterrows():
            if row["min"] in [0, 5]:
                idx_entry = i
                break
        
        idx_exit = min(idx_entry + 12, len(df) - 1)

        entry_bar = df.loc[idx_entry]
        exit_bar = df.loc[idx_exit]

        entry_price = float(entry_bar["open"])
        exit_price = float(exit_bar["close"])

        # 0.15 Lot PnL
        gross_pnl = (exit_price - entry_price) / entry_price * 15000.0
        comm = 0.45 # $3.00/lot commission on 0.15 lot
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

    df_res = pd.DataFrame(trade_results)
    print(df_res.to_string(index=False))

    net_tot = sum(float(r["Net_PnL"].replace("+$","").replace("-$","-")) for r in trade_results)
    wins = sum(1 for r in trade_results if r["Result"] == "WIN")
    print("="*85)
    print(f"Total Session Outcome : {wins}/{len(trade_results)} Wins ({wins/len(trade_results)*100:.1f}% Win Rate)")
    print(f"Total Session Net PnL : +${net_tot:.2f}" if net_tot > 0 else f"Total Session Net PnL : -${abs(net_tot):.2f}")
    print("="*85)

    mt5.shutdown()

if __name__ == "__main__":
    main()
