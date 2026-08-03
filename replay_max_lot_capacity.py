#!/usr/bin/env python3
"""Replay Today's Tokyo H0 Session at Max Lot Capacity for $6k Account."""
import MetaTrader5 as mt5
import pandas as pd

PAIRS = ["EURGBP", "USDCHF", "USDJPY", "EURAUD", "EURNZD"]

def run_session_at_lot(lot_per_pair):
    trade_results = []
    
    for pair in PAIRS:
        rates = mt5.copy_rates_from_pos(pair, mt5.TIMEFRAME_M5, 0, 300)
        if rates is None or len(rates) == 0:
            continue

        df = pd.DataFrame(rates)
        df["time_dt"] = pd.to_datetime(df["time"], unit="s")
        df["hour"] = df["time_dt"].dt.hour
        df["min"] = df["time_dt"].dt.minute

        bars_session = df[(df["hour"] == 3) | (df["hour"] == 0)].copy()
        if bars_session.empty:
            bars_session = df.tail(12)

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

        # Lot PnL calculation
        contract_val = 100000.0 * lot_per_pair
        gross_pnl = (exit_price - entry_price) / entry_price * contract_val
        comm = 3.00 * lot_per_pair # $3.00/lot commission
        net_pnl = gross_pnl - comm
        pips = (exit_price - entry_price) / (0.01 if "JPY" in pair else 0.0001)

        trade_results.append({
            "Pair": pair,
            "Lot": lot_per_pair,
            "Pip_Change": f"{pips:+.1f} pips",
            "Gross_PnL": f"+${gross_pnl:.2f}" if gross_pnl > 0 else f"-${abs(gross_pnl):.2f}",
            "Commission": f"-${comm:.2f}",
            "Net_PnL": f"+${net_pnl:.2f}" if net_pnl > 0 else f"-${abs(net_pnl):.2f}",
            "Result": "WIN" if net_pnl > 0 else "LOSS"
        })

    return trade_results

def main():
    if not mt5.initialize():
        print("Failed to initialize MT5")
        return

    print("="*85)
    print("TOKYO H0 TODAY'S SESSION — LOT CAPACITY & YIELD ANALYSIS ($6,000 ACCOUNT)")
    print("="*85)

    lot_levels = [
        (0.15, "Standard Baseline (0.15 Lot/pair = 0.75 Lot Total)"),
        (0.25, "Optimal High-Yield (0.25 Lot/pair = 1.25 Lot Total)"),
        (0.35, "Maximum Aggressive (0.35 Lot/pair = 1.75 Lot Total)"),
    ]

    for lot, label in lot_levels:
        res = run_session_at_lot(lot)
        net_tot = sum(float(r["Net_PnL"].replace("+$","").replace("-$","-")) for r in res)
        comm_tot = sum(float(r["Commission"].replace("-$","")) for r in res)
        wins = sum(1 for r in res if r["Result"] == "WIN")
        
        # Risk check
        max_possible_loss = lot * 15.0 * 10.0 * 5 # 15-pip stop on 5 pairs
        pct_daily_limit = (max_possible_loss / 300.0) * 100.0

        print(f"\n--- {label} ---")
        print(f"  Total Trades        : 5 Concurrent Positions")
        print(f"  Win Rate            : {wins}/5 ({wins/5*100:.1f}%)")
        print(f"  Total Commission    : -${comm_tot:.2f}")
        print(f"  Total Session Net   : +${net_tot:.2f} ({net_tot/6000.0*100:+.2f}% Account Growth in 60m)")
        print(f"  Max Stop Risk (15p) : -${max_possible_loss:.2f} ({pct_daily_limit:.1f}% of $300 Daily Limit)")

    mt5.shutdown()

if __name__ == "__main__":
    main()
