"""
Local Full Today Backtest (Aug 4, 2026) — Honest Backtest Engine
Evaluates all portfolio strategies over today's M5 data from 00:00 UTC to current time.
"""

import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.providers.mt5_provider import MT5Provider
from proxima_honest_backtest.strategies.tokyo_h0.strategy import TokyoH0Strategy
from proxima_honest_backtest.strategies.ny_h21.strategy import NYH21Strategy
from proxima_honest_backtest.strategies.multi_pair_engine import MultiPairBacktestEngine
from proxima_honest_backtest.execution.execution_simulator import ExecutionSimulator

ALL_PAIRS = [
    "EURUSD","USDJPY","GBPUSD","AUDUSD","EURJPY",
    "GBPJPY","EURAUD","EURNZD","GBPAUD","GBPNZD",
    "GBPCAD","AUDNZD","USDCAD","NZDUSD","EURGBP",
    "EURCHF","USDCHF","AUDJPY",
]

def main():
    print("=================================================================================")
    print("LOCAL HONEST BACKTEST — TODAY (AUG 4, 2026: 00:00 UTC — CURRENT TIME)")
    print("=================================================================================")

    provider = MT5Provider()
    raw = {}
    print("Loading August 2026 M5 data for all 18 pairs...")

    for p in ALL_PAIRS:
        df = provider.load_rates(p, 2026, 8, "m5")
        if df.empty:
            # Try loading July + August if Aug is partial
            df_july = provider.load_rates(p, 2026, 7, "m5")
            df = pd.concat([df_july, df], ignore_index=True) if not df_july.empty else df
        if not df.empty:
            df.sort_values("time", inplace=True)
            df.reset_index(drop=True, inplace=True)
            raw[p] = df

    print(f"Loaded {len(raw)} pairs.")

    # Check date range
    for pair, df in raw.items():
        if not df.empty:
            df_today = df[df["time"] >= "2026-08-04 00:00:00"]
            print(f"  {pair:7s}: {len(df_today):3d} M5 bars today (from {df['time'].iloc[0]} to {df['time'].iloc[-1]})")
            break

    # Pre-align data
    pieces = []
    for pair, df in raw.items():
        sub = df.set_index("time")[["close","open","high","low","tick_volume","spread"]]
        sub.columns = [pair, f"{pair}_open", f"{pair}_high", f"{pair}_low", f"{pair}_volume", f"{pair}_spread"]
        pieces.append(sub)

    aligned = pd.concat(pieces, axis=1, sort=True)
    aligned.sort_index(inplace=True)
    aligned.ffill(inplace=True); aligned.bfill(inplace=True)
    aligned.reset_index(inplace=True); aligned.rename(columns={"index": "time"}, inplace=True)

    aligned_dict = aligned.to_dict("records")
    print(f"Total aligned bars: {len(aligned_dict):,}")

    # 1. Evaluate Tokyo H0 Today
    print("\n---------------------------------------------------------------------------------")
    print("1. TOKYO H0 STRATEGY (lb=6, hold=12, top_n=3)")
    print("---------------------------------------------------------------------------------")
    s_tokyo = TokyoH0Strategy({"lookback_bars": 6, "hold_bars": 12, "top_n": 3})
    e_tokyo = MultiPairBacktestEngine(s_tokyo, ExecutionSimulator("ftmo"))
    res_tokyo = e_tokyo.run(raw, pre_aligned=aligned_dict)

    tokyo_trades_today = [t for t in res_tokyo.trades if str(t.entry_time).startswith("2026-08-04")]
    print(f"Tokyo H0 Trades Today (Aug 4): {len(tokyo_trades_today)}")
    for t in tokyo_trades_today:
        print(f"  -> [{t.entry_time}] {t.symbol:7s} {t.side:4s} {t.volume:.2f}L | Entry: {t.entry_price:.5f} -> Exit: {t.exit_price:.5f} | PnL: ${t.pnl:+7.2f} (Comm: ${t.commission:.2f})")

    # 2. Evaluate NY H21 Today
    print("\n---------------------------------------------------------------------------------")
    print("2. NY H21 STRATEGY (lb=12, hold=12, top_n=5, JPY pairs)")
    print("---------------------------------------------------------------------------------")
    s_ny = NYH21Strategy({"lookback_bars": 12, "hold_bars": 12, "top_n": 5, "trade_pairs": ["EURJPY", "GBPJPY"]})
    e_ny = MultiPairBacktestEngine(s_ny, ExecutionSimulator("ftmo"))
    res_ny = e_ny.run(raw, pre_aligned=aligned_dict)

    ny_trades_today = [t for t in res_ny.trades if str(t.entry_time).startswith("2026-08-04")]
    print(f"NY H21 Trades Today (Aug 4): {len(ny_trades_today)}")
    for t in ny_trades_today:
        print(f"  -> [{t.entry_time}] {t.symbol:7s} {t.side:4s} {t.volume:.2f}L | Entry: {t.entry_price:.5f} -> Exit: {t.exit_price:.5f} | PnL: ${t.pnl:+7.2f} (Comm: ${t.commission:.2f})")

    # Summary
    all_today_trades = tokyo_trades_today + ny_trades_today
    total_pnl = sum(t.pnl for t in all_today_trades)
    total_comm = sum(t.commission for t in all_today_trades)
    wins = sum(1 for t in all_today_trades if t.pnl > 0)

    print("\n=================================================================================")
    print("FULL TODAY PORTFOLIO BACKTEST SUMMARY (AUG 4, 2026)")
    print("=================================================================================")
    print(f"  Total Trades Today:     {len(all_today_trades)}")
    print(f"  Total Wins Today:       {wins}")
    print(f"  Today Win Rate:         {wins/len(all_today_trades)*100 if all_today_trades else 0.0:.1f}%")
    print(f"  Total Gross PnL Today:  ${total_pnl + total_comm:+8.2f}")
    print(f"  Total Comm Today:       ${total_comm:8.2f}")
    print(f"  Total Net PnL Today:    ${total_pnl:+8.2f}")
    print("=================================================================================")

if __name__ == "__main__":
    main()
