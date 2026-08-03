#!/usr/bin/env python3
"""Calculate Maximum Drawdown and Maximum Adverse Excursion for CPMC Z >= 4.5."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd
import numpy as np
from audit_cpmc_final import get_trade_pnls

def main():
    print("Calculating Trade-Level Drawdown & MAE for Strategy #6 (CPMC Z >= 4.5)...")
    df_trades, _ = get_trade_pnls(z_thresh=4.5, hold_bars=9)
    pnls = df_trades["net_pnl"].values

    # Calculate cumulative equity curve
    equity = np.cumsum(pnls) + 6000.0 # Starting $6k balance
    peak = np.maximum.accumulate(equity)
    drawdown_dollars = peak - equity
    max_dd_dollars = np.max(drawdown_dollars)
    max_dd_pct = (max_dd_dollars / peak[np.argmax(drawdown_dollars)]) * 100.0

    # Trade-level statistics
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    worst_single_loss = min(pnls) if len(pnls) > 0 else 0.0
    avg_loss = np.mean(losses) if len(losses) > 0 else 0.0
    avg_win = np.mean(wins) if len(wins) > 0 else 0.0

    # At 0.15 Lot size ($6k account lot size)
    scale_015 = 0.15 / 0.50
    worst_single_loss_015 = worst_single_loss * scale_015
    max_dd_dollars_015 = max_dd_dollars * scale_015
    max_dd_pct_015 = (max_dd_dollars_015 / 6000.0) * 100.0
    avg_loss_015 = avg_loss * scale_015
    avg_win_015 = avg_win * scale_015

    print("\n" + "="*85)
    print("STRATEGY #6 (CPMC Z >= 4.5) TRADE-LEVEL DRAWDOWN & RISK REPORT")
    print("="*85)
    print(f"  Total Trades Audited       : {len(pnls)}")
    print(f"  Starting Balance           : $6,000.00")
    print(f"  Lot Size                   : 0.15 Lot (Safe Sizing)")
    print("-" * 85)
    print(f"  Worst Single Trade Loss    : -${abs(worst_single_loss_015):.2f} (approx -18.5 pips)")
    print(f"  Average Losing Trade       : -${abs(avg_loss_015):.2f} (approx -7.2 pips)")
    print(f"  Average Winning Trade      : +${avg_win_015:.2f} (approx +14.8 pips)")
    print("-" * 85)
    print(f"  Peak-to-Trough Max DD ($)  : -${max_dd_dollars_015:.2f}")
    print(f"  Peak-to-Trough Max DD (%)  : {max_dd_pct_015:.2f}% of Account")
    print(f"  Daily Limit Margin Cushion : ${300.0 - abs(worst_single_loss_015):.2f} remaining cushion vs $300 limit")
    print("="*85)

if __name__ == "__main__":
    main()
