import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent / "proxima_command_center"))
from rolling_backtest_engine import RollingBacktestEngine

def main():
    print("=" * 115)
    print("PROXIMA X — ALL v107 STRATEGIES: TODAY'S (2026-08-04 UTC) DETETERMINISTIC PYTHON SIM TRADES AUDIT")
    print("=" * 115)

    eng = RollingBacktestEngine()
    now_utc = datetime.now(timezone.utc)
    now_ts  = pd.Timestamp(now_utc.replace(tzinfo=None))
    
    trades = eng.compute_deterministic_trades("2026-08-04", max_time_utc=now_ts)

    if not trades:
        print("No Python sim trades generated for today so far.")
        return

    # Group by strategy
    by_strat = {}
    for t in trades:
        s = t['strategy']
        if s not in by_strat:
            by_strat[s] = []
        by_strat[s].append(t)

    total_wins = [t for t in trades if t['is_win']]
    total_losses = [t for t in trades if not t['is_win']]
    total_pnl = sum(t['sim_pnl'] for t in trades)
    overall_wr = len(total_wins) / len(trades) * 100.0

    print(f"\n🏆 TODAY'S OVERALL PORTFOLIO METRICS (2026-08-04 UTC):")
    print(f"  • Total Trades Evaluated : {len(trades)}")
    print(f"  • Winning Trades         : {len(total_wins)} Wins")
    print(f"  • Losing Trades          : {len(total_losses)} Losses")
    print(f"  • Portfolio Win Rate     : {overall_wr:.1f}% 🟢")
    print(f"  • Net Realized PnL       : +${total_pnl:,.2f} 🚀")

    print("\n" + "=" * 115)
    print("📊 PER-STRATEGY BREAKDOWN TODAY:")
    print("=" * 115)
    
    for strat, str_trades in by_strat.items():
        w = [t for t in str_trades if t['is_win']]
        l = [t for t in str_trades if not t['is_win']]
        pnl = sum(t['sim_pnl'] for t in str_trades)
        wr = len(w) / len(str_trades) * 100.0 if len(str_trades) > 0 else 0.0
        print(f"  • {strat:<24} | Trades: {len(str_trades):>2} | Wins: {len(w):>2} | Losses: {len(l):>2} | WR: {wr:>5.1f}% | Net PnL: ${pnl:>8.2f}")

    print("\n" + "=" * 115)
    print("📋 EXHAUSTIVE LOG OF ALL TODAY'S v107 TRADES:")
    print("=" * 115)

    for i, t in enumerate(trades, 1):
        pnl_str = f"+${t['sim_pnl']:.2f}" if t['sim_pnl'] >= 0 else f"-${abs(t['sim_pnl']):.2f}"
        res = "🟢 WIN" if t['is_win'] else "🔴 LOSS"
        print(f"  #{i:02d} | {t['iso_timestamp']} | {t['strategy']:<22} | {t['pair']} {t['side']:<4} | Lot: {t['lot']}L | Pips: {t['pips']:>6.1f}p | PnL: {pnl_str:>10} | {res}")

    print("=" * 115)

if __name__ == "__main__":
    main()
