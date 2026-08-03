#!/usr/bin/env python3
"""Exact Spec-Compliant NY H21 Strategy Audit across 5 Brokers."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd
from proxima_honest_backtest.strategies.ny_h21.strategy import NYH21Strategy
from proxima_honest_backtest.strategies.multi_pair_engine import MultiPairBacktestEngine
from proxima_honest_backtest.execution.execution_simulator import ExecutionSimulator
from proxima_honest_backtest.strategies.sunday_h22.sweep import load_and_align, BROKERS

def main():
    t0 = time.time()
    print("Loading M5 dataset for Spec-Compliant NY H21 Strategy Audit...")
    raw, pre_align = load_and_align()
    print(f"  Loaded {len(raw)} pairs, {len(pre_align):,} bars ({time.time()-t0:.1f}s)")

    print("\n" + "="*85)
    print("OFFICIAL NY H21 STRATEGY ENGINE — 5-BROKER TRANSACTION COST AUDIT")
    print("="*85)

    broker_rows = []
    for b in BROKERS:
        s = NYH21Strategy({"lb": 12, "hold_bars": 12, "top_n": 5, "trade_pairs": ["EURJPY", "GBPJPY"]})
        e = MultiPairBacktestEngine(s, ExecutionSimulator(b))
        r = e.run(raw, pre_aligned=pre_align)
        survives = "PASS" if r.total_pnl > 0 and r.profit_factor > 1.0 else "FAIL"
        avg_win = r.total_pnl / r.n_trades if r.n_trades > 0 else 0.0
        broker_rows.append({
            "Broker": b.upper(),
            "Trades": r.n_trades,
            "Win Rate": f"{r.win_rate*100:.1f}%",
            "Net PnL": f"+${r.total_pnl:.2f}" if r.total_pnl > 0 else f"-${abs(r.total_pnl):.2f}",
            "PF": round(r.profit_factor, 2),
            "Avg$/Trade": f"+${avg_win:.2f}",
            "Max DD%": f"{r.max_drawdown_pct:.2f}%",
            "Status": survives
        })

    print(pd.DataFrame(broker_rows).to_string(index=False))

if __name__ == "__main__":
    main()
