#!/usr/bin/env python3
"""Fast CZR 5-Broker Validation Runner."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd
from proxima_honest_backtest.strategies.czr.strategy import CZRStrategy
from proxima_honest_backtest.strategies.multi_pair_engine import MultiPairBacktestEngine
from proxima_honest_backtest.execution.execution_simulator import ExecutionSimulator
from proxima_honest_backtest.strategies.czr.sweep import load_and_align, BROKERS

def main():
    t0 = time.time()
    print("Loading & pre-aligning 18-pair M5 dataset...")
    raw, pre_align = load_and_align()
    print(f"  Loaded {len(raw)} pairs, {len(pre_align):,} bars ({time.time()-t0:.1f}s)")

    configs = [
        (4.0, 9),   # z>=4.0 hold=45m (9 bars)
        (4.0, 12),  # z>=4.0 hold=60m (12 bars)
        (3.0, 9),   # z>=3.0 hold=45m (9 bars)
        (3.0, 12),  # z>=3.0 hold=60m (12 bars)
    ]

    print("\n" + "="*80)
    print("CZR STRATEGY — 5-BROKER TRANSACTION COST & SURVIVAL AUDIT")
    print("="*80)

    for z_thresh, hold_bars in configs:
        print(f"\n--- Testing Config: z>={z_thresh} | hold={hold_bars*5}min ({hold_bars} M5 bars) ---")
        rows = []
        for b in BROKERS:
            s = CZRStrategy({"z_thresh": z_thresh, "hold_bars": hold_bars, "long_only": True})
            s.set_precomputed_data(raw)
            e = MultiPairBacktestEngine(s, ExecutionSimulator(b))
            r = e.run(raw, pre_aligned=pre_align)
            survives = "PASS" if r.total_pnl > 0 and r.profit_factor > 1.0 else "FAIL"
            avg_win = r.total_pnl / r.n_trades if r.n_trades > 0 else 0.0
            rows.append({
                "Broker": b.upper(),
                "Trades": r.n_trades,
                "Win Rate": f"{r.win_rate*100:.1f}%",
                "Net PnL": f"+${r.total_pnl:.2f}" if r.total_pnl > 0 else f"-${abs(r.total_pnl):.2f}",
                "PF": round(r.profit_factor, 2),
                "Avg$/Trade": f"+${avg_win:.2f}",
                "Max DD%": f"{r.max_drawdown_pct:.2f}%",
                "Status": survives
            })
        print(pd.DataFrame(rows).to_string(index=False))

if __name__ == "__main__":
    main()
