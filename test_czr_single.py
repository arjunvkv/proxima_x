#!/usr/bin/env python3
"""Single Config CZR Test."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd
from proxima_honest_backtest.strategies.czr.strategy import CZRStrategy
from proxima_honest_backtest.strategies.multi_pair_engine import MultiPairBacktestEngine
from proxima_honest_backtest.execution.execution_simulator import ExecutionSimulator
from proxima_honest_backtest.strategies.czr.sweep import load_and_align

t0 = time.time()
raw, pre_align = load_and_align()
s = CZRStrategy({"z_thresh": 4.0, "hold_bars": 9, "long_only": True})
s.set_precomputed_data(raw)
e = MultiPairBacktestEngine(s, ExecutionSimulator("fundednext"))
r = e.run(raw, pre_aligned=pre_align)

print("="*60)
print(f"CZR STRATEGY (z>=4.0, hold=45m) FUNDEDNEXT RESULTS ({time.time()-t0:.1f}s)")
print("="*60)
print(f"Total Trades : {r.n_trades}")
print(f"Win Rate     : {r.win_rate*100:.1f}%")
print(f"Net PnL      : +${r.total_pnl:.2f}")
print(f"Profit Factor: {r.profit_factor:.2f}")
print(f"Max DD %     : {r.max_drawdown_pct:.2f}%")
print(f"Sharpe Ratio : {r.sharpe:.2f}")
print(f"Avg$/Trade   : +${r.total_pnl / r.n_trades:.2f}")
print("="*60)
