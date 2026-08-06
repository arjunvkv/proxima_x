import sys
sys.path.insert(0, 'proxima_command_center')
from mt5_history_loader import get_side_by_side_trade_comparison
from rolling_backtest_engine import RollingBacktestEngine
from datetime import datetime

print("=== LIVE MT5 TRADES AT 13:00 UTC ON 2026-08-03 ===")
trades = get_side_by_side_trade_comparison()
live_13 = [t for t in trades if '2026-08-03 13:00' in t.get('entry_time','')]
for t in live_13:
    print(f"LIVE: #{t['ticket']} | {t['symbol']} {t['type']} {t['lot']}L | PnL=${t['net_pnl']} | {t['strategy']}")

print("\n=== PYTHON SIM ENGINE GENERATION (Single Run) ===")
engine = RollingBacktestEngine()
res = engine.run_cycle()
sim_13 = [t for t in res.get('python_trades', []) if '13:00' in t.get('entry_time','')]
for t in sim_13:
    print(f"SIM: #{t['ticket']} | {t['symbol']} {t['type']} {t['lot']}L | PnL=${t['net_pnl']} | {t['strategy']}")
