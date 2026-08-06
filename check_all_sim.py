import sys
sys.path.insert(0, 'proxima_command_center')
from mt5_history_loader import get_side_by_side_trade_comparison
from rolling_backtest_engine import RollingBacktestEngine

engine = RollingBacktestEngine()
res = engine.run_cycle()
for t in res.get('python_trades', []):
    print(f"{t['entry_time']} | #{t['ticket']} | {t['symbol']} {t['type']} | PnL=${t.get('net_pnl', t.get('pnl',0))} | {t['strategy']}")
