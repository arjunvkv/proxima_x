import sys
sys.path.insert(0, 'proxima_command_center')
from rolling_backtest_engine import RollingBacktestEngine

engine = RollingBacktestEngine()
summary = engine.get_yesterday_full_day_summary()

print("=== LIVE TRADES YESTERDAY (2026-08-03) ===")
for t in summary.get('trades', []):
    if t.get('is_live'):
        print(f"LIVE: #{t['ticket']} | {t['display_time']} UTC | {t['strategy']} | {t['pair']} {t['side']} | PnL=${t['live_close_pnl']}")

print("\n=== PYTHON SIM SYNTHETIC BENCHMARK TRADES AROUND 13:00 UTC ===")
for t in summary.get('trades', []):
    if not t.get('is_live') and '13:00' in t.get('display_time', ''):
        print(f"SIM:  #{t['ticket']} | {t['display_time']} UTC | {t['strategy']} | {t['pair']} {t['side']} | sim_pnl=${t['sim_pnl']} | is_win={t['is_win']}")
