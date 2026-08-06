import sys, json
sys.path.insert(0, 'proxima_command_center')
from rolling_backtest_engine import RollingBacktestEngine

eng = RollingBacktestEngine()
summary = eng.get_yesterday_full_day_summary()

print("=== YESTERDAY PYTHON METRICS ===")
print(json.dumps(summary['python_metrics'], indent=2))

print("\n=== YESTERDAY DETERMINISTIC TRADES BREAKDOWN BY STRATEGY ===")
py_trades = [t for t in summary['trades'] if not t['is_live']]

strat_summary = {}
for t in py_trades:
    st = t['strategy']
    if st not in strat_summary:
        strat_summary[st] = {'trades': 0, 'wins': 0, 'losses': 0, 'pnl': 0.0, 'pips': 0.0}
    strat_summary[st]['trades'] += 1
    if t['is_win']:
        strat_summary[st]['wins'] += 1
    else:
        strat_summary[st]['losses'] += 1
    strat_summary[st]['pnl'] += t['sim_pnl']
    strat_summary[st]['pips'] += t['pips']

for st, s in strat_summary.items():
    wr = round(s['wins'] / s['trades'] * 100.0, 1) if s['trades'] else 0
    print(f"Strategy: {st:25s} | Trades: {s['trades']:2d} | Wins: {s['wins']:2d} | Losses: {s['losses']:2d} | WR: {wr:5.1f}% | PnL: ${s['pnl']:8.2f} | Pips: {s['pips']:6.1f}p")

print("\n=== ALL INDIVIDUAL TRADES FOR YESTERDAY ===")
for t in py_trades:
    print(f"  {t['iso_timestamp']} | #{t['ticket']} | {t['strategy']:20s} | {t['pair']} {t['side']:4s} | pips={t['pips']:6.1f}p | PnL=${t['sim_pnl']:8.2f} | win={t['is_win']}")
