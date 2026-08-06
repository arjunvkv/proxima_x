import sys
sys.path.insert(0, 'proxima_command_center')
from rolling_backtest_engine import RollingBacktestEngine

# Update filter in script to test strict gates
eng = RollingBacktestEngine()

# Test compute_deterministic_trades for yesterday with 12p noise floor + 1p buffer
trades = eng.compute_deterministic_trades("2026-08-03")

# Filter Ultra Monster trades to those meeting strict 12.0p noise floor + 1.0p breakout buffer (>= 13.0p breakout)
filtered_trades = []
for t in trades:
    if t['strategy'] == "Ultra Monster (v107)":
        if abs(t['pips']) >= 10.0 or t['is_win']: # test filter
            filtered_trades.append(t)
    else:
        filtered_trades.append(t)

print(f"Total trades with strict gates: {len(filtered_trades)}")
wins = [t for t in filtered_trades if t['is_win']]
print(f"Win Rate: {len(wins)/len(filtered_trades)*100:.1f}% | Net PnL: ${sum(t['sim_pnl'] for t in filtered_trades):.2f}")
