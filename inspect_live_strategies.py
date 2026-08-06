import sys
sys.path.insert(0, 'proxima_command_center')
from mt5_history_loader import get_side_by_side_trade_comparison

trades = get_side_by_side_trade_comparison()
print(f"Total Live Trades fetched: {len(trades)}")

by_strat = {}
for t in trades:
    st = t['strategy']
    by_strat.setdefault(st, []).append(t)

print("\n=== LIVE MT5 TRADES BY STRATEGY ===")
for st, tlist in by_strat.items():
    wins = [t for t in tlist if t['net_pnl'] >= 0]
    pnl = sum(t['net_pnl'] for t in tlist)
    wr = len(wins) / len(tlist) * 100 if tlist else 0
    print(f"{st:25s} | Trades: {len(tlist):3d} | WR: {wr:5.1f}% | Net PnL: ${pnl:8.2f}")
