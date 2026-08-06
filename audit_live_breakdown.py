import sys
sys.path.insert(0, 'proxima_command_center')
from mt5_history_loader import get_side_by_side_trade_comparison

trades = get_side_by_side_trade_comparison()

print("=== ALL 74 LIVE TRADES SUMMARY ===")
print(f"Total Trades: {len(trades)}")

# Separate strategy vs manual
automated = [t for t in trades if t['strategy'] != "Manual / Test Script"]
manual = [t for t in trades if t['strategy'] == "Manual / Test Script"]

print(f"\n1. MANUAL / TEST SCRIPTS (38 trades):")
m_wins = [t for t in manual if t['net_pnl'] >= 0]
print(f"   Trades: {len(manual)} | Win Rate: {len(m_wins)/len(manual)*100:.1f}% | Net PnL: ${sum(t['net_pnl'] for t in manual):.2f}")

print(f"\n2. PROXIMA AUTOMATED STRATEGIES (36 trades):")
a_wins = [t for t in automated if t['net_pnl'] >= 0]
print(f"   Trades: {len(automated)} | Win Rate: {len(a_wins)/len(automated)*100:.1f}% | Net PnL: ${sum(t['net_pnl'] for t in automated):.2f}")

print("\n3. BREAKDOWN BY AUTOMATED STRATEGY:")
by_strat = {}
for t in automated:
    by_strat.setdefault(t['strategy'], []).append(t)

for st, tlist in by_strat.items():
    wins = [t for t in tlist if t['net_pnl'] >= 0]
    pnl = sum(t['net_pnl'] for t in tlist)
    wr = len(wins) / len(tlist) * 100
    print(f"   - {st:23s}: Trades={len(tlist):2d} | WR={wr:5.1f}% | PnL=${pnl:8.2f}")
