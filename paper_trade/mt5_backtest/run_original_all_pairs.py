"""Run original sim_backtest.py on all 6 cross pairs to reproduce +$16k forward."""
import subprocess, sys, os

pairs = ["AUDNZD", "EURAUD", "EURNZD", "GBPAUD", "GBPCAD", "GBPNZD"]
config = "3.5 3.0 1.0 0.05 1.0"
results = []

for pair in pairs:
    print(f"\n{'='*60}")
    print(f"RUNNING {pair}...")
    print(f"{'='*60}")
    cmd = f'python sim_backtest.py {pair} 2026-06-08 2026-07-25 {config}'
    proc = subprocess.run(cmd, capture_output=True, text=True, shell=True, cwd=os.path.dirname(__file__))
    output = proc.stdout
    print(output[:2000])

    # Parse results
    net = 0
    trades = 0
    wr = 0
    for line in output.split('\n'):
        if 'Net PnL:' in line:
            net = float(line.split('$')[1].replace(',', ''))
        elif 'Total trades:' in line:
            trades = int(line.split(':')[1].strip())
        elif line.strip().startswith('Wins:'):
            wins = int(line.split('(')[0].split(':')[1].strip())
            wr = wins/trades*100 if trades else 0

    results.append({'pair': pair, 'trades': trades, 'wr': wr, 'net': net})
    print(f"  -> {pair}: {trades} trades, {wr:.1f}% WR, ${net:+.2f}")

print(f"\n\n{'='*60}")
print("TOTAL RESULTS (6 pairs)")
print(f"{'='*60}")
total_pnl = sum(r['net'] for r in results)
total_trades = sum(r['trades'] for r in results)
print(f"{'PAIR':<10} {'TRADES':>8} {'WR':>7} {'PnL':>12}")
print("-" * 40)
for r in results:
    print(f"{r['pair']:<10} {r['trades']:>8d} {r['wr']:>6.1f}% ${r['net']:>+8.2f}")
print("-" * 40)
print(f"{'TOTAL':<10} {total_trades:>8d} ${total_pnl:>+8.2f}")
