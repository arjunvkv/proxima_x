"""Run sim_backtest.py with EXACT config from MT5 tester results (V2z_CPPF.mq5)."""
import subprocess, os, sys

pairs = ["AUDNZD", "EURAUD", "EURNZD", "GBPAUD", "GBPCAD", "GBPNZD"]
# Exact config from AGENTS.md: z=3.5, stop_a=3.0, trig_a=1.0, gap_a=0.05, lot=0.75
# Hours: 0-7 UTC, both directions (no TRADE_DIRECTION param in original EA)
config = "3.5 3.0 1.0 0.05 0.75"

# Also print the EA code structure differences
print("=" * 70)
print("EXACT V2z_CPPF.mq5 CONFIG FORWARD (Jun 8-Jul 25)")
print("=" * 70)

results = []
for pair in pairs:
    print(f"\n--- {pair} ---")
    cmd = f'python sim_backtest.py {pair} 2026-06-08 2026-07-25 {config}'
    proc = subprocess.run(cmd, capture_output=True, text=True, shell=True,
                         cwd=os.path.dirname(__file__))
    output = proc.stdout
    print(output[:1500])

    net = trades = wr = pf_val = 0.0
    for line in output.split('\n'):
        if 'Net PnL:' in line: net = float(line.split('$')[1])
        elif 'Total trades:' in line: trades = int(line.split(':')[1])
        elif line.strip().startswith('Wins:'):
            wins = int(line.split('(')[0].split(':')[1])
            wr = wins/trades*100 if trades else 0
        elif 'Profit factor:' in line:
            try: pf_val = float(line.split(':')[1])
            except: pass
    results.append({'pair': pair, 'trades': trades, 'wr': wr, 'net': net, 'pf': pf_val})

print("\n\n" + "=" * 70)
print("SUMMARY vs MT5 FORWARD RESULTS")
print("=" * 70)
print(f"{'PAIR':<10} {'SIM_TRADES':>12} {'MT5_TRADES':>12} {'SIM_WR':>8} {'MT5_WR':>8} {'SIM_PnL':>12} {'MT5_PnL':>12}")
print("-" * 74)
for r in results:
    mt5_map = {
        "AUDNZD": (214, 58.4, -581),
        "EURAUD": (215, 66.5, 313),
        "EURNZD": (119, 66.4, -52),
        "GBPAUD": (179, 70.9, 499),
        "GBPCAD": (180, 61.7, -351),
        "GBPNZD": (115, 60.0, -792),
    }
    mt5 = mt5_map[r['pair']]
    print(f"{r['pair']:<10} {r['trades']:>12d} {mt5[0]:>12d} "
          f"{r['wr']:>7.1f}% {mt5[1]:>7.1f}% "
          f"${r['net']:>+9.2f} ${mt5[2]:>+8.2f}")

total_sim = sum(r['net'] for r in results)
total_mt5 = sum(-581+313-52+499-351-792)
total_trades_sim = sum(r['trades'] for r in results)
total_trades_mt5 = 214+215+119+179+180+115
print("-" * 74)
print(f"{'TOTAL':<10} {total_trades_sim:>12d} {total_trades_mt5:>12d} "
      f"${total_sim:>+9.2f} ${total_mt5:>+8.2f}")
print(f"\nInflation: sim shows ${total_sim:.0f} vs real ${total_mt5:.0f} "
      f"= {total_sim/abs(total_mt5) if total_mt5 else 0:.1f}x")

print("\n\n" + "=" * 70)
print("EA BUGS THAT MAKE SIM INACCURATE")
print("=" * 70)
print("""
1. g_bars_held = 0 on line 294 — resets bars_held EVERY new bar.
   EA: never expires (ManagePosition++ then IsNewBar resets to 0)
   Sim: increments once per bar, expires after 54 bars
   -> EA holds positions MUCH longer (until stopped)

2. EA CheckEntry fires AFTER bar completes (uses completed close via CopyClose)
   Sim: z.iloc[i] uses close[i] - close[i-1] — includes CURRENT bar's close
   -> SIM HAS LOOK-AHEAD BIAS: enters knowing bar's close

3. EA enters at tick.ask/tick.bid (first tick of new bar)
   Sim enters at bar[i].close
   -> Different entry prices, sim may enter better or worse

4. EA checks stop on every tick; sim checks M1 low/high
   -> Sim misses intra-bar stop-outs
""")
