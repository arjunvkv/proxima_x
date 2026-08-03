"""Analyze stability gate components vs trade outcomes."""
import re
from collections import defaultdict

log_path = r"C:\Users\Arjun Sasi\AppData\Roaming\MetaQuotes\Terminal\81A933A9AFC5DE3C23B15CAB19C63850\Tester\logs\20260727.log"

with open(log_path, encoding='utf-16') as f:
    raw = f.read()
lines = raw.replace('\r\n', '\n').split('\n')

# Parse ALL LOST/CLOSE trades with their metadata
trades = []
for line in lines:
    line = line.strip()
    if not line:
        continue
    if 'OPEN ' in line and 'dir=' in line:
        continue

    # CLOSE
    m = re.match(r'^\w+\t\d+\t(\d+:\d+:\d+\.\d+)\tCore \d+\t(\d+\.\d+\.\d+)\s+(\d+:\d+:\d+)\s+(CLOSE \w+ rsn=\w+ pnl=[-\d.]+.*)', line)
    if m:
        msg = m.group(4)
        cm = re.match(r'CLOSE (\w+) rsn=(\w+) pnl=([-\d.]+) z=([-\d.e+]+) atr=([\d.e-]+) sprd=([\d.]+) held=(\d+) bars?', msg)
        if cm:
            trades.append({
                'type': 'CLOSE', 'symbol': cm.group(1), 'pnl': float(cm.group(3)),
                'z_entry': float(cm.group(4)), 'atr': float(cm.group(5)),
                'spread': float(cm.group(6)), 'held_bars': int(cm.group(7)),
                'log_ts': m.group(1), 'date': m.group(2), 'time': m.group(3),
                'result': 'WIN' if float(cm.group(3)) > 0 else 'LOSS',
            })
            continue

    # LOST
    m = re.match(r'^\w+\t\d+\t(\d+:\d+:\d+\.\d+)\tCore \d+\t(\d+\.\d+\.\d+)\s+(\d+:\d+:\d+)\s+(LOST \w+ dir=\d+ entry=[\d.]+.*)', line)
    if m:
        msg = m.group(4)
        lm = re.match(r'LOST (\w+) dir=(\d+) entry=([\d.]+) stop=([\d.]+) pnl=([-\d.]+) atr=([\d.e-]+) z=([-\d.e+]+) sprd=([\d.]+) held=(\d+) bars?', msg)
        if lm:
            trades.append({
                'type': 'LOST', 'symbol': lm.group(1), 'pnl': float(lm.group(5)),
                'z_entry': float(lm.group(7)), 'atr': float(lm.group(6)),
                'spread': float(lm.group(8)), 'held_bars': int(lm.group(9)),
                'log_ts': m.group(1), 'date': m.group(2), 'time': m.group(3),
                'result': 'WIN' if float(lm.group(5)) > 0 else 'LOSS',
            })

# Parse STABLE entry lines
stable_entries = defaultdict(list)
for line in lines:
    line = line.strip()
    m = re.match(r'^\w+\t\d+\t(\d+:\d+:\d+\.\d+)\tCore \d+\t(\d+\.\d+\.\d+)\s+(\d+:\d+:\d+)\s+(STABLE entry z=.*)', line)
    if m:
        msg = m.group(4)
        sm = re.match(r'STABLE entry z=([-\d.e+]+) z_cum=([-\d.e+]+) spd_z=([-\d.e+]+) vol_z=([-\d.e+]+) stab=([\d.e+]+)', msg)
        if sm:
            stable_entries[m.group(1)].append({
                'z': float(sm.group(1)), 'z_cum': float(sm.group(2)),
                'spd_z': float(sm.group(3)), 'vol_z': float(sm.group(4)),
                'stab': float(sm.group(5)),
                'date': m.group(2), 'time': m.group(3),
            })

# Focus on the two working pairs: GBPUSD and AUDUSD
# Skip USDJPY (pnl in JPY not USD)
working_symbols = {'GBPUSD', 'AUDUSD'}
working_trades = [t for t in trades if t['symbol'] in working_symbols]

print(f"=== WORKING PAIRS (GBPUSD + AUDUSD) ===")
print(f"Total trades: {len(working_trades)}")

by_sym = defaultdict(list)
for t in working_trades:
    by_sym[t['symbol']].append(t)

for sym in sorted(by_sym.keys()):
    ts = by_sym[sym]
    total_pnl = sum(t['pnl'] for t in ts)
    wins = sum(1 for t in ts if t['result'] == 'WIN')
    losses = sum(1 for t in ts if t['result'] == 'LOSS')
    avg_win = sum(t['pnl'] for t in ts if t['result'] == 'WIN') / wins if wins > 0 else 0
    avg_loss = sum(t['pnl'] for t in ts if t['result'] == 'LOSS') / losses if losses > 0 else 0
    print(f"\n{sym}: {len(ts)} trades, {wins}W/{losses}L, "
          f"WR={wins/len(ts)*100:.1f}%, AvgW=${avg_win:.2f}, AvgL=${avg_loss:.2f}, "
          f"Net=${total_pnl:.2f}")

# === SPREAD VS OUTCOME ===
print(f"\n=== SPREAD VS WIN RATE (WORKING PAIRS ONLY) ===")
for sprd in sorted(set(t['spread'] for t in working_trades)):
    ts = [t for t in working_trades if t['spread'] == sprd]
    wins = sum(1 for t in ts if t['result'] == 'WIN')
    pnl = sum(t['pnl'] for t in ts)
    print(f"  sprd={int(sprd):2d}: {len(ts):3d} trades, {wins:2d}W, "
          f"WR={wins/len(ts)*100:5.1f}%, PnL=${pnl:7.2f}")

# === Z-SCORE VS OUTCOME ===
print(f"\n=== Z-SCORE VS WIN RATE ===")
for z_bucket in range(3, 10):
    ts = [t for t in working_trades if z_bucket <= abs(t['z_entry']) < z_bucket + 1]
    if not ts:
        continue
    wins = sum(1 for t in ts if t['result'] == 'WIN')
    pnl = sum(t['pnl'] for t in ts)
    print(f"  z={z_bucket}-{z_bucket+1}: {len(ts):3d} trades, {wins:2d}W, "
          f"WR={wins/len(ts)*100:5.1f}%, PnL=${pnl:7.2f}")

# === HOUR ANALYSIS ===
print(f"\n=== HOUR VS WIN RATE ===")
for h in sorted(set(int(t['time'].split(':')[0]) for t in working_trades)):
    ts = [t for t in working_trades if int(t['time'].split(':')[0]) == h]
    wins = sum(1 for t in ts if t['result'] == 'WIN')
    pnl = sum(t['pnl'] for t in ts)
    print(f"  h={h:02d}: {len(ts):3d} trades, {wins:2d}W, "
          f"WR={wins/len(ts)*100:5.1f}%, PnL=${pnl:7.2f}")

# === HELD BARS ANALYSIS ===
print(f"\n=== HELD BARS VS WIN RATE ===")
for hb in sorted(set(t['held_bars'] for t in working_trades)):
    ts = [t for t in working_trades if t['held_bars'] == hb]
    wins = sum(1 for t in ts if t['result'] == 'WIN')
    pnl = sum(t['pnl'] for t in ts)
    print(f"  held={hb:2d}: {len(ts):3d} trades, {wins:2d}W, "
          f"WR={wins/len(ts)*100:5.1f}%, PnL=${pnl:7.2f}")

# === STABILITY GATE COMPONENT ANALYSIS ===
# Match stable entries to trades in the same run
print(f"\n=== STABILITY ENTRY COMPONENTS ===")
print(f"Total stable entry lines: {sum(len(v) for v in stable_entries.values())}")
for run_ts in sorted(stable_entries.keys()):
    entries = stable_entries[run_ts]
    # Get trades from the same run
    run_trades = [t for t in trades if t['log_ts'] == run_ts]
    
    print(f"\nRun {run_ts[:8]} ({len(run_trades)} trades, {len(entries)} stable entries):")
    
    # Analyze stability components
    for comp in ['z', 'z_cum', 'spd_z', 'vol_z', 'stab']:
        values = [e[comp] for e in entries]
        if not values:
            continue
        avg = sum(values) / len(values)
        min_v = min(values)
        max_v = max(values)
        print(f"  {comp}: avg={avg:.3f}, min={min_v:.3f}, max={max_v:.3f}")

# === SPECIFIC ANALYSIS: Which gate component filters what ===
# Look at runs with ENABLE_STABILITY_GATE=true
# Run 18:09:58 = baseline (no gate) - 19 GBPUSD trades
# Run 18:13:05 = stable (gate=0.5, z_cum=5) - 6 GBPUSD trades
# Run 18:28:52 = stable (gate=0.3, z_cum=5) - 6 GBPUSD trades
# Run 18:29:39 = stable (gate=0.3, z_cum=0) - 8 GBPUSD trades

# These show: z_cum_min=5 blocks 2 trades (8→6)
# And gate=0.5 vs 0.3 filters the same (both 6)
# So the z_cum filter is the binding constraint

print(f"\n=== GBPUSD: GATE IMPACT ANALYSIS ===")
runs_gbp = {
    'baseline (no gate)': '18:09:58',
    'stable 0.5 zcum5': '18:13:05',
    'stable 0.3 zcum5': '18:28:52',
    'stable 0.3 zcum0': '18:29:39',
    'z3.5 stable 0.3 zcum0': '19:44:31',
}
for label, ts in runs_gbp.items():
    run_trades = [t for t in trades if t['log_ts'].startswith(ts) and t['symbol'] == 'GBPUSD']
    if not run_trades:
        continue
    wins = sum(1 for t in run_trades if t['result'] == 'WIN')
    pnl = sum(t['pnl'] for t in run_trades)
    print(f"  {label:30s}: {len(run_trades):2d} trades, {wins}W, WR={wins/len(run_trades)*100:.0f}%, PnL=${pnl:.2f}")

# Compare which trades were gained/lost between gate configs
print(f"\n=== GBPUSD: TRADE-LEVEL COMPARISON ===")
base_trades = [t for t in trades if t['log_ts'].startswith('18:09:58') and t['symbol'] == 'GBPUSD']
stable_trades = [t for t in trades if t['log_ts'].startswith('18:29:39') and t['symbol'] == 'GBPUSD']

# The stable 0.3 zcum0 run had 8 trades from baseline 19
# Which 11 were filtered out?
base_by_date = {t['date'] + t['time']: t for t in base_trades}
stable_by_date = {t['date'] + t['time']: t for t in stable_trades}

filtered_out = []
for key, t in sorted(base_by_date.items()):
    if key not in stable_by_date:
        filtered_out.append(t)

print(f"  Baseline: {len(base_trades)} trades")
print(f"  Stable: {len(stable_trades)} trades")
print(f"  Filtered out: {len(filtered_out)} trades")
print(f"  Filtered out PnL: ${sum(t['pnl'] for t in filtered_out):.2f}")
print(f"  Filtered out WR: {sum(1 for t in filtered_out if t['result']=='WIN')}/{len(filtered_out)}")
for t in filtered_out:
    print(f"    {t['date']} {t['time']} spr={t['spread']} z={t['z_entry']:.1f} held={t['held_bars']} "
          f"{t['result']} pnl=${t['pnl']:.1f}")
