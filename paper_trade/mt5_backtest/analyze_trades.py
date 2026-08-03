import re
from collections import defaultdict

log_path = r"C:\Users\Arjun Sasi\AppData\Roaming\MetaQuotes\Terminal\81A933A9AFC5DE3C23B15CAB19C63850\Tester\logs\20260727.log"

with open(log_path, encoding='utf-16') as f:
    raw = f.read()

lines = raw.replace('\r\n', '\n').split('\n')

print(f"Total lines: {len(lines)}")
print(f"File size: {len(raw)} chars")

# Parse all LOST/CLOSE events (these contain full trade data)
trades = []
for line in lines:
    line = line.strip()
    if not line:
        continue
    
    # Only process trade lines
    if 'OPEN ' in line and 'dir=' in line:
        # Also capture OPEN info
        continue  # we'll use close/lost lines instead
    
    # CLOSE rsn=stop pnl=...
    m = re.match(r'^\w+\t\d+\t(\d+:\d+:\d+\.\d+)\tCore \d+\t(\d+\.\d+\.\d+)\s+(\d+:\d+:\d+)\s+(CLOSE \w+ rsn=\w+ pnl=[-\d.]+.*)', line)
    if m:
        log_ts = m.group(1)
        date = m.group(2)
        time = m.group(3)
        msg = m.group(4)
        
        cm = re.match(r'CLOSE (\w+) rsn=(\w+) pnl=([-\d.]+) z=([-\d.e+]+) atr=([\d.e-]+) sprd=([\d.]+) held=(\d+) bars?', msg)
        if cm:
            t = {
                'type': 'CLOSE',
                'symbol': cm.group(1),
                'reason': cm.group(2),
                'pnl': float(cm.group(3)),
                'z_entry': float(cm.group(4)),
                'atr': float(cm.group(5)),
                'spread': float(cm.group(6)),
                'held_bars': int(cm.group(7)),
                'date': date,
                'time': time,
                'log_ts': log_ts,
                'result': 'WIN' if float(cm.group(3)) > 0 else 'LOSS',
            }
            trades.append(t)
            continue
    
    # LOST dir=1 entry=... stop=... pnl=...
    m = re.match(r'^\w+\t\d+\t(\d+:\d+:\d+\.\d+)\tCore \d+\t(\d+\.\d+\.\d+)\s+(\d+:\d+:\d+)\s+(LOST \w+ dir=\d+ entry=[\d.]+.*)', line)
    if m:
        log_ts = m.group(1)
        date = m.group(2)
        time = m.group(3)
        msg = m.group(4)
        
        lm = re.match(r'LOST (\w+) dir=(\d+) entry=([\d.]+) stop=([\d.]+) pnl=([-\d.]+) atr=([\d.e-]+) z=([-\d.e+]+) sprd=([\d.]+) held=(\d+) bars?', msg)
        if lm:
            t = {
                'type': 'LOST',
                'symbol': lm.group(1),
                'dir': int(lm.group(2)),
                'entry': float(lm.group(3)),
                'stop': float(lm.group(4)),
                'pnl': float(lm.group(5)),
                'atr': float(lm.group(6)),
                'z_entry': float(lm.group(7)),
                'spread': float(lm.group(8)),
                'held_bars': int(lm.group(9)),
                'date': date,
                'time': time,
                'log_ts': log_ts,
                'result': 'WIN' if float(lm.group(5)) > 0 else 'LOSS',
            }
            trades.append(t)
            continue
    
    # STABLE entry
    m = re.match(r'^\w+\t\d+\t(\d+:\d+:\d+\.\d+)\tCore \d+\t(\d+\.\d+\.\d+)\s+(\d+:\d+:\d+)\s+(STABLE entry z=.*)', line)
    if m:
        log_ts = m.group(1)
        msg = m.group(4)
        sm = re.match(r'STABLE entry z=([-\d.e+]+) z_cum=([-\d.e+]+) spd_z=([-\d.e+]+) vol_z=([-\d.e+]+) stab=([\d.e+]+)', msg)
        if sm:
            t = {
                'type': 'STABLE_ENTRY',
                'z_stable': float(sm.group(1)),
                'z_cum': float(sm.group(2)),
                'spd_z': float(sm.group(3)),
                'vol_z': float(sm.group(4)),
                'stab': float(sm.group(5)),
                'log_ts': log_ts,
            }
            trades.append(t)

print(f"\nTotal parsed trade events: {len(trades)}")
close_trades = [t for t in trades if t['type'] in ('CLOSE', 'LOST')]
stable_entries = [t for t in trades if t['type'] == 'STABLE_ENTRY']
print(f"  CLOSE/LOST trades: {len(close_trades)}")
print(f"  STABLE_ENTRY lines: {len(stable_entries)}")

# Group by run (log_ts)
by_run = defaultdict(list)
for t in close_trades:
    by_run[t['log_ts']].append(t)

print(f"\n=== RESULTS BY RUN ===")
for run_ts in sorted(by_run.keys()):
    ts = by_run[run_ts]
    total_pnl = sum(t['pnl'] for t in ts)
    wins = sum(1 for t in ts if t['result'] == 'WIN')
    losses = sum(1 for t in ts if t['result'] == 'LOSS')
    sym = ts[0]['symbol'] if ts else '?'
    print(f"Run {run_ts[:8]} | {sym} | {len(ts)} trades | {wins}W/{losses}L "
          f"| WR={wins/len(ts)*100:.1f}% | PnL=${total_pnl:.2f}")

# Aggregate across all runs
print(f"\n=== AGGREGATE BY PAIR ===")
by_sym = defaultdict(list)
for t in close_trades:
    by_sym[t['symbol']].append(t)

for sym in sorted(by_sym.keys()):
    ts = by_sym[sym]
    total_pnl = sum(t['pnl'] for t in ts)
    wins = sum(1 for t in ts if t['result'] == 'WIN')
    losses = sum(1 for t in ts if t['result'] == 'LOSS')
    avg_win = sum(t['pnl'] for t in ts if t['result'] == 'WIN') / wins if wins > 0 else 0
    avg_loss = sum(t['pnl'] for t in ts if t['result'] == 'LOSS') / losses if losses > 0 else 0
    print(f"{sym}: {len(ts)} trades | {wins}W/{losses}L | "
          f"WR={wins/len(ts)*100:.1f}% | AvgW=${avg_win:.2f} | AvgL=${avg_loss:.2f} | "
          f"Net=${total_pnl:.2f}")

# Z-score buckets
print(f"\n=== Z-SCORE BUCKETS (ALL PAIRS) ===")
buckets = defaultdict(list)
for t in close_trades:
    z = abs(t['z_entry'])
    bucket = f"{int(z)}-{int(z+1)}" if z >= 0 else f"neg"
    buckets[bucket].append(t)

for bucket in sorted(buckets.keys()):
    ts = buckets[bucket]
    total_pnl = sum(t['pnl'] for t in ts)
    wins = sum(1 for t in ts if t['result'] == 'WIN')
    print(f"  z={bucket}: {len(ts)} trades | {wins}W | "
          f"WR={wins/len(ts)*100:.1f}% | PnL=${total_pnl:.2f}")

# Spread buckets
print(f"\n=== SPREAD BUCKETS ===")
buckets = defaultdict(list)
for t in close_trades:
    s = t['spread']
    bucket = f"{int(s)}"
    buckets[bucket].append(t)

for bucket in sorted(buckets.keys(), key=int):
    ts = buckets[bucket]
    total_pnl = sum(t['pnl'] for t in ts)
    wins = sum(1 for t in ts if t['result'] == 'WIN')
    print(f"  sprd={bucket}: {len(ts)} trades | {wins}W | "
          f"WR={wins/len(ts)*100:.1f}% | PnL=${total_pnl:.2f}")

# Hour buckets
print(f"\n=== HOUR OF DAY (UTC) ===")
buckets = defaultdict(list)
for t in close_trades:
    h = t['time'].split(':')[0]
    buckets[h].append(t)

for h in sorted(buckets.keys()):
    ts = buckets[h]
    total_pnl = sum(t['pnl'] for t in ts)
    wins = sum(1 for t in ts if t['result'] == 'WIN')
    print(f"  h={h}: {len(ts)} trades | {wins}W | "
          f"WR={wins/len(ts)*100:.1f}% | PnL=${total_pnl:.2f}")

# Held bars
print(f"\n=== HELD BARS ===")
buckets = defaultdict(list)
for t in close_trades:
    hb = t['held_bars']
    bucket = '0-1' if hb <= 1 else '2-3' if hb <= 3 else '4-6' if hb <= 6 else '7-10' if hb <= 10 else '11+'
    buckets[bucket].append(t)

for bucket in ['0-1', '2-3', '4-6', '7-10', '11+']:
    if bucket not in buckets:
        continue
    ts = buckets[bucket]
    total_pnl = sum(t['pnl'] for t in ts)
    wins = sum(1 for t in ts if t['result'] == 'WIN')
    print(f"  held={bucket}: {len(ts)} trades | {wins}W | "
          f"WR={wins/len(ts)*100:.1f}% | PnL=${total_pnl:.2f}")

# Stability component analysis (for stable gate runs)
print(f"\n=== STABILITY GATE COMPONENT ANALYSIS ===")
# Match stable entries to trades
print(f"Stable entry lines: {len(stable_entries)}")
if stable_entries:
    for entry in stable_entries[:5]:
        print(f"  z={entry['z_stable']:.1f} z_cum={entry['z_cum']:.1f} spd_z={entry['spd_z']:.2f} "
              f"vol_z={entry['vol_z']:.2f} stab={entry['stab']:.3f}")
