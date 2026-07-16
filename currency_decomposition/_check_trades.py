import json, statistics
from collections import defaultdict

closes = []
with open('logs/trade_journal.jsonl') as f:
    for line in f:
        r = json.loads(line)
        if r.get('event') == 'close':
            closes.append(r)

reasons = defaultdict(list)
for r in closes:
    reason = r.get('exit_reason', 'UNKNOWN')
    reasons[reason].append(r)

print(f'Total closes: {len(closes)}')
print()
print('=== Exit reasons ===')
for reason, rs in sorted(reasons.items(), key=lambda x: -len(x[1])):
    pnl = sum(r.get('pnl', 0) for r in rs)
    print(f'  {reason:<25s}: {len(rs):3d}  total_pnl=${pnl:.2f}')

print()
print('=== PnL by duration (for closes with duration) ===')
dur_buckets = {'0-30s': [], '30-60s': [], '60-120s': [], '120-180s': [],
               '180-300s': [], '300-600s': [], '600s+': []}
for r in closes:
    d = r.get('duration_sec', 0)
    if d <= 0: continue
    if d < 30: dur_buckets['0-30s'].append(r)
    elif d < 60: dur_buckets['30-60s'].append(r)
    elif d < 120: dur_buckets['60-120s'].append(r)
    elif d < 180: dur_buckets['120-180s'].append(r)
    elif d < 300: dur_buckets['180-300s'].append(r)
    elif d < 600: dur_buckets['300-600s'].append(r)
    else: dur_buckets['600s+'].append(r)

for k, rs in dur_buckets.items():
    if rs:
        avg_dur = statistics.mean(r.get('duration_sec',0) for r in rs)
        avg_pnl = statistics.mean(r.get('pnl',0) for r in rs)
        total_pnl = sum(r.get('pnl',0) for r in rs)
        print(f'  {k:>10s}: n={len(rs):3d}  avg_dur={avg_dur:.0f}s  avg_pnl=${avg_pnl:.2f}  total=${total_pnl:.2f}')
