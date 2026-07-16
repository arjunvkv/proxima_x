import json, statistics

opens = {}
with open('logs/trade_journal.jsonl') as f:
    for line in f:
        r = json.loads(line)
        if r.get('event') == 'open':
            opens[r.get('position_id')] = r

closes = []
with open('logs/trade_journal.jsonl') as f:
    for line in f:
        r = json.loads(line)
        if r.get('event') == 'close':
            closes.append(r)

# Compute hold time per close
for r in closes:
    o = opens.get(r.get('position_id'))
    if o:
        r['_hold'] = r.get('time', 0) - o.get('time', 0)
    else:
        r['_hold'] = 0

# Group into batch-close events by timestamp proximity
closes.sort(key=lambda r: r.get('time', 0))
batches = []
cur = []
for r in closes:
    if not cur:
        cur = [r]
    elif r.get('time', 0) - cur[-1].get('time', 0) <= 3.0:
        cur.append(r)
    else:
        if cur:
            batches.append(cur)
        cur = [r]
if cur:
    batches.append(cur)

print(f'{len(closes)} closes in {len(batches)} batch-close events')
print()

# Bucket batches by avg hold time
early = [(b, statistics.mean([r['_hold'] for r in b])) for b in batches]
by_hold = {}
for b, h in early:
    key = f'{int(h//30*30)}-{int((h//30+1)*30)}s' if h < 180 else '180s+'
    by_hold.setdefault(key, []).append(b)

print('=== Batch close events by hold time ===')
for key in sorted(by_hold.keys()):
    bs = by_hold[key]
    total_pnl = sum(sum(r.get('pnl', 0) for r in b) for b in bs)
    print(f'  {key:>10s}: {len(bs):3d} batches  total_pnl=${total_pnl:.2f}')

print()

# Show batches with hold < 180s sorted by PnL (most negative first = most prematurely killed)
short_batches = [(b, h) for b, h in early if h < 180]
short_batches.sort(key=lambda x: sum(r.get('pnl',0) for r in x[0]))

print(f'=== Early-killed batches (hold < 180s, {len(short_batches)} total) ===')
for b, h in short_batches[:20]:
    pnl = sum(r.get('pnl',0) for r in b)
    syms = ','.join(r.get('symbol','?') for r in b)
    print(f'  hold={h:.0f}s  pnl=${pnl:.2f}  size={len(b)}  [{syms}]')

print()
print('=== PnL distribution by hold time ===')
for r in closes:
    pass
buckets = {'0-30s': [], '30-60s': [], '60-120s': [], '120-180s': [], '180-300s': [], '300-600s': [], '600s+': []}
for r in closes:
    h = r['_hold']
    if h < 30: buckets['0-30s'].append(r.get('pnl',0))
    elif h < 60: buckets['30-60s'].append(r.get('pnl',0))
    elif h < 120: buckets['60-120s'].append(r.get('pnl',0))
    elif h < 180: buckets['120-180s'].append(r.get('pnl',0))
    elif h < 300: buckets['180-300s'].append(r.get('pnl',0))
    elif h < 600: buckets['300-600s'].append(r.get('pnl',0))
    else: buckets['600s+'].append(r.get('pnl',0))
for k, v in buckets.items():
    if v:
        avg = statistics.mean(v)
        print(f'  {k:>10s}: n={len(v):3d}  avg=${avg:.2f}  total=${sum(v):.2f}')
