import json, statistics
from collections import defaultdict

closes = []
with open('logs/trade_journal.jsonl') as f:
    for line in f:
        r = json.loads(line)
        if r.get('event') == 'close':
            closes.append(r)

# Group closes by proximity (sequential <= 2s apart = same batch close)
closes.sort(key=lambda r: r.get('time', 0))
batches = []
current_batch = []
for r in closes:
    if not current_batch:
        current_batch = [r]
    elif r.get('time', 0) - current_batch[-1].get('time', 0) <= 2.0:
        current_batch.append(r)
    else:
        batches.append(current_batch)
        current_batch = [r]
if current_batch:
    batches.append(current_batch)

print(f'Total closes: {len(closes)} in {len(batches)} batch-close events')
print()

# For each batch close, find matching opens and compute hold time
open_map = {}
with open('logs/trade_journal.jsonl') as f:
    for line in f:
        r = json.loads(line)
        if r.get('event') == 'open':
            key = (r.get('symbol'), r.get('position_id'))
            open_map[key] = r.get('time', 0)

# Classify batches as early or normal by avg hold time
early_batches = []
normal_batches = []
for batch in batches:
    hold_times = []
    for r in batch:
        key = (r.get('symbol'), r.get('position_id'))
        if key in open_map:
            hold_times.append(r.get('time', 0) - open_map[key])
    if hold_times:
        avg_hold = statistics.mean(hold_times)
        batch_pnl = sum(r.get('pnl', 0) for r in batch)
        if avg_hold < 180:  # less than 3 min = early kill
            early_batches.append((avg_hold, batch_pnl, len(batch), batch))
        else:
            normal_batches.append((avg_hold, batch_pnl, len(batch), batch))

print(f'Early-killed batches (<180s hold): {len(early_batches)}')
print(f'Normal batches (>=180s hold): {len(normal_batches)}')
print()

if early_batches:
    early_pnl = sum(b[1] for b in early_batches)
    normal_pnl = sum(b[1] for b in normal_batches)
    print(f'Early batch total PnL: ${early_pnl:.2f}')
    print(f'Normal batch total PnL: ${normal_pnl:.2f}')
    print()
    
    early_batches.sort(key=lambda b: b[0])
    print('=== 20 MOST AGGRESSIVELY KILLED BATCHES ===')
    for avg_h, pnl, cnt, batch in early_batches[:20]:
        symbols = ','.join(r.get('symbol','?') for r in batch)
        print(f'  hold={avg_h:.0f}s  pnl=${pnl:.2f}  size={cnt}  [{symbols}]')

print()
print('=== PnL by hold-time bucket ===')
buckets = {'0-30s': [], '30-60s': [], '60-120s': [], '120-180s': [],
           '180-300s': [], '300-600s': [], '600s+': []}
for r in closes:
    key = (r.get('symbol'), r.get('position_id'))
    hold = r.get('time', 0) - open_map.get(key, r.get('time', 0))
    if hold < 30: buckets['0-30s'].append(r.get('pnl',0))
    elif hold < 60: buckets['30-60s'].append(r.get('pnl',0))
    elif hold < 120: buckets['60-120s'].append(r.get('pnl',0))
    elif hold < 180: buckets['120-180s'].append(r.get('pnl',0))
    elif hold < 300: buckets['180-300s'].append(r.get('pnl',0))
    elif hold < 600: buckets['300-600s'].append(r.get('pnl',0))
    else: buckets['600s+'].append(r.get('pnl',0))
for k, v in buckets.items():
    if v:
        avg = statistics.mean(v)
        print(f'  {k:>10s}: n={len(v):3d}  avg=${avg:.2f}  total=${sum(v):.2f}')
