"""Inspect tick data structure."""
import sys; sys.path.insert(0, '.')
from research.replay_cache import ReplayCache

ticks = ReplayCache(['EURJPY'], '2026-04-21', '2026-04-21', tick_limit=1000, seed=42).compute()
t = ticks[0]
print('Tick keys:', list(t.keys()))
print('Sample tick:')
for k, v in t.items():
    print(f'  {k}: {v} ({type(v).__name__})')
print(f'\nTotal ticks: {len(ticks)}')
print(f'Price range: {min(t2["price"] for t2 in ticks):.4f} - {max(t2["price"] for t2 in ticks):.4f}')
print(f'ECDF range: {min(t2["ecdf"] for t2 in ticks):.4f} - {max(t2["ecdf"] for t2 in ticks):.4f}')

ts_keys = [k for k in t if 'time' in k.lower() or 'date' in k.lower()]
print(f'\nTimestamp keys: {ts_keys}')
for k in ts_keys:
    print(f'  {k}: {ticks[0][k]} ... {ticks[-1][k]}')

# Check density
import datetime
if ts_keys:
    k = ts_keys[0]
    ts0 = ticks[0][k]
    ts1 = ticks[-1][k]
    if isinstance(ts0, (int, float)):
        print(f'\nTime span: {ts1 - ts0:.1f} seconds = {(ts1-ts0)/3600:.2f} hours')
        print(f'Density: {len(ticks) / (ts1-ts0):.1f} ticks/sec')
    else:
        print(f'\nFirst: {ts0}, Last: {ts1}')
