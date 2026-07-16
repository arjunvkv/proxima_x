import json, statistics

nd_closes = []
with open('logs/trade_journal.jsonl') as f:
    for line in f:
        r = json.loads(line)
        if r.get('event') == 'close' and r.get('reason') == 'NARRATIVE_DECAY':
            nd_closes.append(r)

total = len(nd_closes)
profit = sum(r.get('pnl', 0) for r in nd_closes)
winners = [r for r in nd_closes if r.get('pnl', 0) > 0]
losers = [r for r in nd_closes if r.get('pnl', 0) < 0]
breakeven = [r for r in nd_closes if r.get('pnl', 0) == 0]

print(f'NARRATIVE_DECAY closes: {total}')
print(f'Total PnL: ${profit:.2f}')
print(f'Winners: {len(winners)} Total: ${sum(r["pnl"] for r in winners):.2f}')
print(f'Losers: {len(losers)} Total: ${sum(r["pnl"] for r in losers):.2f}')
print(f'Breakeven: {len(breakeven)}')
print()

winners.sort(key=lambda r: r['pnl'], reverse=True)
losers.sort(key=lambda r: r['pnl'])
print('=== TOP 10 ND WINNERS (biggest positive PnL when killed) ===')
for r in winners[:10]:
    print(f"  {r.get('symbol','?'):<6s} pnl=${r['pnl']:.2f} hold={r.get('hold_seconds',0):.0f}s")
print()
print('=== TOP 10 ND LOSERS (most negative when killed) ===')
for r in losers[:10]:
    print(f"  {r.get('symbol','?'):<6s} pnl=${r['pnl']:.2f} hold={r.get('hold_seconds',0):.0f}s")
print()

hold_times = [r.get('hold_seconds', 0) for r in nd_closes if r.get('hold_seconds', 0) > 0]
if hold_times:
    print(f'Avg hold: {statistics.mean(hold_times):.0f}s  Median: {statistics.median(hold_times):.0f}s')
    print(f'Min hold: {min(hold_times):.0f}s  Max hold: {max(hold_times):.0f}s')

buckets = {'<-$5': 0, '-$5 to -$2': 0, '-$2 to $0': 0, '$0 to $2': 0, '$2 to $5': 0, '$5+': 0}
for r in nd_closes:
    p = r.get('pnl', 0)
    if p < -5: buckets['<-$5'] += 1
    elif p < -2: buckets['-$5 to -$2'] += 1
    elif p < 0: buckets['-$2 to $0'] += 1
    elif p < 2: buckets['$0 to $2'] += 1
    elif p < 5: buckets['$2 to $5'] += 1
    else: buckets['$5+'] += 1
print()
print('=== PnL Distribution ===')
for k, v in buckets.items():
    bar = '█' * min(v, 40)
    print(f'  {k:>14s}: {v:3d} {bar}')
