import json

closes = []
with open('logs/trade_journal.jsonl') as f:
    for line in f:
        r = json.loads(line)
        if r.get('event') == 'close':
            closes.append(r)

closes.sort(key=lambda r: r.get('ts', 0), reverse=True)

print('=== 5 MOST RECENT CLOSES ===')
for r in closes[:5]:
    print(f"  {r.get('symbol','?'):<6s} dir={r.get('direction','?'):<4s} pnl=${r.get('pnl',0):.2f}  reason={r.get('exit_reason','?')}  dur={r.get('duration_sec',0):.0f}s")
    print(f"    entry str: { {k:f'{v:+.4f}' for k,v in r.get('strengths_entry',{}).items()} }")
    print(f"    exit  str: { {k:f'{v:+.4f}' for k,v in r.get('strengths_exit',{}).items()} }")
    print()

# Also show the most recent open
opens = []
with open('logs/trade_journal.jsonl') as f:
    for line in f:
        r = json.loads(line)
        if r.get('event') == 'open':
            opens.append(r)
opens.sort(key=lambda r: r.get('ts', 0), reverse=True)
print('=== 5 MOST RECENT OPENS (still open?) ===')
for r in opens[:5]:
    print(f"  {r.get('symbol','?'):<6s} dir={r.get('direction','?'):<4s} conf={r.get('confidence',0):.3f}")
