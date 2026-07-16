import json

lines = []
with open('logs/trade_journal.jsonl') as f:
    for i, line in enumerate(f):
        lines.append((i, json.loads(line)))

# Show last 20 records of any event type
for i, r in lines[-20:]:
    ev = r.get('event', '?')
    sym = r.get('symbol', '')
    pnl = r.get('pnl', '')
    reason = r.get('exit_reason', '')
    dur = r.get('duration_sec', '')
    ts = r.get('ts', '')
    conf = r.get('confidence', '')
    drs = r.get('drs_score', '')
    print(f"  line {i:>4d}: {ev:<6s} {sym:<6s} pnl={pnl} reason={reason} dur={dur}s ts={ts}")
