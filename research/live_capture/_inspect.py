import json, glob
from datetime import datetime
captures = sorted(glob.glob('research/live_capture/captures/capture_live_5min_*.json'))
with open(captures[-1]) as f:
    data = json.load(f)
ts_min = min(ev['data']['time'] for ev in data if ev['type'] == 'tick')
ts_max = max(ev['data']['time'] for ev in data if ev['type'] == 'tick')
signals = [ev for ev in data if ev['type'] == 'signal']
print("Date range:", datetime.fromtimestamp(ts_min), "to", datetime.fromtimestamp(ts_max))
print("Total events:", len(data))
print("Ticks:", len([e for e in data if e["type"]=="tick"]))
print("Signals:", len(signals))
for s in signals[:3]:
    print("  Signal:", "event_id=", s.get("event_id","?"), "data=", s["data"])
