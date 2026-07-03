"""Inspect live cycle timestamps from 15-minute capture"""
import json, glob, os
from datetime import datetime

captures = sorted(glob.glob("research/live_capture/captures/capture_live_15min_*.json"), key=os.path.getmtime)
with open(captures[-1]) as f:
    data = json.load(f)

cycles = [ev for ev in data if ev["type"] == "signal"]
print(f"Total cycles: {len(cycles)}")
for c in cycles[:5]:
    d = c["data"]
    ts = d.get("ts", 0)
    eid = c.get("event_id", "?")
    print(f"  event_id={eid} ts={ts} dt={datetime.fromtimestamp(ts) if ts else 'N/A'} price={d.get('price')}")

# Check first and last tick timestamps
ticks = [ev for ev in data if ev["type"] == "tick"]
if ticks:
    ts0 = ticks[0]["data"].get("time", 0)
    ts1 = ticks[-1]["data"].get("time", 0)
    print(f"\nFirst tick: ts={ts0} dt={datetime.fromtimestamp(ts0)}")
    print(f"Last tick: ts={ts1} dt={datetime.fromtimestamp(ts1)}")
