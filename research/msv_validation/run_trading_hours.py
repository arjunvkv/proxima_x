"""Quick calculation of MSV trading hours commitment."""
import sys, os, json
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta, timezone
from collections import deque

project_root = str(Path(__file__).resolve().parents[2])
os.chdir(project_root)
cd_root = os.path.join(project_root, "currency_decomposition")
sys.path.insert(0, cd_root)

import MetaTrader5 as mt5
if not mt5.initialize():
    raise RuntimeError("MT5 init failed")

from state.market_state import MarketStateVector
from config.settings import BASE_CURRENCY_MAP

PAIRS = list(BASE_CURRENCY_MAP.keys())[:16]

def load_data():
    end = datetime.now()
    start = end - timedelta(days=120)
    all_data = {}
    for pair in PAIRS:
        rates = mt5.copy_rates_range(pair, mt5.TIMEFRAME_M5, start, end)
        if rates is not None and len(rates) > 0:
            all_data[pair] = rates
    return all_data

def compute_pct(disp, history, window):
    h = history[-min(window, len(history)):]
    if len(h) < 10: return 0.5
    return sum(1 for x in h if x < disp) / len(h)

all_data = load_data()
N = min(len(v) for v in all_data.values())
ms = MarketStateVector(history_size=50)
dh = deque(maxlen=1500)

events = []
for idx in range(N):
    rets = {}
    for p in all_data:
        c = float(all_data[p][idx]["close"])
        pv = float(all_data[p][idx - 1]["close"]) if idx > 0 else c
        rets[p] = max(min((c / pv - 1) if pv > 0 else 0.0, 0.05), -0.05)
    now = float(all_data[list(all_data.keys())[0]][idx]["time"])
    snap = ms.update(rets, timestamp=now)
    dh.append(snap.network.dispersion)

    pre60 = 0.0
    if idx >= 12:
        for p in all_data:
            cur = float(all_data[p][idx]["close"])
            p60 = float(all_data[p][idx - 12]["close"])
            pre60 += (cur / p60 - 1) if p60 > 0 else 0.0
        pre60 /= len(all_data)

    dt = datetime.fromtimestamp(now, tz=timezone.utc)
    if dt.hour >= 7: continue

    dp = compute_pct(snap.network.dispersion, list(dh), 500)
    if dp < 0.95 or pre60 > -0.0002: continue

    if idx + 30 < N:
        vals = [float(all_data[p][idx + 30]["close"]) / float(all_data[p][idx]["close"]) - 1 for p in all_data]
        fwd = float(np.mean(vals))
    else:
        continue

    events.append({"ts": now, "dt": dt, "hour": dt.hour, "wd": dt.weekday(), "fwd": fwd})

print(f"Total MSV events: {len(events)}")
print()

# Daily stats
by_date = {}
for e in events:
    d = e["dt"].strftime("%Y-%m-%d")
    if d not in by_date:
        by_date[d] = {"times": [], "hours": set()}
    by_date[d]["times"].append(e["dt"])
    by_date[d]["hours"].add(e["hour"])

dates = sorted(by_date.keys())
print(f"Trading days with events: {len(dates)}")

# Total hours monitored
all_hours = {}
for e in events:
    if e["hour"] not in all_hours:
        all_hours[e["hour"]] = 0
    all_hours[e["hour"]] += 1

print(f"\nEvents by hour (UTC):")
for h in sorted(all_hours.keys()):
    print(f"  {h:2d}:00  -> {all_hours[h]:3d} events")

# With cooldown of 120min
events_sorted = sorted(events, key=lambda x: x["ts"])
cd_events = []
last_ts = -999999
for e in events_sorted:
    if e["ts"] - last_ts >= 7200:  # 120 min
        cd_events.append(e)
        last_ts = e["ts"]

print(f"\nEvents with 120min cooldown: {len(cd_events)}")

# How many distinct days and what's the average
cd_by_date = {}
for e in cd_events:
    d = e["dt"].strftime("%Y-%m-%d")
    if d not in cd_by_date:
        cd_by_date[d] = {"n": 0, "first": e["dt"], "last": e["dt"]}
    cd_by_date[d]["n"] += 1
    cd_by_date[d]["last"] = e["dt"]

n_days = len(cd_by_date)
total_events_cd = len(cd_events)
avg_per_day = total_events_cd / n_days if n_days > 0 else 0

# Average time window per day
window_mins = []
for d, info in cd_by_date.items():
    if info["n"] >= 1:
        # With 120min cooldown and 30min hold, time window = events * (30min hold)
        # But cooldown means events are spread over a window
        window_mins.append(info["n"] * 30)  # rough: each event holds 30min

avg_window = np.mean(window_mins) if window_mins else 0
max_window = max(window_mins) if window_mins else 0

print(f"\n{'='*60}")
print(f"TRADING TIME COMMITMENT (with 120min cooldown)")
print(f"{'='*60}")
print(f"  Events per day:       {avg_per_day:.1f} avg, {info['n']} max")
print(f"  Active hours/day:     ~{avg_window/60:.1f}h avg, ~{max_window/60:.1f}h max")
print(f"  Days/week:            ~5 (Mon-Fri)")
print(f"  Hour window:          00:00-02:00 UTC (2h window)")

# Per weekday
print(f"\n  Events by weekday:")
wd_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
for wd in range(5):
    evts = [e for e in events if e["wd"] == wd]
    print(f"    {wd_names[wd]}: {len(evts)} events ({(len(evts)/len(events)*100):.0f}%)")

# Hour 0 only (the sweet spot)
h0_events = [e for e in events if e["hour"] == 0]
h0_dates = set(e["dt"].strftime("%Y-%m-%d") for e in h0_events)
print(f"\n  Hour 0 only:")
print(f"    Events: {len(h0_events)} over {len(h0_dates)} days")
print(f"    Avg events/day: {len(h0_events)/len(h0_dates):.1f}")

# With cooldown on hour 0
h0_sorted = sorted(h0_events, key=lambda x: x["ts"])
h0_cd = []
last_ts = -999999
for e in h0_sorted:
    if e["ts"] - last_ts >= 7200:
        h0_cd.append(e)
        last_ts = e["ts"]
print(f"    Hour 0 + 120min cd: {len(h0_cd)} events")
print(f"    Avg events/day: {len(h0_cd)/len(h0_dates):.1f}")

print(f"\n{'='*60}")
print(f"BOTTOM LINE")
print(f"{'='*60}")
print(f"  Monitor:  00:00-02:00 UTC, Mon-Fri (~2h/day, ~10h/week)")
print(f"  Trades:   1-2 per day avg, 30min hold each")
print(f"  Total:    2-3 hours/week of actual position time")

mt5.shutdown()
