"""Replay benchmark: runs replay over a date window and compares with live capture signals."""
import sys; sys.path.insert(0, '.')
import json
import glob
from replay.environment import build_replay_environment, ReplayConfig
from replay.clock_patcher import patch_clock
from datetime import datetime

captures = sorted(glob.glob('research/live_capture/captures/capture_live_5min_*.json'))
with open(captures[-1]) as f:
    live_data = json.load(f)

live_signals = {}
for ev in live_data:
    if ev["type"] == "signal":
        eid = ev.get("event_id", "")
        live_signals[eid] = ev["data"]

# Build replay
config = ReplayConfig(
    symbols=["EURJPY", "USDJPY"],
    start="2026-06-22",
    end="2026-06-22",
    speed=500000,
    burst=True,
    latency=False,
    slippage=False,
    seed=42,
)
env = build_replay_environment(config)
patch_clock(env.clock)

# Warmup
for _ in range(1000):
    if env.tick_source.get_tick("EURJPY") is None:
        break

# Collect replay signals
from signals.outcome_surface_signal import OutcomeSurfaceSignal
from evaluation.delayed_outcome_engine import DelayedOutcomeEngine
from research.replay_cache import ReplayCache

oss = OutcomeSurfaceSignal()
try:
    cache = ReplayCache(["EURJPY"], "2026-04-01", "2026-04-20", tick_limit=50000, seed=42)
    ticks = cache.compute()
    recs, ed, doa = [], {}, DelayedOutcomeEngine(horizon_ticks=20)
    for t in ticks[:10000]:
        s = t["sym"]
        d = t.get("ecdf", 0.5) - 0.5
        sig = 1 if d > 0.05 else (-1 if d < -0.05 else 0)
        ed[s] = {"price": t["price"], "ecdf_rank": t["ecdf"], "signal": sig}
        doa.record_snapshot(ed)
        if doa.ready:
            for s2, outcome in doa.evaluate({s2: ed[s2]["price"] for s2 in ed}).items():
                recs.append({"sym": s2, "ecdf": ed[s2]["ecdf_rank"], "outcome": outcome})
    if recs:
        oss = OutcomeSurfaceSignal.from_pipeline_records(recs)
except Exception:
    pass

# Walk through ticks and record cycle-level signals
replay_signals = {}
import time
start_wall = time.time()
wall_start = start_wall
tick_source = env.tick_source
cycle_count = 0

while True:
    tick = tick_source.get_tick("EURJPY")
    if tick is None:
        break
    
    now = tick.get("time_sec", 0)
    elapsed = time.time() - wall_start
    
    # Simulate 60-second cycle
    if int(now) % 60 == 0 and tick.get("symbol") == "EURJPY":
        eid = f"live_{now}_EURJPY"
        sig = oss.predict(0.5)
        replay_signals[eid] = {
            "oss_signal": sig,
            "ecdf_rank": 0.5,
            "price": tick.get("ask", 0),
            "spread": tick.get("spread", 0),
            "ts": now,
        }

# Compare
print("=" * 60)
print("LIVE vs REPLAY SIGNAL COMPARISON")
print("=" * 60)
match = 0
total = 0
for eid, ld in live_signals.items():
    rd = replay_signals.get(eid)
    if rd is None:
        print(f"MISS: {eid} not found in replay")
        continue
    total += 1
    if ld.get("oss_signal") == rd.get("oss_signal"):
        match += 1
        print(f"MATCH: {eid} signal={ld.get('oss_signal')}")
    else:
        print(f"MISMATCH: {eid} live={ld.get('oss_signal')} replay={rd.get('oss_signal')}")

print(f"\nParity: {match}/{total} = {match/total*100:.1f}%" if total else "No signals matched")
print(f"Live signals: {len(live_signals)}, Replay cycles: {cycle_count}")
