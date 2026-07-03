"""Lightweight replay benchmark — uses build_replay_environment directly."""
import sys; sys.path.insert(0, ".")
import json, glob, os, time
from replay.environment import build_replay_environment, ReplayConfig
from replay.clock_patcher import patch_clock
from datetime import datetime

# ── Load live capture ──
captures = sorted(glob.glob("research/live_capture/captures/capture_live_15min_*.json"), key=os.path.getmtime)
with open(captures[-1]) as f:
    live_data = json.load(f)

live_ticks = [ev for ev in live_data if ev["type"] == "tick"]
live_cycles = [ev for ev in live_data if ev["type"] == "signal"]
print(f"Live capture: {len(live_ticks)} ticks, {len(live_cycles)} cycles")

# ── Build replay environment ──
config = ReplayConfig(
    symbols=["EURJPY", "USDJPY"],
    start="2026-06-22",
    end="2026-06-22",
    speed=500000, burst=True, latency=False, slippage=False, seed=42,
)
env = build_replay_environment(config)
patch_clock(env.clock)
feed = env.replay_feed
print(f"Feed loaded: {feed._total_loaded} ticks across {len(feed._symbols)} symbols")
print(f"Feed merged: {len(feed._merged)} ticks")

# ── Walk replay ticks (single pass) ──
replay_cycles = {}
replay_first = {}
tick_count = 0
last_min_ts = -1

while True:
    tick = feed.next()
    if tick is None:
        break
    tick_count += 1
    ts = int(tick.get("time_sec", 0))

    # First tick per (timestamp, symbol) pair
    key = (ts, tick.get("symbol", ""))
    if key not in replay_first:
        replay_first[key] = tick

    # Minute boundary (approximating 60s cycle)
    minute_key = ts // 60
    if minute_key > last_min_ts:
        last_min_ts = minute_key
        sym = tick.get("symbol", "")
        eid = f"live_{ts}_{sym}"
        replay_cycles[eid] = {
            "ts": ts, "sym": sym,
            "price": tick.get("ask", 0),
            "spread": tick.get("spread", 0),
        }
        if sym == "EURJPY":
            print(f"  Replay cycle: ts={ts} {sym} price={tick.get('ask', 0):.3f}")

print(f"\nReplay: {tick_count} ticks, {len(replay_cycles)} minute events")

# ── Live tick map ──
live_first = {}
for ev in live_ticks:
    d = ev["data"]
    ts = int(d.get("time", 0))
    sym = d.get("symbol", "").upper()
    key = (ts, sym)
    if key not in live_first:
        live_first[key] = d

# ── Compare ticks ──
common = set(live_first.keys()) & set(replay_first.keys())
print(f"\nTick comparison:")
print(f"  Live (ts,sym) pairs: {len(live_first)}")
print(f"  Replay (ts,sym) pairs: {len(replay_first)}")
print(f"  Common: {len(common)}")

diffs, perfect = [], 0
for key in sorted(common):
    ld = live_first[key]
    rd = replay_first[key]
    la = float(ld.get("ask", 0))
    ra = float(rd.get("ask", 0))
    d = abs(la - ra)
    diffs.append(d)
    if d == 0:
        perfect += 1

if diffs:
    print(f"  Price: perfect={perfect}/{len(diffs)} ({perfect/len(diffs)*100:.1f}%)")
    print(f"  Max diff: {max(diffs):.5f}, Avg: {sum(diffs)/len(diffs):.5f}")

# ── Compare cycles by minute ──
# Live cycle timestamps use system clock (UTC-3 from MT5), replay uses MT5 UTC
# Detect offset from first tick timestamps
first_live_cycle_ts = min(int(ev["data"].get("ts", 0)) for ev in live_cycles)
first_tick_ts = min(int(ev["data"].get("time", 0)) for ev in live_ticks)
clock_offset = first_tick_ts - first_live_cycle_ts
live_minutes = set((int(ev["data"].get("ts", 0)) + clock_offset) // 60 for ev in live_cycles)
replay_minutes = set(d["ts"] // 60 for d in replay_cycles.values())
common_minutes = live_minutes & replay_minutes
print(f"\nCycle comparison (by minute):")
print(f"  Live: {len(live_minutes)}, Replay: {len(replay_minutes)}, Common: {len(common_minutes)}")
print(f"  Live minutes: {sorted(live_minutes)[:5]}...{sorted(live_minutes)[-3:]}")
print(f"  Replay minutes: {sorted(replay_minutes)[:5]}...{sorted(replay_minutes)[-3:]}")
if common_minutes:
    print(f"  Shared: {sorted(common_minutes)}")
print(f"  Clock offset detected: {clock_offset}s (system clock vs MT5)")

# ── Parity score ──
tick_overlap = len(common) / max(len(live_first), 1) * 100
cycle_match = len(common_minutes) / max(len(live_minutes), 1) * 100 if live_minutes else 0
price_fidelity = perfect / max(len(diffs), 1) * 100 if diffs else 0

print("\n" + "=" * 60)
print("COMPOSITE PARITY SCORE")
print("=" * 60)
print(f"  Tick overlap (40%): {tick_overlap:.1f}%")
print(f"  Cycle match (35%): {cycle_match:.1f}%")
print(f"  Price fidelity (25%): {price_fidelity:.1f}%")
score = tick_overlap * 0.40 + cycle_match * 0.35 + price_fidelity * 0.25
grade = "PRODUCTION" if score >= 95 else ("RESEARCH" if score >= 90 else ("USEFUL" if score >= 85 else "REALISM_GAP"))
print(f"\n  Composite: {score:.1f}% — Grade: {grade}")

# ── Save ──
os.makedirs("replay/golden", exist_ok=True)
result = {
    "capture": os.path.basename(captures[-1]),
    "date": "2026-06-22",
    "live_ticks": len(live_ticks), "live_cycles": len(live_cycles),
    "replay_ticks": tick_count, "replay_cycles": len(replay_cycles),
    "tick_overlap_pct": round(tick_overlap, 1),
    "cycle_match_pct": round(cycle_match, 1),
    "price_fidelity_pct": round(price_fidelity, 1),
    "composite_score": round(score, 1), "grade": grade,
    "common_timestamps": len(common),
    "perfect_price_matches": perfect,
}
with open("replay/golden/parity_benchmark.json", "w") as f:
    json.dump(result, f, indent=2)
print(f"\nSaved to replay/golden/parity_benchmark.json")
