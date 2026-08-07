"""Phase 0 verification: canonical tick contract + MVS engine on replay source."""
import sys, os, tempfile, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.canonical_tick import normalize_tick, CANONICAL_FIELDS

print("=" * 60)
print("TEST 1: normalize_tick across all producer variants")
print("=" * 60)

# Variant A: live MT5 connector shape
live = {"symbol": "EURJPY", "bid": 162.301, "ask": 162.303, "spread": 20, "time": 1786000000, "time_msc": 1786000000123, "flags": 2, "volume": 1.0, "_source": "live_mt5"}
# Variant B: replay archive row
archive = {"symbol": "EURJPY", "time_sec": 1786000000, "time_msc": 1786000000123, "timestamp_ns": 1786000000123456789, "bid": 162.301, "ask": 162.303, "last": 162.302, "volume": 0.0, "volume_real": 0.5, "flags": 0, "_seq": 42, "_source": "archive"}
# Variant C: MT5 history tick
hist = {"symbol": "EURJPY", "time_msc": 1786000000123, "bid": 162.301, "ask": 162.303, "last": 162.302, "volume": 1, "flags": 2, "_source": "mt5_history"}
# Variant D: MVS TickLoader input (minimal)
minimal = {"symbol": "EURJPY", "time": 1786000000, "bid": 162.301, "ask": 162.303, "_source": "minimal"}

for name, raw in [("live", live), ("archive", archive), ("mt5_hist", hist), ("minimal", minimal)]:
    t = normalize_tick(raw)
    missing = [f for f in CANONICAL_FIELDS if f not in t]
    assert not missing, f"{name}: missing {missing}"
    assert abs(t["mid"] - 162.302) < 1e-9, f"{name}: mid wrong {t['mid']}"
    assert t["spread"] > 0, f"{name}: spread not positive"
    assert t["ts_ns"] > 0, f"{name}: ts_ns zero"
    assert t["symbol"] == "EURJPY", f"{name}: symbol lost"
    # spot check spread_pts consistent
    print(f"  {name:10s} bid={t['bid']:.5f} ask={t['ask']:.5f} mid={t['mid']:.5f} "
          f"spread={t['spread']:.6f} pts={t['spread_pts']} ts_ns={t['ts_ns']} src={t['_source']}")

# idempotency: normalizing a canonical tick is a no-op
t1 = normalize_tick(live)
t2 = normalize_tick(t1)
assert t1["ts_ns"] == t2["ts_ns"] and abs(t1["mid"] - t2["mid"]) < 1e-9
assert abs(t1["spread"] - t2["spread"]) < 1e-9
print("  [OK] normalize_tick is idempotent on canonical input")

print()
print("=" * 60)
print("TEST 2: TickLoader via ReplayTickSource (injectable source)")
print("=" * 60)

from data.replay_tick_source import ReplayTickSource
from mvs.reconstruction.tick_loader import TickLoader

# Build a fake feed with archive-shaped ticks (like ReplayFeed._merged)
class FakeFeed:
    def __init__(self, ticks):
        self._ticks = ticks
        self._i = 0
    def next(self):
        if self._i >= len(self._ticks):
            return None
        t = self._ticks[self._i]; self._i += 1
        return t
    def seek(self, i): self._i = i

fake_ticks = []
for i in range(10):
    bid = 162.300 + i * 0.001
    fake_ticks.append({"symbol": "EURJPY", "time_sec": 1786000000 + i,
                       "time_msc": (1786000000 + i) * 1000,
                       "bid": bid, "ask": bid + 0.002,
                       "last": bid, "volume": 0.0, "volume_real": 0.3,
                       "flags": 0, "_seq": i, "_source": "archive"})

src = ReplayTickSource(FakeFeed(fake_ticks))
loader = TickLoader("EURJPY", tick_source=src)
ticks = []
for _ in range(10):
    ticks.append(loader.next())

assert len(ticks) == 10
first = ticks[0]
for k in ("tick_id", "symbol", "ts_ns", "bid", "ask", "mid", "spread", "delta", "velocity", "acceleration", "jerk"):
    assert k in first, f"missing {k} in MVS tick"
# velocity should be nonzero once prices move
assert ticks[1]["velocity"] != 0, "velocity not computed"
print(f"  [OK] 10 ticks loaded through replay source")
print(f"  first: mid={first['mid']:.5f} ts_ns={first['ts_ns']} delta={first['delta']}")
print(f"  second: velocity={ticks[1]['velocity']:.6f} (non-zero, computed)")

print()
print("=" * 60)
print("TEST 3: MVSEngine runs UNCHANGED against replay source")
print("=" * 60)

from mvs.orchestrator import MVSEngine

tmp = tempfile.mkdtemp()
db = os.path.join(tmp, "mvs_test.duckdb")
try:
    src2 = ReplayTickSource(FakeFeed(fake_ticks))
    engine = MVSEngine("EURJPY", db_path=db, tick_source=src2)
    for _ in range(10):
        tick = engine.run_tick()
    ranking = engine.update_honesty()
    assert ranking, "honesty ranking empty"
    print(f"  [OK] engine ran 10 ticks on replay source, honesty ranking computed")
    print(f"  ranking sample: {[(name, round(s, 1)) for name, s, _ in ranking[:4]]}")
    engine.close()
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print()
print("ALL PHASE 0 TESTS PASSED")
