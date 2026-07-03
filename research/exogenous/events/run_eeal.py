import sys, time
sys.path.insert(0, ".")
sys.path.insert(0, "proxima_x")

t0 = time.perf_counter()
print("=== EEAL FULL TEST ===")
print()

from datetime import datetime
from research.exogenous.events.event_loader import EventLoader
from research.exogenous.events.event_amplitude_cache import EventAmplitudeCache
from research.exogenous.events.event_surface import EventSurface
from research.exogenous.events.event_promotion import EventPromotionEngine
import polars as pl

print("[1/5] Loading events...")
loader = EventLoader()
events = loader.load_builtin_events()
print(f"  {len(events)} events loaded")
for e in events:
    dt = datetime.utcfromtimestamp(e.ts)
    print(f"    {e.name:25s} {e.currency:5s} {e.impact:5s} @ {dt}")
print()

print("[2/5] Building event amplitude cache for EURJPY+USDJPY...")
t1 = time.perf_counter()
cache = EventAmplitudeCache()
# Use single symbol with high tick count
# Apr 3 = NFP day. Apr 1 = ISM day. Both have events at 12:30-14:00 UTC
# Need enough ticks to reach those times from midnight
# Start at 11:00 UTC on April 1 (3h before ISM PMI at 14:00)
# EURJPY ticks ~88k/day = ~3.7k/hour. 3h buffer = ~11k ticks before ISM
# Using n_ticks=80000 covers ~22 hours of data, reaching into April 2
df = cache.build(
    symbols=["EURJPY"],
    start="2026-04-01",
    end="2026-04-02",
    events=events,
    horizons=[60, 300, 900, 1800],
    n_ticks=300000,
    warmup=150000
)
print(f"  {len(df)} observations in {time.perf_counter()-t1:.1f}s")

if len(df) > 0:
    from datetime import datetime
    ts_min = df["state_ts"].min()
    ts_max = df["state_ts"].max()
    print(f"  Time range: {datetime.utcfromtimestamp(ts_min)} to {datetime.utcfromtimestamp(ts_max)}")
    buckets = df["event_bucket"].unique().to_list()
    print(f"  Buckets: {buckets}")
    impacts = df["event_impact"].unique().to_list()
    print(f"  Impacts: {impacts}")
    events_detected = df["event_name"].n_unique()
    print(f"  Unique events: {events_detected}")
    print(f"  Obs per bucket:")
    for b in sorted(buckets):
        cnt = df.filter(pl.col("event_bucket") == b).height
        print(f"    {b}: {cnt}")

print()
print("[3/5] Fitting event surface...")
t1 = time.perf_counter()
surface = EventSurface()
surface.fit(df)
elapsed = time.perf_counter() - t1
print(f"  Surface fitted in {elapsed:.1f}s")
try:
    summary = surface.get_bucket_summary()
    print(f"  Bucket summary ({len(summary)} rows):")
    for r in summary.iter_rows(named=True):
        print(f"    {r['event_bucket']:15s} n={r.get('n', '?'):<6} AER={r.get('aer', 0):.2f} SM={r.get('spread_multiple', 0):.2f}")
except:
    pass

import polars as pl

print()
print("[4/5] Checking promotions...")
entries = []
for key, hz_map in surface._surface.items():
    for hz, entry in hz_map.items():
        entries.append({"key": key, "horizon": hz, "entry": entry})

promoter = EventPromotionEngine()
promotions = promoter.evaluate_all(surface._surface)
tier1 = promotions.filter(pl.col("tier") == 1) if len(promotions) > 0 else promotions
tier2 = promotions.filter(pl.col("tier") == 2) if len(promotions) > 0 else promotions
print(f"  {len(entries)} entries evaluated")
print(f"  Tier 1: {len(tier1)}")
print(f"  Tier 2: {len(tier2)}")

if len(tier1) > 0:
    for r in tier1.iter_rows(named=True):
        print(f"  TIER1: {r['key']} h={r['horizon']}s AER={r['aer']:.2f} SM={r['spread_multiple']:.2f} n={r['n']}")
if len(tier2) > 0:
    for r in tier2.iter_rows(named=True):
        print(f"  TIER2: {r['key']} h={r['horizon']}s AER={r['aer']:.2f} SM={r['spread_multiple']:.2f} n={r['n']}")

print()
print("[5/5] Top entries by AER and spread_multiple:")
if len(promotions) > 0:
    by_aer = promotions.sort("aer", descending=True)
    print("  By AER:")
    for r in by_aer.head(10).iter_rows(named=True):
        print(f"    {r['key']:25s} h={r['horizon']}s AER={r['aer']:.2f} SM={r['spread_multiple']:.2f} n={r['n']} tier={r['tier']}")

    by_sm = promotions.sort("spread_multiple", descending=True)
    print("  By Spread Multiple:")
    for r in by_sm.head(10).iter_rows(named=True):
        print(f"    {r['key']:25s} h={r['horizon']}s AER={r['aer']:.2f} SM={r['spread_multiple']:.2f} n={r['n']} tier={r['tier']}")

# Export reports
from research.exogenous.events.reports import export
export(promotions, "reports")

total = time.perf_counter() - t0
print(f"\nTotal time: {total:.1f}s")
print("=== EEAL COMPLETE ===")

# Verdict
max_aer = promotions["aer"].max() if len(promotions) > 0 else 0
print(f"\n=== VERDICT ===")
print(f"Max AER across all event windows: {max_aer:.2f}")
if max_aer >= 2.5:
    print("AER > 2.5: PROMOTION CANDIDATE — event proximity creates amplitude expansion")
elif max_aer >= 1.8:
    print("AER > 1.8: WEAK SIGNAL — event proximity may weakly condition amplitude")
else:
    print(f"AER < 1.8: FALSIFIED — macro event proximity does NOT break the AER≈1 invariant")
    print("Program VII is dead. Full stop.")
