"""Isolate the dedup bug"""
import sys; sys.path.insert(0, ".")
import polars as pl

# Simulate existing: 30-sec capture (285 ticks at timestamps 100-384) + test tick (at 500)
existing = pl.from_dicts([
    {"timestamp_ns": int(1782129900 + i) * 1_000_000_000, "symbol": "EURJPY", "bid": 100.0 + i*0.001, "time_sec": 1782129900 + i}
    for i in range(285)
] + [
    {"timestamp_ns": 1782130582 * 1_000_000_000, "symbol": "EURJPY", "bid": 100.5, "time_sec": 1782130582}
])
print(f"Existing: {len(existing)} rows, time_sec range: {existing['time_sec'].min()}-{existing['time_sec'].max()}")

# New: 5-min capture (2850 ticks at timestamps 5000-5301)
new = pl.from_dicts([
    {"timestamp_ns": int(1782140780 + i//10) * 1_000_000_000, "symbol": "EURJPY", "bid": 101.0 + i*0.001, "time_sec": 1782140780 + i//10}
    for i in range(2850)
])
print(f"New: {len(new)} rows, time_sec range: {new['time_sec'].min()}-{new['time_sec'].max()}")

# Merge and dedup
merged = pl.concat([existing, new])
print(f"Merged: {len(merged)} rows")

deduped = merged.unique(subset=["timestamp_ns", "symbol"], keep="first")
print(f"Deduped: {len(deduped)} rows")

# Check if new tick timestamps overlap with existing
existing_ts = set(existing["timestamp_ns"].to_list())
new_ts = set(new["timestamp_ns"].to_list())
overlap = existing_ts & new_ts
print(f"Overlapping timestamps: {len(overlap)}")
if overlap:
    overlap_list = sorted(overlap)[:5]
    print(f"  Sample overlapping: {overlap_list}")

# Now test with unique keeping for overlapping timestamps
print(f"\nTicks where timestamp_ns in overlap (existing): {len(existing.filter(pl.col('timestamp_ns').is_in(overlap)))}")
print(f"Ticks where timestamp_ns in overlap (new): {len(new.filter(pl.col('timestamp_ns').is_in(overlap)))}")
