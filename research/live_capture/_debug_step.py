"""Debug ingest step by step"""
import sys; sys.path.insert(0, ".")
import json, glob, time
import polars as pl
from replay.tick_archive import TickArchive

captures = sorted(glob.glob("research/live_capture/captures/capture_live_5min_*.json"))
with open(captures[-1]) as f:
    events = json.load(f)

# Build just EURJPY ticks
ticks = []
for ev in events:
    if ev["type"] != "tick":
        continue
    d = ev["data"]
    sym = d.get("symbol", "").upper()
    if sym != "EURJPY":
        continue
    ts = int(d.get("time", 0))
    bid = float(d.get("bid", 0))
    ask = float(d.get("ask", 0))
    ticks.append({
        "timestamp_ns": ts * 1_000_000_000,
        "time_sec": ts,
        "time_msc": ts * 1000,
        "bid": bid,
        "ask": ask,
        "spread": ask - bid,
        "last": bid,
        "volume": 0.0,
        "volume_real": 0.0,
        "flags": 0,
        "symbol": sym,
    })

print(f"Building {len(ticks)} ticks into DataFrame...")
t0 = time.time()
df = pl.from_dicts(ticks, schema={
    "timestamp_ns": pl.Int64, "time_sec": pl.Int64, "time_msc": pl.Int64,
    "bid": pl.Float64, "ask": pl.Float64, "spread": pl.Float64, "last": pl.Float64,
    "volume": pl.Float64, "volume_real": pl.Float64,
    "flags": pl.Int32, "symbol": pl.Utf8,
})
print(f"DataFrame built: {len(df)} rows in {time.time()-t0:.2f}s")

# Read existing
path = "C:/Trading/Agentic_Trading/data/ticks/EURJPY/2026/06/22.parquet"
existing = pl.read_parquet(path)
print(f"Existing: {len(existing)} rows")

# Concatenate
t0 = time.time()
merged = pl.concat([existing, df])
print(f"After concat: {len(merged)} rows in {time.time()-t0:.2f}s")

t0 = time.time()
dedup = merged.unique(subset=["timestamp_ns", "symbol"], keep="first")
print(f"After dedup: {len(dedup)} rows in {time.time()-t0:.2f}s")

t0 = time.time()
sorted_df = dedup.sort("timestamp_ns")
print(f"After sort: {len(sorted_df)} rows in {time.time()-t0:.2f}s")

# Write
t0 = time.time()
table = sorted_df.to_arrow()
print(f"to_arrow: {len(table)} rows in {time.time()-t0:.2f}s")
import pyarrow.parquet as pq
pq.write_table(table, path, compression="zstd")
print(f"Written in {time.time()-t0:.2f}s")

# Verify
verify = pl.read_parquet(path)
print(f"Verify: {len(verify)} rows")
print(f"time_sec range: {verify['time_sec'].min()} - {verify['time_sec'].max()}")
