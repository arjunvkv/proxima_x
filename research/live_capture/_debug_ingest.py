"""Debug ingest - check partition after store"""
import sys; sys.path.insert(0, ".")
import json, glob
from replay.tick_archive import TickArchive

captures = sorted(glob.glob("research/live_capture/captures/capture_live_5min_*.json"))
with open(captures[-1]) as f:
    events = json.load(f)

archive = TickArchive()
by_symbol = {}
for ev in events:
    if ev["type"] != "tick":
        continue
    d = ev["data"]
    sym = d.get("symbol", "").upper()
    ts = int(d.get("time", 0))
    bid = float(d.get("bid", 0))
    ask = float(d.get("ask", 0))
    spread_price = ask - bid
    archive_tick = {
        "timestamp_ns": ts * 1_000_000_000,
        "time_sec": ts,
        "time_msc": ts * 1000,
        "bid": bid,
        "ask": ask,
        "spread": d.get("spread_raw", spread_price),
        "last": bid,
        "volume": 0.0,
        "volume_real": 0.0,
        "flags": 0,
        "symbol": sym,
    }
    by_symbol.setdefault(sym, []).append(archive_tick)

print("Before ingest:")
import polars as pl
try:
    old = pl.read_parquet("C:/Trading/Agentic_Trading/data/ticks/EURJPY/2026/06/22.parquet")
    print(f"  EURJPY June 22: {len(old)} rows")
    print(f"  time_sec range: {old['time_sec'].min()} - {old['time_sec'].max()}")
except:
    print("  No existing partition")

import time
t0 = time.time()
for sym, ticks_list in by_symbol.items():
    print(f"Storing {len(ticks_list)} ticks for {sym}...")
    archive.store_ticks(sym, ticks_list)
print(f"Store took {time.time()-t0:.2f}s")

print("After ingest:")
df = pl.read_parquet("C:/Trading/Agentic_Trading/data/ticks/EURJPY/2026/06/22.parquet")
print(f"  EURJPY June 22: {len(df)} rows")
ts_col = df["time_sec"]
print(f"  time_sec range: {ts_col.min()} - {ts_col.max()}")
unique_ts = sorted(ts_col.unique().to_list())
print(f"  Unique times (first 5): {unique_ts[:5]}")
print(f"  Unique times (last 5): {unique_ts[-5:]}")
