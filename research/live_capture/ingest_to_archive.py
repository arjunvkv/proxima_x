"""Ingest live-captured ticks into the tick archive for replay benchmarking."""
import sys; sys.path.insert(0, '.')
import json
import os
from replay.tick_archive import TickArchive


def ingest_capture(capture_path: str):
    with open(capture_path) as f:
        events = json.load(f)

    archive = TickArchive()
    by_symbol: dict[str, list[dict]] = {}

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
        if sym not in by_symbol:
            by_symbol[sym] = []
        by_symbol[sym].append(archive_tick)

    for sym, ticks in by_symbol.items():
        print(f"Ingesting {len(ticks)} ticks for {sym}...")
        archive.store_ticks(sym, ticks)

    print(f"Ingested {sum(len(v) for v in by_symbol.values())} total ticks into archive")
    return by_symbol


if __name__ == "__main__":
    import glob, os
    captures = glob.glob("research/live_capture/captures/capture_live_*.json")
    if not captures:
        print("No live captures found")
    else:
        latest = max(captures, key=os.path.getmtime)
        print(f"Ingesting {latest}...")
        ingest_capture(latest)
