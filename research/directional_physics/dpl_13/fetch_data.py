import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
import polars as pl
import numpy as np
from datetime import datetime, timedelta
import MetaTrader5 as mt5

if not mt5.initialize():
    print("MT5 init failed")
    sys.exit(1)

out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "data", "intraday")
os.makedirs(out_dir, exist_ok=True)

symbols = ["XAUUSD", "EURJPY", "USDJPY", "GBPJPY"]
CHUNK_SIZE = 50000

for sym in symbols:
    print(f"Fetching {sym}...", end=" ")
    mt5.symbol_select(sym, True)
    all_rates = []
    start_dt = datetime.now()
    while True:
        rates = mt5.copy_rates_from(sym, mt5.TIMEFRAME_M5, start_dt, CHUNK_SIZE)
        if rates is None or len(rates) == 0:
            break
        all_rates.extend(rates)
        start_dt = datetime.fromtimestamp(rates[0]["time"]) - timedelta(minutes=5)
        if len(rates) < CHUNK_SIZE:
            break
    if not all_rates:
        print(f"FAILED: {mt5.last_error()}")
        continue
    all_rates.reverse()
    seen = set()
    deduped = []
    for r in all_rates:
        t = r["time"]
        if t not in seen:
            seen.add(t)
            deduped.append(r)
    print(f"{len(deduped)} bars")
    df = pl.DataFrame({
        "timestamp": [int(r["time"]) for r in deduped],
        "open": [float(r["open"]) for r in deduped],
        "high": [float(r["high"]) for r in deduped],
        "low": [float(r["low"]) for r in deduped],
        "close": [float(r["close"]) for r in deduped],
        "volume": [float(r["tick_volume"]) for r in deduped],
    })
    dt0 = datetime.fromtimestamp(deduped[0]["time"])
    dt1 = datetime.fromtimestamp(deduped[-1]["time"])
    print(f"  Range: {dt0.date()} to {dt1.date()}")
    out_path = os.path.join(out_dir, f"{sym}_M5.parquet")
    df.write_parquet(out_path)
    print(f"  Saved {len(df)} rows to {out_path}")

mt5.shutdown()
print("Done")
