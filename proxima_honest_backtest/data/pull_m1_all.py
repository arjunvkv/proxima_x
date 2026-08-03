"""Pull M1 data from FTMO terminal for all 18 pairs."""
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import MetaTrader5 as mt5
import pandas as pd

SYMBOLS = [
    "EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "EURJPY",
    "GBPJPY", "EURAUD", "EURNZD", "GBPAUD", "GBPNZD",
    "GBPCAD", "AUDNZD", "USDCAD", "NZDUSD", "EURGBP",
    "EURCHF", "USDCHF", "AUDJPY",
]

DATA_DIR = Path(__file__).parent / "m1"
DATA_DIR.mkdir(exist_ok=True)

FROM = datetime(2026, 6, 1)
TO = datetime(2026, 7, 29)

def pull_and_save(symbol: str) -> int:
    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M1, FROM, TO)
    if rates is None or len(rates) == 0:
        print(f"  {symbol}: no data")
        return 0
    
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    
    # Split by month and save
    df["_ym"] = df["time"].dt.strftime("%Y_%m")
    total = 0
    for ym, group in df.groupby("_ym"):
        year, month = ym.split("_")
        pair_dir = DATA_DIR / symbol
        pair_dir.mkdir(exist_ok=True)
        path = pair_dir / f"{year}_{month}.parquet"
        group.drop(columns=["_ym"]).to_parquet(path, index=False)
        total += len(group)
    
    print(f"  {symbol}: {len(df)} M1 bars ({df['time'].min()} -> {df['time'].max()})")
    return len(df)

print(f"Connecting to MT5...")
if not mt5.initialize():
    print(f"Failed: {mt5.last_error()}")
    sys.exit(1)

print(f"Pulling M1 data from {FROM} to {TO}")
print(f"Pairs: {len(SYMBOLS)}")
print()

total = 0
t0 = time.time()
for i, sym in enumerate(SYMBOLS, 1):
    n = pull_and_save(sym)
    total += n
    elapsed = time.time() - t0
    eta = (elapsed / i) * (len(SYMBOLS) - i) if i > 0 else 0
    print(f"  [{i}/{len(SYMBOLS)}] {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining")

mt5.shutdown()
print(f"\nDone: {total:,} M1 bars across {len(SYMBOLS)} pairs in {time.time()-t0:.0f}s")
