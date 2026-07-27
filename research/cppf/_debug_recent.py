"""Debug: show bars around live trade times."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import pandas as pd
import datetime
import numpy as np

LIVE_TRADES = {
    "GBPNZD": {"dir": -1, "entry": 2.30782, "entry_ts": 1784830140},
    "EURNZD": {"dir": -1, "entry": 1.97113, "entry_ts": 1784830201},
    "GBPAUD": {"dir": 1,  "entry": 1.91089, "entry_ts": 1784830261},
    "EURAUD": {"dir": -1, "entry": 1.63194, "entry_ts": 1784830321},
    "GBPCAD": {"dir": 1,  "entry": 1.87472, "entry_ts": 1784830381},
    "AUDNZD": {"dir": -1, "entry": 1.20770, "entry_ts": 1784830441},
}

# Check local vs UTC interpretation
print(f"System TZ offset check:")
print(f"  1784830140 as UTC: {datetime.datetime.fromtimestamp(1784830140, tz=datetime.timezone.utc)}")
print(f"  1784830140 as local: {datetime.datetime.fromtimestamp(1784830140)}")

for pair, live in sorted(LIVE_TRADES.items()):
    fpath = f"research/cppf/_live_data/{pair.lower()}.parquet"
    df = pd.read_parquet(fpath)
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    df.index = pd.to_datetime(df.index)
    
    # Show index dtype
    print(f"\n{pair}: index dtype={df.index.dtype}, len={len(df)}")
    print(f"  Range: {df.index[0]} to {df.index[-1]}")
    
    # Find bar around entry time
    entry_ts = live["entry_ts"]
    entry_dt_utc = datetime.datetime.fromtimestamp(entry_ts, tz=datetime.timezone.utc)
    
    # Search for bar at or before entry_ts
    mask = df.index <= entry_dt_utc.replace(tzinfo=None)
    if mask.any():
        idx = mask.sum() - 1
        row = df.iloc[idx]
        print(f"  Entry bar (UTC {entry_dt_utc}): idx={idx} time={df.index[idx]} open={row['open']:.5f} high={row['high']:.5f} low={row['low']:.5f} close={row['close']:.5f}")
        expected_close = live["entry"]
        close_diff = abs(row["close"] - expected_close)
        print(f"  Close vs live entry: {row['close']:.5f} vs {expected_close:.5f} diff={close_diff:.5f}")
        if idx + 1 < len(df):
            row2 = df.iloc[idx + 1]
            print(f"  Next bar: time={df.index[idx+1]} open={row2['open']:.5f} high={row2['high']:.5f} low={row2['low']:.5f} close={row2['close']:.5f}")
    else:
        print(f"  Entry bar {entry_dt_utc} NOT FOUND in data range")
        # Check if the data index is tz-aware
        print(f"  Index tz: {df.index.tz}")
