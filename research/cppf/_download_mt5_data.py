"""Download all available MT5 M1 data for all pairs and save as parquet."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import pandas as pd
import datetime
import MetaTrader5 as mt5

PAIRS = ["GBPNZD", "EURNZD", "GBPAUD", "EURAUD", "GBPCAD", "AUDNZD"]
OUT_DIR = "research/cppf/_mt5_data"

if not mt5.initialize():
    print("MT5 init failed")
    sys.exit(1)

os.makedirs(OUT_DIR, exist_ok=True)

# Get data ending at current time
now_ts = int(datetime.datetime.now().timestamp())

for pair in PAIRS:
    rates = mt5.copy_rates_from(pair, mt5.TIMEFRAME_M1, now_ts, 50000)
    if rates is None or len(rates) == 0:
        print(f"{pair}: NO DATA")
        continue
    
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.set_index("time")
    
    out_path = os.path.join(OUT_DIR, f"{pair.lower()}.parquet")
    df.to_parquet(out_path)
    
    first = df.index[0]
    last = df.index[-1]
    days = (last - first).days
    print(f"{pair}: {len(df)} bars, {first} to {last} ({days} days)")

mt5.shutdown()
print("Done")
