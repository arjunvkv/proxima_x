import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime
import os

BASE = r"C:\Trading\Agentic_Trading\proxima_x\paper_trade\mt5_backtest\custom_data"

print("Downloading fresh data...")
if not mt5.initialize():
    print("MT5 init failed")
    exit()

pairs = ["EURJPY", "EURUSD", "GBPJPY"]

dt_from = datetime(2026, 6, 8, 0, 0)
dt_to = datetime(2026, 7, 29, 23, 59)

data = {}
times_dict = {}

for pair in pairs:
    rates = mt5.copy_rates_range(pair, mt5.TIMEFRAME_M1, dt_from, dt_to)
    if rates is None:
        print(f"{pair}: download failed")
        continue
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    data[pair] = df
    times_dict[pair] = set(df["time"].values)
    print(f"{pair}: {len(df)} bars")

mt5.shutdown()

# Find common timestamps
common = sorted(times_dict["EURJPY"] & times_dict["EURUSD"] & times_dict["GBPJPY"])
print(f"Common bars: {len(common)}")

# Export CSVs
for pair in pairs:
    df = data[pair]
    df = df[df["time"].isin(common)]
    df = df.sort_values("time")
    csv_path = os.path.join(BASE, f"FN_{pair}.csv")
    df.to_csv(
        csv_path, index=False, header=False,
        date_format="%Y.%m.%d,%H:%M",
        columns=["time", "open", "high", "low", "close", "tick_volume"]
    )
    print(f"Exported {csv_path}: {len(df)} bars")

print("Done")
