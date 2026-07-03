import polars as pl
from datetime import datetime

base = "C:/Trading/Agentic_Trading/data/intraday"
symbols = ["XAUUSD", "EURJPY", "USDJPY", "GBPJPY"]
for sym in symbols:
    path = f"{base}/{sym}_M5.parquet"
    df = pl.read_parquet(path)
    ts0 = df["timestamp"][0]
    ts1 = df["timestamp"][-1]
    print(f"{sym}: {len(df)} rows")
    print(f"  {datetime.fromtimestamp(ts0)} to {datetime.fromtimestamp(ts1)}")
