import polars as pl
from datetime import datetime

path = "C:/Trading/Agentic_Trading/proxima_x/data/intraday/XAUUSD_M5.parquet"
df = pl.read_parquet(path)
print(f"Rows: {len(df)}")
ts0 = df[0]["timestamp"]
ts1 = df[-1]["timestamp"]
print(f"First ts: {ts0} -> {datetime.fromtimestamp(ts0)}")
print(f"Last ts:  {ts1} -> {datetime.fromtimestamp(ts1)}")
