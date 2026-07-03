import polars as pl
from datetime import datetime

path = "C:/Trading/Agentic_Trading/data/intraday/XAUUSD_M5.parquet"
df = pl.read_parquet(path)
print(f"Total rows: {len(df)}")

ts_col = df["timestamp"]
print(f"First 5 timestamps:")
for i in range(5):
    t = ts_col[i]
    print(f"  [{i}] {t} = {datetime.fromtimestamp(t)}")

print(f"Last 5 timestamps:")
n = len(ts_col)
for i in range(n-5, n):
    t = ts_col[i]
    print(f"  [{i}] {t} = {datetime.fromtimestamp(t)}")

print(f"Unique timestamps: {ts_col.n_unique()}")
print(f"Min ts: {ts_col.min()} = {datetime.fromtimestamp(ts_col.min())}")
print(f"Max ts: {ts_col.max()} = {datetime.fromtimestamp(ts_col.max())}")
