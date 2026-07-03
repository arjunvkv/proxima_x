import polars as pl
from datetime import datetime
for sym in ["EURJPY", "USDJPY"]:
    p = "C:/Trading/Agentic_Trading/data/ticks/" + sym + "/2026/06/22.parquet"
    try:
        df = pl.read_parquet(p)
        print(f"{sym}: {len(df)} rows")
        print(f"  time_sec range: {datetime.fromtimestamp(df['time_sec'].min())} - {datetime.fromtimestamp(df['time_sec'].max())}")
    except Exception as e:
        print(f"{sym}: {e}")
