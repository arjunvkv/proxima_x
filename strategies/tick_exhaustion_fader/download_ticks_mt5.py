"""Download real GBPNZD ticks from MT5 demo feed and analyze for exhaustion patterns."""
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/tick_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SYMBOL = "GBPNZD"
DAYS_BACK = 7
TICK_FILE = os.path.join(OUTPUT_DIR, f"{SYMBOL}_ticks_{DAYS_BACK}d.csv")

if not mt5.initialize():
    print("mt5.initialize() failed")
    sys.exit(1)

print(f"MT5 initialized: {mt5.terminal_info().name}")
print(f"Account: {mt5.account_info().login}")

now = datetime.now()
from_date = now - timedelta(days=DAYS_BACK)

ticks = mt5.copy_ticks_range(SYMBOL, from_date, now, mt5.COPY_TICKS_ALL)
if ticks is None:
    err = mt5.last_error()
    print(f"copy_ticks_range failed: {err}")
    mt5.shutdown()
    sys.exit(1)

df = pd.DataFrame(ticks)
df["time"] = pd.to_datetime(df["time"], unit="s")
df = df.sort_values("time").reset_index(drop=True)

df.to_csv(TICK_FILE, index=False)
print(f"Saved {len(df)} ticks to {TICK_FILE}")
print(f"Date range: {df['time'].min()} to {df['time'].max()}")
print(f"Columns: {list(df.columns)}")
if len(df) > 0:
    print(f"Bid range: {df['bid'].min():.5f} - {df['bid'].max():.5f}")
    print(f"Ask range: {df['ask'].min():.5f} - {df['ask'].max():.5f}")
    print(f"Spread stats: mean={np.mean(df['ask']-df['bid'])*10000:.1f} pip, max={np.max(df['ask']-df['bid'])*10000:.1f} pip")

mt5.shutdown()
