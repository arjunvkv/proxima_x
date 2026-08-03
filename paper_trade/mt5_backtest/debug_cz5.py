"""Minimal test: does the warmup condition work?"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
from datetime import datetime, timezone

fn = np.load(os.path.join(os.path.dirname(__file__), 'fundednext_audusd_m1.npy'), allow_pickle=True)
df = pd.DataFrame(fn)
times_ns = df['time'].values.astype('int64') // 10**9

start_dt = datetime(2026, 6, 8, tzinfo=timezone.utc)
z_window = 50
warmup_bars = max(z_window + 3, 60)

print(f"start_dt = {start_dt}")
print(f"warmup_bars = {warmup_bars}")
print(f"First timestamp = {times_ns[0]} → {datetime.fromtimestamp(times_ns[0], tz=timezone.utc)}")

# Check a few bars
for i in [0, 50, 61, 100, 200, 250]:
    dt = datetime.fromtimestamp(times_ns[i], tz=timezone.utc)
    is_warmup = (dt < start_dt or i < warmup_bars)
    print(f"bar {i}: dt={dt}  dt<start_dt={dt < start_dt}  i<{warmup_bars}={i < warmup_bars}  is_warmup={is_warmup}")
