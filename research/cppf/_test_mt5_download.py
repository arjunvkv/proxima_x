"""Debug MT5 data download - try different methods."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import pandas as pd
import datetime
import MetaTrader5 as mt5

if not mt5.initialize():
    print("init failed")
    sys.exit(1)

pair = "GBPNZD"
print(f"Initialize status: {mt5.initialize()}")

# Method 1: copy_rates_from with now timestamp
now_local = int(datetime.datetime.now().timestamp())
print(f"\n1. copy_rates_from with now_local={now_local} ({datetime.datetime.fromtimestamp(now_local)})")
r = mt5.copy_rates_from(pair, mt5.TIMEFRAME_M1, now_local, 5)
if r is not None:
    for row in r:
        dt = datetime.datetime.fromtimestamp(row[0])
        print(f"   {dt} O={row[1]:.5f} H={row[2]:.5f} L={row[3]:.5f} C={row[4]:.5f}")
else:
    print(f"   NO DATA (error={mt5.last_error()})")

# Method 2: copy_rates_from with now in UTC
now_utc = int(datetime.datetime.utcnow().timestamp())
print(f"\n2. copy_rates_from with now_utc={now_utc} ({datetime.datetime.fromtimestamp(now_utc)})")
r = mt5.copy_rates_from(pair, mt5.TIMEFRAME_M1, now_utc, 5)
if r is not None:
    for row in r:
        dt = datetime.datetime.fromtimestamp(row[0])
        print(f"   {dt} O={row[1]:.5f} H={row[2]:.5f} L={row[3]:.5f} C={row[4]:.5f}")
else:
    print(f"   NO DATA (error={mt5.last_error()})")

# Method 3: Use copy_rates_from to get first actual bar
print(f"\n3. copy_rates_from with known valid timestamp (mid-2026)")
mid_ts = 1782000000  # ~June 2026
r = mt5.copy_rates_from(pair, mt5.TIMEFRAME_M1, mid_ts, 5)
if r is not None:
    for row in r:
        dt = datetime.datetime.fromtimestamp(row[0])
        print(f"   {dt} O={row[1]:.5f} H={row[2]:.5f} L={row[3]:.5f} C={row[4]:.5f}")
else:
    print(f"   NO DATA (error={mt5.last_error()})")

# Method 4: copy_rates_from_pos to get last bar
print(f"\n4. copy_rates_from_pos, pos=0, count=5")
r = mt5.copy_rates_from_pos(pair, mt5.TIMEFRAME_M1, 0, 5)
if r is not None:
    for row in r:
        dt = datetime.datetime.fromtimestamp(row[0])
        print(f"   {dt} O={row[1]:.5f} H={row[2]:.5f} L={row[3]:.5f} C={row[4]:.5f}")
else:
    print(f"   NO DATA (error={mt5.last_error()})")

# Method 5: copy_rates_range
print(f"\n5. copy_rates_range")
from_ts = now_utc - 3600  # 1 hour ago
to_ts = now_utc
r = mt5.copy_rates_range(pair, mt5.TIMEFRAME_M1, from_ts, to_ts)
if r is not None:
    print(f"   Got {len(r)} bars from {datetime.datetime.fromtimestamp(r[0][0])} to {datetime.datetime.fromtimestamp(r[-1][0])}")
else:
    print(f"   NO DATA (error={mt5.last_error()})")

# Method 6: Check how many bars available with different count
print(f"\n6. Testing max count from mid-2026")
for count in [100, 1000, 10000, 50000, 100000]:
    r = mt5.copy_rates_from(pair, mt5.TIMEFRAME_M1, mid_ts, count)
    if r is not None and len(r) > 0:
        dt_first = datetime.datetime.fromtimestamp(r[0][0])
        dt_last = datetime.datetime.fromtimestamp(r[-1][0])
        print(f"   count={count}: {len(r)} bars, {dt_first} to {dt_last}")
    else:
        print(f"   count={count}: NO DATA")

mt5.shutdown()
