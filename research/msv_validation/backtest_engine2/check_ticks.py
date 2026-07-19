"""Check tick availability and measure tick rates via polling."""
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import time
import numpy as np

PAIRS = ['EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'NZDUSD', 'EURJPY', 'GBPJPY']

for attempt in range(3):
    init = mt5.initialize()
    if init:
        break
    time.sleep(1)

now_str = datetime.now().strftime("%H:%M:%S")
print(f"Measuring tick arrival rate via 10s polling...")
print(f"Current time: {now_str}")
print()

tick_counts = {p: 0 for p in PAIRS}
last_bid = {p: None for p in PAIRS}
changes = {p: 0 for p in PAIRS}

start = time.time()
while time.time() - start < 10:
    for pair in PAIRS:
        tick = mt5.symbol_info_tick(pair)
        if tick:
            tick_counts[pair] += 1
            if last_bid[pair] is not None and tick.bid != last_bid[pair]:
                changes[pair] += 1
            last_bid[pair] = tick.bid
    time.sleep(0.001)

elapsed = time.time() - start
print(f"Over {elapsed:.1f}s of polling (1ms intervals):")
for p in PAIRS:
    rate = tick_counts[p] / elapsed
    chg_rate = changes[p] / elapsed
    print(f"  {p}: {tick_counts[p]} polls ({rate:.0f}/s), {changes[p]} bid changes ({chg_rate:.1f}/s)")

mt5.shutdown()
