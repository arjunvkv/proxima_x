"""s4_cache_probe.py — raw cached EURUSD/DXY M5 tail: timestamps + prices.

Settles which side (cache or terminal) is shifted: if the cached EURUSD M5
series' last bars carry real Aug-11 timestamps, the cache is current and the
terminal history is the odd one; if the last cached bars are actually Aug-6
prices stamped as Aug-11, the cache is stale.
"""
import sys
import numpy as np

sys.path.insert(0, "scripts"); sys.path.insert(0, "scripts/_absorb"); sys.path.insert(0, ".")
from proxima_ops.backtest.feed import load_bars_cached
from proxima_ops.nova import feed as nfeed

for sym in ("EURUSD", "DXY.cash"):
    b = nfeed.bars_list_to_arrays(load_bars_cached(sym))
    n = len(b["ts"])
    print(f"== {sym}: n={n} first={b['ts'][0]} last={b['ts'][-1]}")
    # last 6 M5 bars: ts, day, close
    for k in range(n - 6, n):
        ts = int(b["ts"][k])
        import datetime
        dt = datetime.datetime.utcfromtimestamp(ts)
        print(f"   ts={ts} ({dt:%Y-%m-%d %H:%M}Z) close={float(b['close'][k]):.5f}")
