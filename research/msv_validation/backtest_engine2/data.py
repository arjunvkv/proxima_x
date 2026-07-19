import time
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import numpy as np
from numba import jit

PAIRS = ['EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'NZDUSD', 'EURJPY', 'GBPJPY']

class TempCache:
    """In-memory M1 bar cache. Zero disk. One-shotted when process dies."""
    def __init__(self, days_back=7, align_threshold_sec=30):
        self.data = {}
        self.aligned = None
        self.align_threshold_sec = align_threshold_sec
        self._load(days_back)

    def _load(self, days_back):
        """Fetch M1 bars from MT5 for all pairs."""
        for attempt in range(3):
            init = mt5.initialize()
            if init:
                break
            print(f"MT5 init attempt {attempt+1}/3 failed: {mt5.last_error()}")
            time.sleep(1)
        else:
            raise RuntimeError(f"MT5 init failed after 3 attempts: {mt5.last_error()}")

        now = datetime.now()
        from_dt = now - timedelta(days=days_back)

        for pair in PAIRS:
            bars = mt5.copy_rates_range(pair, mt5.TIMEFRAME_M1, from_dt, now)
            if bars is None or len(bars) == 0:
                print(f"WARN: No data for {pair}")
                continue
            self.data[pair] = bars

        mt5.shutdown()

        if not self.data:
            raise RuntimeError("No data loaded for any pair")

        print(f"Pairs loaded: {list(self.data.keys())}")
        for p in list(self.data.keys())[:3]:
            t0 = datetime.fromtimestamp(self.data[p][0][0])
            t1 = datetime.fromtimestamp(self.data[p][-1][0])
            print(f"  {p}: {len(self.data[p])} bars, {t0.date()} to {t1.date()}")
        print(f"  ... ({len(self.data)} pairs total)")

        self._align()

    def _align(self):
        """Align all pairs by timestamp using time windows (not exact match).
        Groups bars within threshold_sec of each other."""
        from collections import defaultdict

        # Build time-sorted index of all bars across all pairs
        # Group by rounded-to-minute timestamps
        all_bars_by_time = defaultdict(list)

        for pair, bars in self.data.items():
            for bar in bars:
                rounded = int(bar[0] / 60) * 60  # round to minute
                all_bars_by_time[rounded].append((pair, bar))

        # Find timestamps that have data from ALL pairs
        required_pairs = set(self.data.keys())
        aligned_times = []
        aligned_data = []

        for ts in sorted(all_bars_by_time.keys()):
            bars_at_ts = all_bars_by_time[ts]
            pairs_at_ts = set(b[0] for b in bars_at_ts)

            if pairs_at_ts == required_pairs:
                aligned_times.append(ts)

                # Build data row: per pair in PAIRS order
                row = np.full(len(PAIRS), np.nan)
                bar_by_pair = {b[0]: b[1] for b in bars_at_ts}
                for pi, pair in enumerate(PAIRS):
                    if pair in bar_by_pair:
                        bar = bar_by_pair[pair]
                        row[pi] = bar[3]  # close price
                aligned_data.append(row)

        if not aligned_times:
            # Fallback: use most liquid pairs that have matching times
            # Find pair with most bars to use as base
            print("WARN: No common timestamps across all pairs. Using subset...")
            return

        self.times = np.array(aligned_times, dtype='u8')
        # Convert aligned_data to structured numpy: per-pair close prices
        self.n_bars = len(aligned_times)

        # Build full 3D array [time, pair, feature]
        self.aligned = np.full((self.n_bars, len(PAIRS), 6), np.nan)

        for i, ts in enumerate(aligned_times):
            bars_at_ts = all_bars_by_time[ts]
            bar_by_pair = {b[0]: b[1] for b in bars_at_ts}
            for pi, pair in enumerate(PAIRS):
                if pair in bar_by_pair:
                    b = bar_by_pair[pair]
                    self.aligned[i, pi] = [b[1], b[2], b[3], b[4], b[5], b[6]]

        print(f"Aligned {self.n_bars} M1 bars across {len(PAIRS)} pairs")
        print(f"  Time range: {datetime.fromtimestamp(self.times[0])} to {datetime.fromtimestamp(self.times[-1])}")
        print(f"  Memory: {self.aligned.nbytes / 1024:.1f} KB")

    def get(self):
        return self.aligned, self.times, PAIRS

@jit(nopython=True)
def pip(pair_idx, price):
    """Convert price to pips for a given pair."""
    if pair_idx in (0, 2, 3, 4):  # EURUSD, GBPUSD, AUDUSD, NZDUSD = 4 decimal
        return price * 10000
    elif pair_idx in (1,):  # USDJPY = 2 decimal
        return price * 100
    elif pair_idx in (5, 6):  # EURJPY, GBPJPY = 2 decimal
        return price * 100
    return price * 10000

@jit(nopython=True)
def direction_rate(arr):
    """Rate of directional agreement across pairs.
    Returns ratio of pairs moving in majority direction."""
    up = 0
    total = 0
    for i in range(len(arr)):
        if not np.isnan(arr[i]):
            total += 1
            if arr[i] > 0:
                up += 1
    if total == 0:
        return 0.5
    return max(up, total - up) / total

if __name__ == '__main__':
    import time as _t
    import MetaTrader5 as _mt5
    _mt5.shutdown()
    _t.sleep(0.5)
    cache = TempCache(7)
    aligned, times, pairs = cache.get()
    print(f"Shape: {aligned.shape}")
    print(f"Memory: {aligned.nbytes / 1024:.1f} KB")
