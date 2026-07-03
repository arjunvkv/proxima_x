"""Temporal Compression Layer — Phase III.
Detects compressed directional bursts (clusters of same-direction signals).

Components:
  CompressionWindow — rolling burst tracker
  VolatilityExpansionTracker — vol expansion detection
"""
from collections import deque
import numpy as np


class CompressionWindow:
    """Tracks rolling signal bursts.

    compression_ratio = same_dir_count / window_size
    density = nonzero_signals / window_size
    """
    def __init__(self, window=8, min_same_dir=3, min_compression=0.60, min_density=0.50):
        self.window = window
        self.min_same_dir = min_same_dir
        self.min_compression = min_compression
        self.min_density = min_density
        self._buffer = deque(maxlen=window)

    def update(self, signal, confidence=1.0, price=None):
        self._buffer.append((signal, confidence, price))

    def same_dir_count(self, direction):
        return sum(1 for s, c, p in self._buffer if s == direction)

    def compression_ratio(self, direction):
        if not self._buffer:
            return 0.0
        return self.same_dir_count(direction) / len(self._buffer)

    def density(self):
        if not self._buffer:
            return 0.0
        non_zero = sum(1 for s, c, p in self._buffer if s != 0)
        return non_zero / len(self._buffer)

    def dominant_direction(self):
        if not self._buffer:
            return 0
        net = sum(s for s, c, p in self._buffer)
        return 1 if net > 0 else (-1 if net < 0 else 0)

    def has_burst(self, direction):
        count = self.same_dir_count(direction)
        return (count >= self.min_same_dir
                and self.compression_ratio(direction) >= self.min_compression
                and self.density() >= self.min_density)

    def clear(self):
        self._buffer.clear()


class VolatilityExpansionTracker:
    """Tracks short-term vs long-term realized vol ratio.

    vol_ratio = short_vol / long_vol
    is_expanding if vol_ratio > threshold.
    """
    def __init__(self, short_window=5, long_window=20, expansion_threshold=1.25):
        self.short_window = short_window
        self.long_window = long_window
        self.expansion_threshold = expansion_threshold
        self._prices_short = deque(maxlen=short_window + 1)
        self._prices_long = deque(maxlen=long_window + 1)

    def update(self, price):
        self._prices_short.append(price)
        self._prices_long.append(price)

    def _compute_vol(self, prices):
        if len(prices) < 2:
            return 0.0
        arr = np.array(prices)
        returns = np.abs(np.diff(arr) / arr[:-1])
        return float(np.std(returns)) if len(returns) > 0 else 0.0

    def short_vol(self):
        return self._compute_vol(list(self._prices_short))

    def long_vol(self):
        return self._compute_vol(list(self._prices_long))

    def vol_ratio(self):
        sv = self.short_vol()
        lv = self.long_vol()
        if lv == 0:
            return 1.0
        return sv / lv

    def is_expanding(self):
        return self.vol_ratio() >= self.expansion_threshold

    def reset(self):
        self._prices_short.clear()
        self._prices_long.clear()
