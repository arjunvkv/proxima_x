from __future__ import annotations

from collections import deque
from typing import Dict

import numpy as np

from layer7.entropy_compression import EntropyCompressionEngine
from mvs.utils.vector_ops import shannon_entropy


class EntropyRebuilder:
    __slots__ = ("engine", "window", "prices", "_last_entropy")

    def __init__(self, window: int = 64) -> None:
        self.engine = EntropyCompressionEngine()
        self.window = window
        self.prices = deque(maxlen=window)
        self._last_entropy = 0.0

    def _bucketize(self) -> np.ndarray:
        if len(self.prices) < 2:
            return np.array([1.0, 1.0, 1.0], dtype=np.float64)
        up = 0; down = 0; flat = 0
        arr = np.array(self.prices, dtype=np.float64)
        for i in range(1, len(arr)):
            d = arr[i] - arr[i - 1]
            if d > 0:
                up += 1
            elif d < 0:
                down += 1
            else:
                flat += 1
        return np.array([up, down, flat], dtype=np.float64)

    def on_tick(self, tick_id: int, symbol: str, ts_ns: int, mid: float) -> Dict[str, float]:
        self.prices.append(mid)
        buckets = self._bucketize()
        entropy = float(shannon_entropy(buckets))
        d_entropy = entropy - self._last_entropy
        arr = np.array(self.prices, dtype=np.float64)
        if len(arr) > 1:
            compression_ratio = np.std(arr) / max(np.mean(np.abs(np.diff(arr))), 1e-9)
        else:
            compression_ratio = 0.0
        burst_density = float(np.sum(np.abs(np.diff(arr)) > np.std(arr))) if len(arr) > 2 else 0.0
        self._last_entropy = entropy
        return {
            "entropy": entropy,
            "d_entropy": d_entropy,
            "compression_ratio": compression_ratio,
            "burst_density": burst_density,
        }
