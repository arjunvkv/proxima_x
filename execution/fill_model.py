"""Fill probability model — simulates partial fills and rejections."""
import numpy as np
from typing import Optional


class FillModel:
    def __init__(self, seed: Optional[int] = None):
        self._rng = np.random.default_rng(seed)

    def should_fill(self, spread: float, latency_ms: float,
                    queue_prob: float, spread_threshold: float = 0.005) -> bool:
        if spread > spread_threshold:
            return False
        base = queue_prob * np.exp(-latency_ms / 180.0)
        return float(self._rng.uniform()) < base

    def fill_ratio(self, queue_prob: float) -> float:
        return max(0.3, min(1.0, float(self._rng.uniform(0.3, 1.0) * queue_prob)))

    def should_reject(self, spread: float, spread_threshold: float = 0.005) -> bool:
        return spread > spread_threshold
