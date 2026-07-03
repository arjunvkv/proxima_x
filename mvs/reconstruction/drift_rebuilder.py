from __future__ import annotations

from typing import Dict
from collections import deque
import numpy as np


class DriftRebuilder:
    __slots__ = ("window", "ref_buffer", "live_buffer")

    def __init__(self, window: int = 90) -> None:
        self.window = window
        self.ref_buffer = deque(maxlen=window)
        self.live_buffer = deque(maxlen=window)

    def _ks_statistic(self, a: np.ndarray, b: np.ndarray) -> float:
        if len(a) < 5 or len(b) < 5:
            return 0.0
        combined = np.concatenate([a, b])
        a_sorted = np.sort(a)
        b_sorted = np.sort(b)
        all_sorted = np.sort(combined)
        cdf_a = np.searchsorted(a_sorted, all_sorted) / len(a)
        cdf_b = np.searchsorted(b_sorted, all_sorted) / len(b)
        return float(np.max(np.abs(cdf_a - cdf_b)))

    def on_tick(self, tick_id: int, symbol: str, ts_ns: int, mid: float) -> Dict:
        self.live_buffer.append(mid)

        if len(self.ref_buffer) < self.window:
            self.ref_buffer.append(mid)
            return {"drift_score": 0.0, "drift_flag": False}

        if len(self.ref_buffer) == self.window:
            self.ref_buffer.popleft()
            self.ref_buffer.append(mid)

        ref_arr = np.array(self.ref_buffer, dtype=np.float64)
        live_arr = np.array(self.live_buffer, dtype=np.float64)

        drift_score = self._ks_statistic(ref_arr, live_arr)

        return {"drift_score": float(drift_score), "drift_flag": bool(drift_score > 0.15)}
