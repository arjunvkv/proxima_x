from __future__ import annotations

from typing import Optional

import numpy as np
import numba
from numpy.typing import NDArray

from config.settings import settings
from core.event_engine import EventEngine, Event, EventType


@numba.jit(nopython=True, cache=True)
def _ewm_abs_returns(abs_returns: NDArray[np.float64], window: int) -> NDArray[np.float32]:
    n = len(abs_returns)
    result = np.zeros(n, dtype=np.float32)
    decay = np.exp(-np.arange(window, dtype=np.float64) / (window / 5.0))
    decay = decay / np.sum(decay)
    for i in range(window - 1, n):
        s = 0.0
        for j in range(window):
            s += abs_returns[i - j] * decay[window - 1 - j]
        result[i] = s
    return result


@numba.jit(nopython=True, cache=True)
def _rolling_density(prices: NDArray[np.float64], window: int, bins: int) -> NDArray[np.float32]:
    n = len(prices)
    result = np.zeros(n, dtype=np.float32)
    for i in range(window - 1, n):
        seg = prices[i - window + 1 : i + 1]
        lo = seg.min()
        hi = seg.max()
        if hi - lo < 1e-10:
            result[i] = 1.0
            continue
        bin_edges = np.linspace(lo, hi, bins + 1)
        counts = np.zeros(bins, dtype=np.int64)
        for j in range(window):
            for b in range(bins):
                if bin_edges[b] <= seg[j] < bin_edges[b + 1]:
                    counts[b] += 1
                    break
        result[i] = counts.max() / window
    return result


@numba.jit(nopython=True, cache=True)
def _rolling_alignment(returns: NDArray[np.float64], memory: NDArray[np.float32], window: int) -> NDArray[np.float32]:
    n = len(returns)
    result = np.zeros(n, dtype=np.float32)
    for i in range(window - 1, n):
        r = returns[i - window + 1 : i + 1]
        m = memory[i - window + 1 : i + 1]
        r_mean = np.mean(r)
        m_mean = np.mean(m)
        num = 0.0
        den_r = 0.0
        den_m = 0.0
        for j in range(window):
            rd = r[j] - r_mean
            md = m[j] - m_mean
            num += rd * md
            den_r += rd * rd
            den_m += md * md
        d = np.sqrt(den_r * den_m)
        if d > 1e-10:
            result[i] = num / d
    return result


@numba.jit(nopython=True, cache=True)
def _rolling_conflict(prices: NDArray[np.float64], window: int, bins: int, eps: float) -> NDArray[np.float32]:
    n = len(prices)
    result = np.zeros(n, dtype=np.float32)
    returns = np.diff(prices)
    for i in range(window, n):
        seg = prices[i - window + 1 : i + 1]
        ret_seg = returns[i - window : i]
        lo = seg.min()
        hi = seg.max()
        if hi - lo < 1e-10:
            result[i] = 0.0
            continue
        bin_edges = np.linspace(lo, hi, bins + 1)
        bin_dirs = np.zeros(bins, dtype=np.float64)
        bin_counts = np.zeros(bins, dtype=np.int64)
        for j in range(window - 1):
            for b in range(bins):
                if bin_edges[b] <= seg[j] < bin_edges[b + 1]:
                    bin_dirs[b] += np.sign(ret_seg[j])
                    bin_counts[b] += 1
                    break
        conflict = 0.0
        for b in range(bins):
            if bin_counts[b] > 1:
                avg_dir = bin_dirs[b] / bin_counts[b]
                conflict += 1.0 - abs(avg_dir)
        result[i] = conflict / bins
    return result


class MemoryFieldResearch:
    def __init__(self, event_engine: Optional[EventEngine] = None, max_memory: int = settings.research.min_memory_samples * 100):
        self.event_engine = event_engine
        self.max_memory = max_memory

    def compute_memory_strength(self, price_series: NDArray[np.float64], window: int = 100) -> NDArray[np.float32]:
        abs_returns = np.abs(np.diff(price_series, prepend=price_series[0:1]))
        return _ewm_abs_returns(abs_returns.astype(np.float64), min(window, len(price_series)))

    def compute_memory_density(self, price_series: NDArray[np.float64], window: int = 100, bin_count: int = 50) -> NDArray[np.float32]:
        return _rolling_density(price_series.astype(np.float64), min(window, len(price_series)), bin_count)

    def compute_memory_decay_rate(self, price_series: NDArray[np.float64], half_life: int = settings.research.echo_decay_half_life) -> NDArray[np.float32]:
        n = len(price_series)
        decay = 0.5 ** (np.arange(n, dtype=np.float64) / half_life)
        return decay.astype(np.float32)

    def compute_memory_alignment(self, returns: NDArray[np.float64], memory_strength: NDArray[np.float32], window: int = 100) -> NDArray[np.float32]:
        min_len = min(len(returns), len(memory_strength))
        r = returns[:min_len].astype(np.float64)
        m = memory_strength[:min_len]
        return _rolling_alignment(r, m, min(window, min_len))

    def compute_memory_conflict(self, price_series: NDArray[np.float64], window: int = 100) -> NDArray[np.float32]:
        return _rolling_conflict(price_series.astype(np.float64), min(window, len(price_series)), 30, 1e-10)

    def compute_all(self, price_series: NDArray[np.float64], returns: NDArray[np.float64]) -> dict[str, NDArray[np.float32]]:
        strength = self.compute_memory_strength(price_series)
        density = self.compute_memory_density(price_series)
        decay = self.compute_memory_decay_rate(price_series)
        alignment = self.compute_memory_alignment(returns, strength)
        conflict = self.compute_memory_conflict(price_series)
        return {
            "memory_strength": strength,
            "memory_density": density,
            "memory_decay_rate": decay,
            "memory_alignment": alignment,
            "memory_conflict": conflict,
        }

    def emit_events(self, timestamps: list[int], memory_strength: NDArray[np.float32], threshold: float = 2.0) -> None:
        if self.event_engine is None:
            return
        for i in range(len(memory_strength)):
            if memory_strength[i] > threshold:
                self.event_engine.emit(Event(
                    event_type=EventType.MEMORY_FORMATION,
                    timestamp=timestamps[i] if i < len(timestamps) else 0,
                    data={"index": i, "strength": float(memory_strength[i])},
                    source="memory_field",
                    confidence=min(1.0, float(memory_strength[i]) / (threshold * 2)),
                ))
