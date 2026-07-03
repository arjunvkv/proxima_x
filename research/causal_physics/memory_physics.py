from __future__ import annotations

from typing import Any

import numpy as np
import numba
from numpy.typing import NDArray

from research.temporal_reality.causality_analysis import AdaptiveTimeCausality


@numba.jit(nopython=True, cache=True)
def _find_birth_events(density: NDArray[np.float64], threshold: float) -> NDArray[np.int32]:
    n = len(density)
    result = np.zeros(n, dtype=np.int32)
    above = density > threshold
    for i in range(1, n):
        if above[i] and not above[i - 1]:
            result[i] = 1
    return result


@numba.jit(nopython=True, cache=True)
def _rolling_mean_diff(diff: NDArray[np.float64], window: int, positive: bool) -> NDArray[np.float64]:
    n = len(diff)
    result = np.zeros(n, dtype=np.float64)
    for i in range(window, n):
        s = 0.0
        c = 0
        for j in range(i - window, i):
            d = diff[j]
            if positive and d > 0.0:
                s += d
                c += 1
            elif not positive and d < 0.0:
                s += abs(d)
                c += 1
        if c > 0:
            result[i] = s / c
    return result


@numba.jit(nopython=True, cache=True)
def _find_reset_events(density: NDArray[np.float64], threshold: float) -> int:
    n = len(density)
    count = 0
    for i in range(1, n):
        drop = density[i - 1] - density[i]
        if drop > threshold:
            count += 1
    return count


@numba.jit(nopython=True, cache=True)
def _binary_from_indices(signal: NDArray[np.int32], n: int) -> NDArray[np.float64]:
    result = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if signal[i] > 0:
            result[i] = 1.0
    return result


class MemoryPhysicsAnalyzer:
    def __init__(self, max_lag: int = 100) -> None:
        self.max_lag = max_lag
        self._causality = AdaptiveTimeCausality(max_lag=max_lag)

    def compute(self, data: dict[str, np.ndarray]) -> dict[str, Any]:
        density = np.asarray(data["memory_density"], dtype=np.float64)
        gradient = np.asarray(data["memory_gradient"], dtype=np.float64)
        n = len(density)

        threshold = float(np.nanmean(density) + 0.5 * np.nanstd(density))
        drop_threshold = float(np.nanstd(density))

        birth_binary = _find_birth_events(density, threshold)
        birth_events_count = int(np.sum(birth_binary))

        diff = np.diff(density)
        diff_padded = np.zeros(n, dtype=np.float64)
        if n > 1:
            diff_padded[1:] = diff

        growth = _rolling_mean_diff(diff_padded, 10, True)
        decay = _rolling_mean_diff(diff_padded, 10, False)

        reset_events_count = int(_find_reset_events(density, drop_threshold))
        decay_rate_mean = float(np.nanmean(decay))
        growth_rate_mean = float(np.nanmean(growth))

        candidates = {
            "adaptive_time": np.asarray(data.get("adaptive_time", np.zeros(n)), dtype=np.float64),
            "energy_storage": np.asarray(data.get("energy_storage", np.zeros(n)), dtype=np.float64),
            "memory_gradient": gradient,
            "state_mutation_rate": np.asarray(data.get("state_mutation_rate", np.zeros(n)), dtype=np.float64),
        }

        formation_result = self._lead_lag_formation(birth_binary, candidates)
        decay_result = self._lead_lag_decay(decay, candidates)

        return {
            "memory_formation": formation_result,
            "memory_decay": decay_result,
            "birth_events_count": birth_events_count,
            "reset_events_count": reset_events_count,
            "decay_rate_mean": decay_rate_mean,
            "growth_rate_mean": growth_rate_mean,
        }

    def _lead_lag_formation(
        self, birth_binary: NDArray[np.int32], candidates: dict[str, NDArray[np.float64]]
    ) -> dict[str, Any]:
        n = len(birth_binary)
        best_lag = 0
        best_corr = -1.0
        best_gen = "none"
        birth_signal = _binary_from_indices(birth_binary, n)

        for name, sig in candidates.items():
            corr = AdaptiveTimeCausality._cross_correlate(birth_signal, sig, self.max_lag)
            lags = np.arange(-self.max_lag, self.max_lag + 1, dtype=np.intp)
            peak_idx = int(np.argmax(np.abs(corr)))
            pl = int(lags[peak_idx])
            pc = float(corr[peak_idx])
            if abs(pc) > abs(best_corr):
                best_corr = pc
                best_lag = pl
                best_gen = name

        return {"peak_lag": best_lag, "peak_corr": best_corr, "primary_generator": best_gen}

    def _lead_lag_decay(
        self, decay_rate: NDArray[np.float64], candidates: dict[str, NDArray[np.float64]]
    ) -> dict[str, Any]:
        best_lag = 0
        best_corr = -1.0
        best_gen = "none"

        for name, sig in candidates.items():
            corr = AdaptiveTimeCausality._cross_correlate(decay_rate, sig, self.max_lag)
            lags = np.arange(-self.max_lag, self.max_lag + 1, dtype=np.intp)
            peak_idx = int(np.argmax(np.abs(corr)))
            pl = int(lags[peak_idx])
            pc = float(corr[peak_idx])
            if abs(pc) > abs(best_corr):
                best_corr = pc
                best_lag = pl
                best_gen = name

        return {"peak_lag": best_lag, "peak_corr": best_corr, "primary_generator": best_gen}
