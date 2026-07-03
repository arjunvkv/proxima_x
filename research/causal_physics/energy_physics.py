from __future__ import annotations

from typing import Any, Optional

import numpy as np
from numba import jit

from research.information_discovery.mi_estimator import _fast_conditional_mutual_info
from research.temporal_reality.causality_analysis import AdaptiveTimeCausality


@jit(nopython=True, cache=True)
def _rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    n = len(arr)
    result = np.zeros(n, dtype=np.float64)
    for i in range(n):
        start = 0
        if i >= window:
            start = i - window + 1
        chunk = arr[start:i + 1]
        length = len(chunk)
        if length < 2:
            continue
        mean = 0.0
        for j in range(length):
            mean += chunk[j]
        mean /= length
        var = 0.0
        for j in range(length):
            var += (chunk[j] - mean) ** 2
        var /= length
        result[i] = np.sqrt(var)
    return result


@jit(nopython=True, cache=True)
def _rolling_entropy(arr: np.ndarray, window: int, n_bins: int) -> np.ndarray:
    n = len(arr)
    result = np.zeros(n, dtype=np.float64)
    for i in range(n):
        start = 0
        if i >= window:
            start = i - window + 1
        chunk = arr[start:i + 1]
        length = len(chunk)
        if length < 2:
            continue
        chunk_sorted = np.sort(chunk)
        q = np.linspace(0.0, 1.0, n_bins + 1)
        bins = np.zeros(n_bins + 1)
        for k in range(n_bins + 1):
            idx = q[k] * (length - 1)
            idx_low = int(np.floor(idx))
            idx_high = int(np.ceil(idx))
            if idx_low == idx_high:
                bins[k] = chunk_sorted[idx_low]
            else:
                weight = idx - idx_low
                bins[k] = chunk_sorted[idx_low] * (1.0 - weight) + chunk_sorted[idx_high] * weight
        u = np.unique(bins)
        if len(u) < 2:
            continue
        dig = np.zeros(length, dtype=np.int32)
        for j in range(length):
            val = chunk[j]
            if val <= bins[0]:
                dig[j] = 0
            elif val >= bins[n_bins - 1]:
                dig[j] = n_bins - 2
            else:
                low = 0
                high = n_bins - 2
                best = 0
                while low <= high:
                    mid = (low + high) // 2
                    if bins[mid] <= val:
                        best = mid
                        low = mid + 1
                    else:
                        high = mid - 1
                dig[j] = best
        counts = np.zeros(n_bins, dtype=np.int32)
        for j in range(length):
            counts[dig[j]] += 1
        entropy = 0.0
        for j in range(n_bins):
            if counts[j] > 0:
                p = counts[j] / length
                entropy -= p * np.log(p)
        result[i] = entropy
    return result


@jit(nopython=True, cache=True)
def _rolling_correlation(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
    n = min(len(x), len(y))
    result = np.zeros(n, dtype=np.float64)
    for i in range(n):
        start = 0
        if i >= window:
            start = i - window + 1
        x_seg = x[start:i + 1]
        y_seg = y[start:i + 1]
        length = len(x_seg)
        if length < 2:
            continue
        x_mean = 0.0
        y_mean = 0.0
        for j in range(length):
            x_mean += x_seg[j]
            y_mean += y_seg[j]
        x_mean /= length
        y_mean /= length
        x_var = 0.0
        y_var = 0.0
        cov = 0.0
        for j in range(length):
            dx = x_seg[j] - x_mean
            dy = y_seg[j] - y_mean
            x_var += dx * dx
            y_var += dy * dy
            cov += dx * dy
        x_std = np.sqrt(x_var / length)
        y_std = np.sqrt(y_var / length)
        if x_std == 0.0 or y_std == 0.0:
            result[i] = 0.0
        else:
            result[i] = (cov / length) / (x_std * y_std)
    return result


class EnergyPhysicsAnalyzer:
    """Identify what causal forces generate energy_storage."""

    def __init__(self, max_lag: int = 50, n_bins: int = 20) -> None:
        self.max_lag = max_lag
        self.n_bins = n_bins

    def compute(
        self,
        data: dict,
        price: Optional[np.ndarray] = None,
        returns: Optional[np.ndarray] = None,
    ) -> dict[str, Any]:
        energy_storage = np.asarray(data["energy_storage"], dtype=np.float64)
        adaptive_time = np.asarray(data["adaptive_time"], dtype=np.float64)
        memory_density = np.asarray(data["memory_density"], dtype=np.float64)
        memory_gradient = np.asarray(data["memory_gradient"], dtype=np.float64)
        state_mutation_rate = np.asarray(data["state_mutation_rate"], dtype=np.float64)
        regime_change_probability = np.asarray(data["regime_change_probability"], dtype=np.float64)

        if returns is None and price is not None:
            log_p = np.log(np.asarray(price, dtype=np.float64))
            returns = np.diff(log_p)
            returns = np.concatenate((np.array([0.0]), returns))
        elif returns is not None:
            returns = np.asarray(returns, dtype=np.float64)
        else:
            returns = np.zeros(len(energy_storage), dtype=np.float64)

        compression = _rolling_std(returns, 20)
        rolling_ent = _rolling_entropy(returns, 20, self.n_bins)
        entropy_change = np.zeros_like(rolling_ent)
        if len(rolling_ent) > 1:
            entropy_change[1:] = np.abs(np.diff(rolling_ent))
        memory_alignment = _rolling_correlation(memory_density, memory_gradient, 20)

        candidates = {
            "adaptive_time": adaptive_time,
            "memory_density": memory_density,
            "memory_gradient": memory_gradient,
            "state_mutation_rate": state_mutation_rate,
            "regime_change_probability": regime_change_probability,
            "compression": compression,
            "entropy_change": entropy_change,
            "memory_alignment": memory_alignment,
        }

        target = energy_storage
        generators = []

        for var_name, candidate_arr in candidates.items():
            corr = AdaptiveTimeCausality._cross_correlate(
                candidate_arr, target, self.max_lag
            )
            lags = np.arange(-self.max_lag, self.max_lag + 1, dtype=np.intp)
            peak_idx = int(np.argmax(np.abs(corr)))
            peak_lag = int(lags[peak_idx])
            peak_corr = float(corr[peak_idx])

            min_len = min(len(candidate_arr), len(target))
            c = candidate_arr[:min_len]
            t = target[:min_len]

            info_flow = _fast_conditional_mutual_info(
                c[:-1], t[1:], t[:-1], self.n_bins
            )
            te = _fast_conditional_mutual_info(
                t[1:], c[:-1], t[:-1], self.n_bins
            )

            item = {
                "variable": var_name,
                "peak_lag": peak_lag,
                "peak_corr": peak_corr,
                "transfer_entropy": float(te),
                "causal_strength": float(info_flow),
            }

            if peak_lag < 0:
                generators.append(item)

        primary_generator = None
        if generators:
            primary_generator = max(generators, key=lambda g: g["causal_strength"])

        energy_creation_chain = sorted(generators, key=lambda g: g["peak_lag"])

        return {
            "generators": generators,
            "primary_generator": primary_generator["variable"] if primary_generator else None,
            "energy_creation_chain": [g["variable"] for g in energy_creation_chain],
        }
