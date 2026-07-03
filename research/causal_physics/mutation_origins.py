from __future__ import annotations

from typing import Any

import numpy as np
from numba import jit

from research.information_discovery.mi_estimator import _fast_conditional_mutual_info
from research.temporal_reality.causality_analysis import AdaptiveTimeCausality


@jit(nopython=True, cache=True)
def _rolling_unique_count(arr: np.ndarray, window: int) -> np.ndarray:
    n = len(arr)
    result = np.zeros(n, dtype=np.float64)
    for i in range(n):
        start = 0
        if i >= window:
            start = i - window + 1
        chunk = arr[start:i + 1].copy()
        chunk.sort()
        unique = 1 if len(chunk) > 0 else 0
        for j in range(1, len(chunk)):
            if chunk[j] != chunk[j - 1]:
                unique += 1
        result[i] = float(unique)
    return result


class MutationOriginsAnalyzer:
    """Identify what causal forces generate state_mutation_rate."""

    def __init__(self, max_lag: int = 50, n_bins: int = 20) -> None:
        self.max_lag = max_lag
        self.n_bins = n_bins

    def compute(self, data: dict) -> dict[str, Any]:
        mutation_rate = np.asarray(data["state_mutation_rate"], dtype=np.float64)
        energy_storage = np.asarray(data["energy_storage"], dtype=np.float64)
        memory_density = np.asarray(data["memory_density"], dtype=np.float64)
        memory_gradient = np.asarray(data["memory_gradient"], dtype=np.float64)
        adaptive_time = np.asarray(data["adaptive_time"], dtype=np.float64)
        regime_change_probability = np.asarray(data["regime_change_probability"], dtype=np.float64)

        memory_conflict = np.abs(memory_density - memory_gradient)
        behavior_density = _rolling_unique_count(mutation_rate, 20)

        candidates = {
            "energy_storage": energy_storage,
            "memory_density": memory_density,
            "memory_gradient": memory_gradient,
            "adaptive_time": adaptive_time,
            "regime_change_probability": regime_change_probability,
            "memory_conflict": memory_conflict,
            "behavior_density": behavior_density,
        }

        target = mutation_rate
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

            if peak_lag < 0 and abs(peak_corr) > 0.1:
                generators.append(item)

        primary_generator = None
        if generators:
            primary_generator = max(generators, key=lambda g: g["causal_strength"])

        generator_chain = sorted(generators, key=lambda g: g["peak_lag"])

        return {
            "generators": generators,
            "primary_generator": primary_generator["variable"] if primary_generator else None,
            "generator_chain": [g["variable"] for g in generator_chain],
        }
