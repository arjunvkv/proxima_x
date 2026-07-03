from __future__ import annotations

from typing import Any

import numpy as np
import numba
from numpy.typing import NDArray

from research.memory_physics.memory_validator import MemoryValidator, MPRResult
from research.temporal_reality.causality_analysis import AdaptiveTimeCausality


@numba.jit(nopython=True, cache=True)
def _numba_rolling_std(arr: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    n = len(arr)
    result = np.zeros(n, dtype=np.float64)
    for i in range(window, n):
        s = 0.0
        for j in range(i - window, i):
            s += arr[j]
        mean = s / window
        var = 0.0
        for j in range(i - window, i):
            var += (arr[j] - mean) ** 2
        result[i] = np.sqrt(var / window)
    return result


@numba.jit(nopython=True, cache=True)
def _find_birth_events(density: NDArray[np.float64], threshold: float) -> NDArray[np.int32]:
    n = len(density)
    result = np.zeros(n, dtype=np.int32)
    above = density > threshold
    for i in range(1, n):
        if above[i] and not above[i - 1]:
            result[i] = 1
    return result


class MemoryBirthModel:
    """RQ5: What creates memory density?

    Investigate whether energy_storage, compression, or memory_conflict
    generate memory density. Estimate birth/growth/decay/reset rates.
    """

    def __init__(self, validator: MemoryValidator, asset: str = "EURJPY"):
        self.validator = validator
        self.asset = asset
        self._max_lag = 200

    def run(self) -> MPRResult:
        data = self.validator.load_asset_data(self.asset)
        signals = self.validator.compute_signals(data)

        density = np.asarray(signals["memory_density"], dtype=np.float64)
        conflict = np.asarray(signals["memory_conflict"], dtype=np.float64)
        es = np.asarray(signals["energy_storage"], dtype=np.float64)
        ec = np.asarray(signals["energy_creation"], dtype=np.float64)
        returns = np.asarray(signals["returns"], dtype=np.float64)
        n = len(density)

        compression = _numba_rolling_std(returns, 20)

        # Birth event detection
        threshold = float(np.nanmean(density) + 0.5 * np.nanstd(density))
        drop_threshold = float(np.nanstd(density))
        birth_binary = _find_birth_events(density, threshold)
        birth_rate = float(np.sum(birth_binary)) / max(n, 1)

        # Decay/growth rates
        diff = np.diff(density, prepend=density[0])
        growth_mask = diff > 0
        decay_mask = diff < 0
        growth_rate = float(np.mean(diff[growth_mask])) if np.any(growth_mask) else 0.0
        decay_rate = float(np.mean(np.abs(diff[decay_mask]))) if np.any(decay_mask) else 0.0

        # Reset events
        reset_count = 0
        for i in range(1, n):
            if density[i - 1] - density[i] > drop_threshold:
                reset_count += 1
        reset_rate = reset_count / max(n, 1)

        from research.memory_physics.memory_validator import _find_peak_lag
        lag_ec, r_ec = _find_peak_lag(ec[:n], density[:n], self._max_lag)
        lag_cp, r_cp = _find_peak_lag(compression[:n], density[:n], self._max_lag)
        lag_cf, r_cf = _find_peak_lag(conflict[:n], density[:n], self._max_lag)
        lag_es, r_es = _find_peak_lag(es[:n], density[:n], self._max_lag)

        generators = {
            "energy_creation": (lag_ec, r_ec),
            "compression": (lag_cp, r_cp),
            "memory_conflict": (lag_cf, r_cf),
            "energy_storage": (lag_es, r_es),
        }

        best_gen = max(generators, key=lambda g: abs(generators[g][1]))
        best_lag, best_r = generators[best_gen]

        flow_scores = {}
        for name in generators:
            flow_scores[name] = self.validator.information_flow(name, "memory_density", signals)

        metrics = {
            "birth_rate": birth_rate,
            "growth_rate": growth_rate,
            "decay_rate": decay_rate,
            "reset_rate": reset_rate,
            "generator_rankings": {name: {"peak_lag": l, "peak_corr": r}
                                   for name, (l, r) in generators.items()},
            "information_flow_rankings": flow_scores,
            "primary_generator": best_gen,
            "primary_lag": best_lag,
            "primary_corr": best_r,
            "n_birth_events": int(np.sum(birth_binary)),
            "n_reset_events": reset_count,
        }

        print(f"  Memory birth components:")
        for name, (l, r) in sorted(generators.items(), key=lambda x: abs(x[1][1]), reverse=True):
            print(f"    {name}: lag={l}, corr={r:.4f}")
        print(f"  Primary generator: {best_gen} (lag={best_lag}, corr={best_r:.4f})")
        print(f"  Birth rate: {birth_rate:.4f}, Growth: {growth_rate:.6f}, Decay: {decay_rate:.6f}, Reset: {reset_rate:.4f}")

        return MPRResult("memory_birth_model", "PASSED" if abs(best_r) > 0.1 else "INCONCLUSIVE",
                         metrics=metrics)
