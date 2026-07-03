"""RQ7: Generator tournament v2 - head-to-head compression vs all other generators."""

from __future__ import annotations

from typing import Any

import numpy as np
import numba
from numpy.typing import NDArray

from research.compression_physics.compression_validator import CompressionValidator, CPIResult


COMPETITORS = [
    "compression", "energy_storage", "memory_density", "memory_gradient",
    "tension_score", "entropy_change", "liquidity_entropy",
    "information_pressure", "cohort_alignment", "behavior_density",
    "memory_alignment", "cohort_conflict",
]


@numba.jit(nopython=True, cache=True)
def _score_competitor(dep_var: NDArray[np.float64], candidate: NDArray[np.float64],
                      max_lag: int) -> tuple[float, int]:
    n = min(len(dep_var), len(candidate)) - max_lag
    if n < 10:
        return 0.0, 0
    best = 0.0
    best_lag = 0
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            a = candidate[-lag:n]
            b = dep_var[:n + lag]
        elif lag > 0:
            a = candidate[:n - lag]
            b = dep_var[lag:n]
        else:
            a = candidate[:n]
            b = dep_var[:n]
        v = min(len(a), len(b))
        if v < 10:
            continue
        xf, yf = a[:v], b[:v]
        mx, my = 0.0, 0.0
        for i in range(v):
            mx += xf[i]
            my += yf[i]
        mx /= v
        my /= v
        num, den_x, den_y = 0.0, 0.0, 0.0
        for i in range(v):
            dx = xf[i] - mx
            dy = yf[i] - my
            num += dx * dy
            den_x += dx * dx
            den_y += dy * dy
        r = num / (np.sqrt(den_x) * np.sqrt(den_y) + 1e-12)
        if abs(r) > abs(best):
            best = r
            best_lag = lag
    return best, best_lag


class GeneratorTournament:
    def __init__(self, validator: CompressionValidator, asset: str = "EURJPY"):
        self.validator = validator
        self.asset = asset
        self._max_lag = 200

    def run(self) -> CPIResult:
        data = self.validator.load_asset_data(self.asset)
        signals = self.validator.compute_signals(data)

        adaptive_time = np.asarray(signals.get("adaptive_time", signals.get("state_mutation_rate", np.zeros(100))), dtype=np.float64)
        n = len(adaptive_time)

        scores = []
        for comp in COMPETITORS:
            if comp not in signals:
                continue
            signal = np.asarray(signals[comp], dtype=np.float64)
            if len(signal) < self._max_lag * 2 + 1:
                continue
            lag_adjust = 0
            sig = signal[:max(len(signal), len(adaptive_time))]
            at = adaptive_time[:max(len(signal), len(adaptive_time))]
            r, best_lag = _score_competitor(at, sig, self._max_lag)
            flow = self.validator.information_flow(comp, "adaptive_time", signals)

            scores.append({
                "generator": comp,
                "r_adaptive_time": r,
                "lag": best_lag,
                "information_flow": flow,
            })

        scores.sort(key=lambda x: abs(x["r_adaptive_time"]), reverse=True)

        compression_rank = next((i + 1 for i, s in enumerate(scores) if s["generator"] == "compression"), None)
        compression_score = next((s for s in scores if s["generator"] == "compression"), None)

        metrics = {
            "scores": scores,
            "compression_rank": compression_rank,
            "compression_r": compression_score["r_adaptive_time"] if compression_score else None,
            "compression_lag": compression_score["lag"] if compression_score else None,
            "best_generator": scores[0]["generator"] if scores else None,
            "best_r": scores[0]["r_adaptive_time"] if scores else None,
            "n_competitors": len(scores),
        }

        print(f"  Generator tournament (predicting adaptive_time):")
        for i, s in enumerate(scores):
            mark = " <<<" if s["generator"] == "compression" else ""
            print(f"    {i+1:2d}. {s['generator']:25s}: r={s['r_adaptive_time']:.4f}, lag={s['lag']:+4d}, flow={s['information_flow']:.6f}{mark}")

        if compression_rank is not None and compression_rank <= 2:
            status = "PASSED"
        elif compression_rank is not None and compression_rank <= len(scores) // 2:
            status = "INCONCLUSIVE"
        else:
            status = "FAILED"

        return CPIResult("generator_tournament", status, metrics=metrics)
