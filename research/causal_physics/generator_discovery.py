from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numba
import numpy as np

from research.information_discovery.mi_estimator import _fast_conditional_mutual_info
from research.temporal_reality.causality_analysis import AdaptiveTimeCausality

TARGET_VARS = [
    "adaptive_time",
    "energy_storage",
    "memory_density",
    "memory_gradient",
    "state_mutation_rate",
    "regime_change_probability",
]


@dataclass
class GeneratorCandidate:
    source_variable: str
    target_variable: str
    peak_lag: int
    peak_corr: float
    transfer_entropy: float
    causal_strength: float


class GeneratorDiscoveryEngine:
    def __init__(self, max_lag: int = 50):
        self.max_lag = max_lag

    def compute(self, data: dict[str, np.ndarray]) -> list[GeneratorCandidate]:
        n_vars = len(TARGET_VARS)
        candidates: list[GeneratorCandidate] = []

        for i in range(n_vars):
            src_name = TARGET_VARS[i]
            x = data[src_name]
            for j in range(n_vars):
                if i == j:
                    continue
                tgt_name = TARGET_VARS[j]
                y = data[tgt_name]

                common = min(len(x), len(y))
                xf = np.ascontiguousarray(x[:common], dtype=np.float64)
                yf = np.ascontiguousarray(y[:common], dtype=np.float64)

                corr = AdaptiveTimeCausality._cross_correlate(xf, yf, self.max_lag)
                lags = np.arange(-self.max_lag, self.max_lag + 1)
                peak_idx = int(np.argmax(np.abs(corr)))
                peak_lag = int(lags[peak_idx])
                peak_corr = float(corr[peak_idx])
                causal_strength = _compute_causal_strength(peak_corr, xf.size, abs(peak_lag))

                te = 0.0
                if common > 1:
                    s_aligned = xf[: common - 1]
                    t_forward = yf[1:common]
                    t_aligned = yf[: common - 1]
                    te = _fast_conditional_mutual_info(s_aligned, t_forward, t_aligned, 10)

                candidates.append(
                    GeneratorCandidate(
                        source_variable=src_name,
                        target_variable=tgt_name,
                        peak_lag=peak_lag,
                        peak_corr=peak_corr,
                        transfer_entropy=te,
                        causal_strength=causal_strength,
                    )
                )

        return candidates


@numba.jit(nopython=True, cache=True)
def _compute_causal_strength(peak_corr: float, n_total: int, abs_lag: int) -> float:
    overlap = n_total - abs_lag
    if overlap < 3:
        return 0.0
    return abs(peak_corr) * np.sqrt(max(overlap - 2, 1))
