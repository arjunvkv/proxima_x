from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numba
from numpy.typing import NDArray


@dataclass
class EvolutionStats:
    state_birth_rate: float = 0.0
    state_death_rate: float = 0.0
    state_mutation_rate: float = 0.0
    state_survival_rate: float = 0.0


@dataclass
class EvolutionClockReport:
    asset: str
    buckets: dict[str, EvolutionStats] = field(default_factory=dict)
    verdict: str = ""


class EvolutionClockAnalyzer:
    DEFAULT_WINDOW: int = 20

    def __init__(self, n_buckets: int = 5) -> None:
        if n_buckets < 2:
            raise ValueError("n_buckets must be >= 2, got %d" % n_buckets)
        self.n_buckets = n_buckets

    def compute(
        self,
        adaptive_time: np.ndarray,
        states: np.ndarray,
        asset: str = "unknown",
        window: int | None = None,
    ) -> EvolutionClockReport:
        adaptive_time = np.asarray(adaptive_time, dtype=np.float64).ravel()
        states = np.asarray(states, dtype=np.int64).ravel()

        if len(adaptive_time) == 0 or len(states) == 0:
            return self._empty_report(asset)

        if len(adaptive_time) != len(states):
            raise ValueError(
                f"adaptive_time ({len(adaptive_time)}) and states ({len(states)}) "
                "must have the same length"
            )

        w = window if window is not None else self.DEFAULT_WINDOW
        buckets_idx = self._bucket_adaptive_time(adaptive_time)
        birth, death, mutation, survival = self._compute_state_rates(states, w)

        bucket_labels = self._bucket_labels()
        buckets_dict: dict[str, EvolutionStats] = {}
        for b in range(self.n_buckets):
            label = bucket_labels[b]
            mask = buckets_idx == b
            if mask.sum() > 0:
                buckets_dict[label] = EvolutionStats(
                    state_birth_rate=float(np.mean(birth[mask])),
                    state_death_rate=float(np.mean(death[mask])),
                    state_mutation_rate=float(np.mean(mutation[mask])),
                    state_survival_rate=float(np.mean(survival[mask])),
                )
            else:
                buckets_dict[label] = EvolutionStats()

        verdict = self._determine_verdict(buckets_dict, bucket_labels)

        return EvolutionClockReport(
            asset=asset,
            buckets=buckets_dict,
            verdict=verdict,
        )

    def _bucket_labels(self) -> list[str]:
        default_labels = ["very_low", "low", "medium", "high", "extreme"]
        if self.n_buckets <= len(default_labels):
            return default_labels[: self.n_buckets]
        return [f"bucket_{i}" for i in range(self.n_buckets)]

    def _bucket_adaptive_time(self, adaptive_time: np.ndarray) -> NDArray[np.int64]:
        n = len(adaptive_time)
        if n < self.n_buckets + 1:
            return np.zeros(n, dtype=np.int64)

        percentiles = np.linspace(0, 100, self.n_buckets + 1)[1:-1]
        thresholds = np.percentile(adaptive_time, percentiles)
        thresholds = np.unique(thresholds)
        if thresholds.shape[0] == 0:
            return np.zeros(n, dtype=np.int64)

        return np.digitize(adaptive_time, thresholds, right=False).astype(np.int64)

    @staticmethod
    @numba.jit(nopython=True, cache=True)
    def _compute_state_rates(
        states: np.ndarray, window: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        n = len(states)
        birth = np.zeros(n, dtype=np.float64)
        death = np.zeros(n, dtype=np.float64)
        mutation = np.zeros(n, dtype=np.float64)
        survival = np.zeros(n, dtype=np.float64)

        if n < window:
            return birth, death, mutation, survival

        for i in range(window, n):
            curr = states[i - window : i]

            n_mutation = 0
            for j in range(1, window):
                if curr[j] != curr[j - 1]:
                    n_mutation += 1
            mutation[i] = n_mutation / window

            curr_unique = np.unique(curr)
            n_curr = curr_unique.shape[0]

            if i == window:
                birth[i] = n_curr / window
                death[i] = 0.0
                survival[i] = 0.0
            else:
                prev = states[i - window - 1 : i - 1]
                prev_unique = np.unique(prev)
                n_prev = prev_unique.shape[0]

                n_birth = 0
                for si in range(n_curr):
                    s = curr_unique[si]
                    found = False
                    for pj in range(n_prev):
                        if s == prev_unique[pj]:
                            found = True
                            break
                    if not found:
                        n_birth += 1
                birth[i] = n_birth / window

                n_death = 0
                for pj in range(n_prev):
                    ps = prev_unique[pj]
                    found = False
                    for si in range(n_curr):
                        if ps == curr_unique[si]:
                            found = True
                            break
                    if not found:
                        n_death += 1
                death[i] = n_death / window

                n_survival = 0
                for si in range(n_curr):
                    s = curr_unique[si]
                    for pj in range(n_prev):
                        if s == prev_unique[pj]:
                            n_survival += 1
                            break
                survival[i] = n_survival / window

        return birth, death, mutation, survival

    def _determine_verdict(
        self,
        buckets: dict[str, EvolutionStats],
        labels: list[str],
    ) -> str:
        active = [lbl for lbl in labels if lbl in buckets]
        if len(active) < 2:
            return "WEAK_EVOLUTION_CLOCK"

        mutation_rates = [buckets[lbl].state_mutation_rate for lbl in active]
        birth_rates = [buckets[lbl].state_birth_rate for lbl in active]

        mutation_increasing = all(
            mutation_rates[i] <= mutation_rates[i + 1]
            for i in range(len(mutation_rates) - 1)
        )
        mutation_strict = mutation_rates[-1] > mutation_rates[0]

        birth_increasing = all(
            birth_rates[i] <= birth_rates[i + 1]
            for i in range(len(birth_rates) - 1)
        )
        birth_strict = birth_rates[-1] > birth_rates[0]

        if mutation_increasing and mutation_strict:
            return "STRONG_EVOLUTION_CLOCK"
        if birth_increasing and birth_strict:
            return "MODERATE_EVOLUTION_CLOCK"
        return "WEAK_EVOLUTION_CLOCK"

    def _empty_report(self, asset: str) -> EvolutionClockReport:
        labels = self._bucket_labels()
        empty_buckets = {lbl: EvolutionStats() for lbl in labels}
        return EvolutionClockReport(
            asset=asset,
            buckets=empty_buckets,
            verdict="WEAK_EVOLUTION_CLOCK",
        )
