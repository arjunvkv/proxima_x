from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numba
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Numba-accelerated helpers
# ---------------------------------------------------------------------------


@numba.jit(nopython=True, cache=True)
def _count_positive_at_h(
    returns: NDArray[np.float64],
    start_indices: NDArray[np.int64],
    h: int,
) -> tuple[int, int]:
    n = len(start_indices)
    pos = 0
    neg = 0
    for i in range(n):
        idx = start_indices[i]
        if idx + h > len(returns):
            continue
        cum = np.sum(returns[idx : idx + h])
        if cum > 0.0:
            pos += 1
        elif cum < 0.0:
            neg += 1
    return pos, neg


@numba.jit(nopython=True, cache=True)
def _time_to_first_threshold(
    returns: NDArray[np.float64],
    start_idx: int,
    threshold: float,
    max_h: int,
) -> int:
    cum = 0.0
    end = min(start_idx + max_h, len(returns))
    for i in range(start_idx, end):
        cum += returns[i]
        if cum >= threshold:
            return i - start_idx + 1
    return max_h


@numba.jit(nopython=True, cache=True)
def _time_to_first_negative_threshold(
    returns: NDArray[np.float64],
    start_idx: int,
    threshold: float,
    max_h: int,
) -> int:
    cum = 0.0
    end = min(start_idx + max_h, len(returns))
    for i in range(start_idx, end):
        cum += returns[i]
        if cum <= threshold:
            return i - start_idx + 1
    return max_h


@numba.jit(nopython=True, cache=True)
def _avg_time_to_profit_batch(
    returns: NDArray[np.float64],
    start_indices: NDArray[np.int64],
    max_h: int,
) -> float:
    n = len(start_indices)
    if n == 0:
        return float(max_h)
    total = 0.0
    count = 0
    for i in range(n):
        idx = start_indices[i]
        steps = _time_to_first_threshold(returns, idx, 0.0, max_h)
        if steps < max_h:
            total += steps
            count += 1
    return total / count if count > 0 else float(max_h)


@numba.jit(nopython=True, cache=True)
def _avg_time_to_loss_batch(
    returns: NDArray[np.float64],
    start_indices: NDArray[np.int64],
    max_h: int,
) -> float:
    n = len(start_indices)
    if n == 0:
        return float(max_h)
    total = 0.0
    count = 0
    for i in range(n):
        idx = start_indices[i]
        steps = _time_to_first_negative_threshold(returns, idx, 0.0, max_h)
        if steps < max_h:
            total += steps
            count += 1
    return total / count if count > 0 else float(max_h)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class BucketSurvivability:
    count: int = 0
    probability_positive: dict[int, float] = field(default_factory=dict)
    probability_negative: dict[int, float] = field(default_factory=dict)
    average_time_to_profit: float = 0.0
    average_time_to_loss: float = 0.0
    survival_ratio: float = 0.0


@dataclass
class TradeSurvivabilityReport:
    asset: str
    n_samples: int = 0
    buckets: dict[str, BucketSurvivability] = field(default_factory=dict)
    verdict: str = "no_survivability_effect"


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class TradeSurvivabilityAnalyzer:
    """Investigate RQ2 — Trade Survivability.

    For each adaptive-time bucket, measure:
      - Probability of positive cumulative return at each horizon.
      - Probability of negative cumulative return at each horizon.
      - Average steps to first positive (profit).
      - Average steps to first negative (loss).
      - Survival ratio (prob_positive / prob_negative at the longest
        horizon).
    """

    def __init__(
        self,
        n_buckets: int = 5,
        horizons: list[int] | None = None,
    ) -> None:
        if n_buckets < 2:
            raise ValueError(f"n_buckets must be >= 2, got {n_buckets}")
        self.n_buckets = n_buckets
        self.horizons = horizons or [1, 5, 20, 50, 100, 500]
        self.bucket_labels = ["very_low", "low", "medium", "high", "extreme"]

    # ------------------------------------------------------------------
    # Public entry-point
    # ------------------------------------------------------------------

    def compute(
        self,
        adaptive_time: NDArray[np.float64],
        returns: NDArray[np.float64],
        asset: str = "unknown",
    ) -> TradeSurvivabilityReport:
        at = np.asarray(adaptive_time, dtype=np.float64).ravel()
        r = np.asarray(returns, dtype=np.float64).ravel()

        n = len(at)
        if n == 0:
            return self._empty_report(asset)

        # 1. Bucket adaptive time
        bucket_idx = self._bucket(at)
        max_h = max(self.horizons)

        buckets: dict[str, BucketSurvivability] = {}

        for b in range(self.n_buckets):
            label = self._bucket_label(b)
            mask = bucket_idx == b
            count = int(mask.sum())
            starts = np.where(mask)[0].astype(np.int64)

            if count < 2:
                buckets[label] = BucketSurvivability(count=count)
                continue

            prob_pos: dict[int, float] = {}
            prob_neg: dict[int, float] = {}

            for h in self.horizons:
                h_valid = starts[starts + h < n]
                if len(h_valid) < 1:
                    continue
                pos, neg = _count_positive_at_h(r, h_valid, h)
                total = pos + neg
                if total > 0:
                    prob_pos[h] = pos / total
                    prob_neg[h] = neg / total
                else:
                    prob_pos[h] = 0.0
                    prob_neg[h] = 0.0

            # Average time to profit / loss
            avg_profit = _avg_time_to_profit_batch(r, starts, max_h)
            avg_loss = _avg_time_to_loss_batch(r, starts, max_h)

            # Survival ratio at longest horizon
            longest = max(self.horizons)
            p_pos = prob_pos.get(longest, 0.0)
            p_neg = prob_neg.get(longest, 0.0)
            survival_ratio = p_pos / p_neg if p_neg > 0 else (float("inf") if p_pos > 0 else 1.0)

            buckets[label] = BucketSurvivability(
                count=count,
                probability_positive=prob_pos,
                probability_negative=prob_neg,
                average_time_to_profit=avg_profit,
                average_time_to_loss=avg_loss,
                survival_ratio=survival_ratio,
            )

        verdict = self._determine_verdict(buckets)

        return TradeSurvivabilityReport(
            asset=asset,
            n_samples=n,
            buckets=buckets,
            verdict=verdict,
        )

    # ------------------------------------------------------------------
    # Bucketing
    # ------------------------------------------------------------------

    def _bucket(self, adaptive_time: NDArray[np.float64]) -> NDArray[np.int64]:
        n = len(adaptive_time)
        if n < self.n_buckets + 1:
            return np.zeros(n, dtype=np.int64)
        pcts = np.linspace(0, 100, self.n_buckets + 1)[1:-1]
        thresholds = np.percentile(adaptive_time, pcts)
        thresholds = np.unique(thresholds)
        if thresholds.shape[0] == 0:
            return np.zeros(n, dtype=np.int64)
        return np.digitize(adaptive_time, thresholds, right=False).astype(np.int64)

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    @numba.jit(nopython=True, cache=True)
    def _time_to_first(arr: NDArray[np.float64], threshold: float) -> int:
        cum = 0.0
        for i in range(len(arr)):
            cum += arr[i]
            if cum >= threshold:
                return i + 1
        return len(arr)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _bucket_label(self, b: int) -> str:
        if b < len(self.bucket_labels):
            return self.bucket_labels[b]
        return f"bucket_{b}"

    def _determine_verdict(
        self,
        buckets: dict[str, BucketSurvivability],
    ) -> str:
        active = [b for b in buckets.values() if b.count >= 2]
        if len(active) < 2:
            return "no_survivability_effect"

        # Check monotonicity of survival_ratio across buckets
        ratios = [b.survival_ratio for b in active]
        # Filter inf/nan
        clean = [r for r in ratios if np.isfinite(r)]
        if len(clean) < 2:
            return "no_survivability_effect"

        is_monotonic = all(clean[i] <= clean[i + 1] for i in range(len(clean) - 1))
        spread = max(clean) - min(clean)

        if is_monotonic and spread > 0.2:
            return "adaptive_time_affects_survivability"
        if spread > 0.3:
            return "adaptive_time_affects_survivability"
        return "no_survivability_effect"

    def _empty_report(self, asset: str) -> TradeSurvivabilityReport:
        labels = [self._bucket_label(b) for b in range(self.n_buckets)]
        empty = {lb: BucketSurvivability(count=0) for lb in labels}
        return TradeSurvivabilityReport(
            asset=asset,
            buckets=empty,
            verdict="no_survivability_effect",
        )
