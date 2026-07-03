from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numba
from numpy.typing import NDArray

from research.information_discovery.mi_estimator import (
    _fast_percentile,
    _fast_digitize,
    _fast_entropy_digitized,
    _fast_mutual_info,
)


# ---------------------------------------------------------------------------
# Numba-accelerated helpers
# ---------------------------------------------------------------------------


@numba.jit(nopython=True, cache=True)
def _numba_ks_between(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    """Two-sample Kolmogorov-Smirnov statistic."""
    x_sorted = np.sort(x)
    y_sorted = np.sort(y)
    nx = x_sorted.shape[0]
    ny = y_sorted.shape[0]
    i = 0
    j = 0
    d = 0.0
    while i < nx and j < ny:
        if x_sorted[i] < y_sorted[j]:
            i += 1
        elif x_sorted[i] > y_sorted[j]:
            j += 1
        else:
            i += 1
            j += 1
        diff = abs(i / nx - j / ny)
        if diff > d:
            d = diff
    return d


@numba.jit(nopython=True, cache=True)
def _numba_bucket_quantile(
    adaptive_time: NDArray[np.float64],
    n_buckets: int,
) -> NDArray[np.int32]:
    """Quantile-based bucket assignment.

    Each observation is assigned to an integer bucket [0, n_buckets - 1]
    based on its quantile rank in *adaptive_time*.

    Parameters
    ----------
    adaptive_time : NDArray[np.float64]
        1-D input series.
    n_buckets : int
        Number of equally-populated buckets.

    Returns
    -------
    NDArray[np.int32]
        Bucket index per observation.
    """
    n = len(adaptive_time)
    buckets = np.empty(n, dtype=np.int32)
    order = np.argsort(adaptive_time)
    for i in range(n):
        idx = order[i]
        bucket = int(i * n_buckets / n)
        if bucket >= n_buckets:
            bucket = n_buckets - 1
        buckets[idx] = bucket
    return buckets


@numba.jit(nopython=True, cache=True)
def _numba_distribution_entropy(
    arr: NDArray[np.float64],
    n_bins: int = 20,
) -> float:
    """Shannon entropy of a 1-D distribution using an equal-width histogram.

    Parameters
    ----------
    arr : NDArray[np.float64]
        Input sample.
    n_bins : int
        Number of histogram bins.

    Returns
    -------
    float
        Shannon entropy in nats.
    """
    valid = arr[~np.isnan(arr)]
    n = len(valid)
    if n < 2:
        return 0.0

    lo = valid[0]
    hi = valid[0]
    for i in range(1, n):
        v = valid[i]
        if v < lo:
            lo = v
        if v > hi:
            hi = v

    span = hi - lo
    if span < 1e-15:
        return 0.0

    counts = np.zeros(n_bins, dtype=np.int64)
    for i in range(n):
        v = valid[i]
        idx = int((v - lo) / span * n_bins)
        if idx >= n_bins:
            idx = n_bins - 1
        counts[idx] += 1

    entropy = 0.0
    for j in range(n_bins):
        c = counts[j]
        if c > 0:
            p = c / n
            entropy -= p * np.log(p)
    return entropy


@numba.jit(nopython=True, cache=True)
def _numba_entropy_by_bucket(
    returns: NDArray[np.float64],
    buckets: NDArray[np.int32],
    n_buckets: int,
    hist_bins: int,
) -> NDArray[np.float64]:
    """Compute distribution entropy of returns per bucket.

    Parameters
    ----------
    returns : NDArray[np.float64]
        1-D return series.
    buckets : NDArray[np.int32]
        Bucket assignment per observation (same length).
    n_buckets : int
        Number of buckets.
    hist_bins : int
        Number of histogram bins for entropy computation.

    Returns
    -------
    NDArray[np.float64]
        Entropy per bucket; NaN for empty buckets.
    """
    n = len(returns)
    bucket_entropies = np.full(n_buckets, np.nan, dtype=np.float64)

    for b in range(n_buckets):
        mask_sum = 0
        for i in range(n):
            if buckets[i] == b:
                mask_sum += 1
        if mask_sum < 2:
            continue

        vals = np.empty(mask_sum, dtype=np.float64)
        j = 0
        for i in range(n):
            if buckets[i] == b:
                vals[j] = returns[i]
                j += 1
        bucket_entropies[b] = _numba_distribution_entropy(vals, hist_bins)

    return bucket_entropies


@numba.jit(nopython=True, cache=True)
def _numba_pairwise_ks(
    returns: NDArray[np.float64],
    buckets: NDArray[np.int32],
    n_buckets: int,
) -> float:
    """Average pairwise KS statistic between bucket return distributions.

    Parameters
    ----------
    returns : NDArray[np.float64]
        1-D return series.
    buckets : NDArray[np.int32]
        Bucket assignment per observation.
    n_buckets : int
        Number of buckets.

    Returns
    -------
    float
        Mean pairwise KS statistic.
    """
    n = len(returns)

    # Collect values per bucket
    bucket_vals: list[NDArray[np.float64]] = []
    for b in range(n_buckets):
        cnt = 0
        for i in range(n):
            if buckets[i] == b:
                cnt += 1
        vals = np.empty(cnt, dtype=np.float64)
        j = 0
        for i in range(n):
            if buckets[i] == b:
                vals[j] = returns[i]
                j += 1
        bucket_vals.append(vals)

    total_ks = 0.0
    pair_count = 0
    for i in range(n_buckets):
        for j in range(i + 1, n_buckets):
            xi = bucket_vals[i]
            yj = bucket_vals[j]
            if len(xi) > 1 and len(yj) > 1:
                total_ks += _numba_ks_between(xi, yj)
                pair_count += 1

    return total_ks / pair_count if pair_count > 0 else 0.0


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------


@dataclass
class EconomicValueReport:
    """Report of economic-value analysis for adaptive_time (RQ10)."""

    asset: str = ""
    unconditioned_entropy: float = 0.0
    """Shannon entropy of unconditioned return distribution."""
    conditioned_entropy: dict[str, float] = field(default_factory=dict)
    """Shannon entropy of return distribution conditioned on adaptive-time
    bucket.  Keys are bucket labels."""
    information_gain: float = 0.0
    """``unconditioned_entropy - weighted_avg_conditioned_entropy``."""
    uncertainty_reduction: float = 0.0
    """``information_gain / unconditioned_entropy`` (fractional reduction)."""
    distribution_separation: float = 0.0
    """Average pairwise KS statistic between bucket return distributions."""
    verdict: str = ""
    """One of ``"adaptive_time_reduces_uncertainty"`` /
    ``"no_uncertainty_reduction"`` / ``"mixed_results"``."""


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class EconomicValueAnalyzer:
    """Evaluate the economic value of adaptive_time by comparing unconditioned
    vs conditioned outcome distributions (RQ10).

    If conditioning on adaptive_time reduces uncertainty about future
    outcomes, it has economic value even without a price-direction prediction.

    Parameters
    ----------
    n_buckets : int
        Number of adaptive-time quantile buckets (default 5).
    horizons : list[int] | None
        Forward horizons at which to evaluate.  Defaults to
        ``[1, 5, 20, 50, 100, 500]``.
    """

    def __init__(
        self,
        n_buckets: int = 5,
        horizons: list[int] | None = None,
    ) -> None:
        self.n_buckets = n_buckets
        self.horizons = horizons or [1, 5, 20, 50, 100, 500]
        self.bucket_labels = ["very_low", "low", "medium", "high", "extreme"]

    # ------------------------------------------------------------------
    # Public entry-point
    # ------------------------------------------------------------------

    def compute(
        self,
        adaptive_time: NDArray,
        returns: NDArray,
        asset_name: str = "",
    ) -> EconomicValueReport:
        """Compute economic-value metrics.

        Parameters
        ----------
        adaptive_time : NDArray
            1-D adaptive-time coordinate.
        returns : NDArray
            1-D aligned return series.
        asset_name : str
            Optional asset identifier for the report.

        Returns
        -------
        EconomicValueReport
        """
        at = np.asarray(adaptive_time, dtype=np.float64).ravel()
        r = np.asarray(returns, dtype=np.float64).ravel()

        # 1. Unconditioned outcome entropy
        unconditioned_entropy = self._distribution_entropy(r)

        # 2. Bucket adaptive_time and compute conditioned entropy per bucket
        buckets = self._bucket(at)
        bucket_entropies = _numba_entropy_by_bucket(
            r, buckets, self.n_buckets, 20
        )

        # Build label -> entropy map
        conditioned_entropy: dict[str, float] = {}
        weights: list[float] = []
        weighted_sum = 0.0
        valid_buckets = 0
        for b in range(self.n_buckets):
            label = self.bucket_labels[b] if b < len(self.bucket_labels) else str(b)
            ent = float(bucket_entropies[b])
            conditioned_entropy[label] = ent
            if not np.isnan(ent):
                # Weight = proportion of observations in this bucket
                weight = float(np.sum(buckets == b)) / max(len(buckets), 1)
                weights.append(weight)
                weighted_sum += weight * ent
                valid_buckets += 1

        # 3. Information gain
        total_weight = sum(weights) if weights else 0.0
        avg_conditioned = weighted_sum / total_weight if total_weight > 0 else unconditioned_entropy
        information_gain = unconditioned_entropy - avg_conditioned

        # 4. Uncertainty reduction
        uncertainty_reduction = (
            information_gain / unconditioned_entropy
            if unconditioned_entropy > 1e-15
            else 0.0
        )

        # 5. Distribution separation (average pairwise KS)
        distribution_separation = float(
            _numba_pairwise_ks(r, buckets, self.n_buckets)
        )

        report = EconomicValueReport(
            asset=asset_name,
            unconditioned_entropy=unconditioned_entropy,
            conditioned_entropy=conditioned_entropy,
            information_gain=information_gain,
            uncertainty_reduction=uncertainty_reduction,
            distribution_separation=distribution_separation,
            verdict="",
        )
        report.verdict = self._determine_verdict(report)
        return report

    # ------------------------------------------------------------------
    # Bucketing
    # ------------------------------------------------------------------

    def _bucket(self, adaptive_time: NDArray) -> NDArray:
        """Quantile-based bucketing of *adaptive_time*.

        Returns an integer array (same shape as input) with values in
        ``[0, n_buckets - 1]``.
        """
        at = np.asarray(adaptive_time, dtype=np.float64).ravel()
        n = len(at)
        if n < self.n_buckets:
            return np.zeros(n, dtype=np.int32)
        return _numba_bucket_quantile(at, self.n_buckets)

    # ------------------------------------------------------------------
    # Entropy
    # ------------------------------------------------------------------

    @staticmethod
    @numba.jit(nopython=True, cache=True)
    def _distribution_entropy(arr: NDArray, n_bins: int = 20) -> float:
        """Compute Shannon entropy of a distribution using histogram bins.

        Parameters
        ----------
        arr : NDArray
            1-D input array.
        n_bins : int
            Number of equal-width histogram bins.

        Returns
        -------
        float
            Shannon entropy in nats.
        """
        x = np.asarray(arr, dtype=np.float64).ravel()
        return _numba_distribution_entropy(x, n_bins)

    # ------------------------------------------------------------------
    # KS statistic
    # ------------------------------------------------------------------

    @staticmethod
    def _ks_between(x: NDArray, y: NDArray) -> float:
        """KS statistic between two sample distributions.

        Parameters
        ----------
        x : NDArray
            First 1-D sample.
        y : NDArray
            Second 1-D sample.

        Returns
        -------
        float
            KS statistic in [0, 1].
        """
        xa = np.asarray(x, dtype=np.float64).ravel()
        ya = np.asarray(y, dtype=np.float64).ravel()
        if len(xa) < 2 or len(ya) < 2:
            return 1.0
        return float(_numba_ks_between(xa, ya))

    # ------------------------------------------------------------------
    # Verdict logic
    # ------------------------------------------------------------------

    @staticmethod
    def _determine_verdict(report: EconomicValueReport) -> str:
        """Determine if adaptive_time reduces uncertainty.

        Rules:

        1. ``adaptive_time_reduces_uncertainty`` — uncertainty reduction
           > 5 % AND distribution separation > 0.15.
        2. ``no_uncertainty_reduction`` — uncertainty reduction < 0 %
           AND distribution_separation < 0.05.
        3. ``mixed_results`` — otherwise.
        """
        ur = report.uncertainty_reduction
        ds = report.distribution_separation

        if ur > 0.05 and ds > 0.15:
            return "adaptive_time_reduces_uncertainty"
        if ur < 0.0 and ds < 0.05:
            return "no_uncertainty_reduction"
        return "mixed_results"
