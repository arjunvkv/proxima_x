"""
Null model generators for validating adaptive time coordinate uniqueness.

Provides a suite of synthetic "time" coordinates that serve as null
hypotheses against the adaptive-time coordinate of Proxima X Reality Phase 4.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from numba import jit


@dataclass
class NullModelResult:
    """Result of comparing a single null model against the target variable."""

    mi: float
    mi_ratio: float
    correlation_with_adaptive: float
    notes: str


@dataclass
class NullModelComparison:
    """Aggregated comparison of all null models against adaptive time.

    Attributes
    ----------
    asset : str
        Asset identifier.
    adaptive_time_mi : float
        Mutual information between adaptive-time coordinate and returns.
    models : dict[str, NullModelResult]
        Per-model comparison results.
    verdict : str
        One of ``"adaptive_time_is_unique"``, ``"simpler_explanations_exist"``,
        or ``"inconclusive"``.
    """

    asset: str
    adaptive_time_mi: float
    models: Dict[str, NullModelResult]
    verdict: str


class NullModelGenerator:
    """Generate null-model time coordinates and compare against adaptive time.

    Produces four null-model coordinates:

      1. **randomized_time_coordinate** — sorted uniform random values,
         preserving a monotonic trend without any market-driven structure.
      2. **shuffled_time_coordinate** — randomly permuted adaptive-time
         values, destroying temporal ordering.
      3. **volatility_based_coordinate** — cumulative normalised rolling
         standard deviation of returns.
      4. **entropy_based_coordinate** — cumulative normalised rolling
         Shannon entropy of returns.

    Each is compared against the same target (returns) via mutual
    information to see whether the adaptive-time coordinate carries
    unique predictive information.
    """

    def __init__(self, seed: int = 42) -> None:
        self._rng: np.random.Generator = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        adaptive_time: np.ndarray,
        returns: np.ndarray,
        volume: np.ndarray,
        volatility: np.ndarray,
        entropy: np.ndarray,
        asset: str = "",
    ) -> NullModelComparison:
        """Compare adaptive-time MI against four null models.

        Parameters
        ----------
        adaptive_time : np.ndarray
            1-D adaptive-time coordinate.
        returns : np.ndarray
            1-D observed returns (same length as *adaptive_time*).
        volume : np.ndarray
            1-D volume data (reserved for future null models).
        volatility : np.ndarray
            1-D volatility data (reserved for future null models).
        entropy : np.ndarray
            1-D entropy data (reserved for future null models).
        asset : str, optional
            Asset identifier for the comparison record.

        Returns
        -------
        NullModelComparison
            Aggregated comparison with verdict.
        """
        adaptive_time = self._as_float64_1d(adaptive_time)
        returns = self._as_float64_1d(returns)
        # volume, volatility, entropy are validated but reserved for
        # future null-model implementations.
        volume = self._as_float64_1d(volume)  # noqa: F841
        volatility = self._as_float64_1d(volatility)  # noqa: F841
        entropy = self._as_float64_1d(entropy)  # noqa: F841

        n = len(adaptive_time)
        if n != len(returns):
            raise ValueError(
                f"adaptive_time ({n}) and returns ({len(returns)}) must have the same length"
            )

        # Baseline MI (adaptive-time vs returns)
        adaptive_time_mi = self._compute_mi(adaptive_time, returns)

        # Build null-model coordinates
        null_coords: Dict[str, np.ndarray] = {
            "randomized_time_coordinate": self._randomized_time(n),
            "shuffled_time_coordinate": self._shuffled_time(adaptive_time),
            "volatility_based_coordinate": self._volatility_based(returns),
            "entropy_based_coordinate": self._entropy_based(returns),
        }

        models: Dict[str, NullModelResult] = {}
        for name, coord in null_coords.items():
            mi = self._compute_mi(coord, returns)
            mi_ratio = mi / adaptive_time_mi if adaptive_time_mi > 1e-12 else 0.0
            corr = (
                float(np.corrcoef(adaptive_time, coord)[0, 1])
                if n > 1
                else 0.0
            )
            notes = self._describe_model(name)

            models[name] = NullModelResult(
                mi=mi,
                mi_ratio=mi_ratio,
                correlation_with_adaptive=corr,
                notes=notes,
            )

        verdict = self._determine_verdict(adaptive_time_mi, models)

        return NullModelComparison(
            asset=asset,
            adaptive_time_mi=adaptive_time_mi,
            models=models,
            verdict=verdict,
        )

    # ------------------------------------------------------------------
    # Null model construction
    # ------------------------------------------------------------------

    def _randomized_time(self, n: int) -> np.ndarray:
        """Uniform random [0, 1] sorted to preserve monotonic trend."""
        return np.sort(self._rng.uniform(0.0, 1.0, size=n))

    def _shuffled_time(self, adaptive_time: np.ndarray) -> np.ndarray:
        """Randomly shuffle adaptive-time values (destroys temporal structure)."""
        arr = adaptive_time.copy()
        self._rng.shuffle(arr)
        return arr

    def _volatility_based(
        self, returns: np.ndarray, window: int = 20
    ) -> np.ndarray:
        """Cumulative normalised rolling volatility of returns.

        Parameters
        ----------
        returns : np.ndarray
            1-D returns array.
        window : int, optional
            Rolling window size (default 20).

        Returns
        -------
        np.ndarray
            Time coordinate in [0, 1] based on cumulative volatility.
        """
        n = len(returns)
        if n < 2:
            return np.full(n, 0.5)

        win = max(2, min(window, n))
        roll_std = np.empty(n, dtype=np.float64)

        for i in range(n):
            start = max(0, i - win + 1)
            seg = returns[start : i + 1]
            roll_std[i] = float(np.std(seg, ddof=1)) if len(seg) >= 2 else 0.0

        cum = np.cumsum(roll_std)
        if cum[-1] > 1e-12:
            cum /= cum[-1]
        return cum

    def _entropy_based(
        self, returns: np.ndarray, window: int = 20, n_bins: int = 5
    ) -> np.ndarray:
        """Cumulative normalised rolling entropy of returns.

        Parameters
        ----------
        returns : np.ndarray
            1-D returns array.
        window : int, optional
            Rolling window size (default 20).
        n_bins : int, optional
            Number of bins for discretisation within each window (default 5).

        Returns
        -------
        np.ndarray
            Time coordinate in [0, 1] based on cumulative return entropy.
        """
        n = len(returns)
        if n < 2:
            return np.full(n, 0.5)

        win = max(2, min(window, n))
        entropies = np.empty(n, dtype=np.float64)

        for i in range(n):
            start = max(0, i - win + 1)
            seg = returns[start : i + 1]
            entropies[i] = self._window_entropy(seg, n_bins)

        cum = np.cumsum(entropies)
        if cum[-1] > 1e-12:
            cum /= cum[-1]
        return cum

    # ------------------------------------------------------------------
    # Mutual information
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_mi(x: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
        """Discretisation-based mutual information ``I(X; Y)``.

        Parameters
        ----------
        x : np.ndarray
            First 1-D signal.
        y : np.ndarray
            Second 1-D signal.
        n_bins : int, optional
            Number of bins per dimension (default 10).

        Returns
        -------
        float
            Mutual information in nats.
        """
        return NullModelGenerator._numba_mi(
            np.ascontiguousarray(x, dtype=np.float64),
            np.ascontiguousarray(y, dtype=np.float64),
            n_bins,
        )

    @staticmethod
    @jit(nopython=True, cache=True)
    def _numba_mi(x: np.ndarray, y: np.ndarray, n_bins: int) -> float:
        """Numba-accelerated mutual information via joint histogram."""
        n = len(x)
        if n < 2:
            return 0.0

        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()

        if x_max - x_min < 1e-12 or y_max - y_min < 1e-12:
            return 0.0

        eps = 1e-12

        # Digitize both signals into [0, n_bins - 1]
        x_bins = np.floor(
            (x - x_min) / (x_max - x_min + eps) * n_bins
        ).astype(np.int64)
        x_bins = np.clip(x_bins, 0, n_bins - 1)

        y_bins = np.floor(
            (y - y_min) / (y_max - y_min + eps) * n_bins
        ).astype(np.int64)
        y_bins = np.clip(y_bins, 0, n_bins - 1)

        # Joint histogram
        joint = np.zeros((n_bins, n_bins), dtype=np.float64)
        for i in range(n):
            joint[x_bins[i], y_bins[i]] += 1.0

        joint /= n

        # Marginal distributions
        px = np.sum(joint, axis=1)
        py = np.sum(joint, axis=0)

        # Mutual information: sum_{i,j} p(i,j) * log(p(i,j) / (p(i) * p(j)))
        mi = 0.0
        for i in range(n_bins):
            if px[i] <= eps:
                continue
            for j in range(n_bins):
                pij = joint[i, j]
                if pij > eps and py[j] > eps:
                    mi += pij * np.log(pij / (px[i] * py[j]))

        return max(0.0, mi)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _as_float64_1d(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float64)
        if arr.ndim != 1:
            raise ValueError(f"Expected 1-D array, got {arr.ndim} dimensions")
        return arr

    @staticmethod
    def _window_entropy(segment: np.ndarray, n_bins: int) -> float:
        """Shannon entropy of a discretised return segment ``H(X) = -Σ p log p``."""
        m = len(segment)
        if m < 2:
            return 0.0

        lo, hi = segment.min(), segment.max()
        if hi - lo < 1e-12:
            return 0.0

        eps = 1e-12
        bins = np.floor(
            (segment - lo) / (hi - lo + eps) * n_bins
        ).astype(np.int64)
        bins = np.clip(bins, 0, n_bins - 1)

        counts = np.zeros(n_bins, dtype=np.float64)
        for v in bins:
            counts[v] += 1.0

        p = counts / m
        h = 0.0
        for pi in p:
            if pi > eps:
                h -= pi * np.log(pi)

        return h

    @staticmethod
    def _describe_model(name: str) -> str:
        descriptions = {
            "randomized_time_coordinate": (
                "Uniform random [0,1] sorted to preserve monotonic trend"
            ),
            "shuffled_time_coordinate": (
                "Randomly shuffled adaptive_time \u2014 destroys temporal structure"
            ),
            "volatility_based_coordinate": (
                "Cumulative normalised rolling volatility of returns"
            ),
            "entropy_based_coordinate": (
                "Cumulative normalised rolling entropy of returns"
            ),
        }
        return descriptions.get(name, "")

    @staticmethod
    def _determine_verdict(
        adaptive_time_mi: float,
        models: Dict[str, NullModelResult],
    ) -> str:
        """Classify the comparison outcome.

        Rules
        -----
        * ``"inconclusive"``
            Both adaptive-time MI and all null-model MIs are near zero.
        * ``"adaptive_time_is_unique"``
            Adaptive-time MI exceeds every null-model MI by more than 2\u00d7.
        * ``"simpler_explanations_exist"``
            Otherwise (at least one null model has comparable or better MI).
        """
        eps = 1e-12
        best_null_mi = max(m.mi for m in models.values())

        if adaptive_time_mi < eps and best_null_mi < eps:
            return "inconclusive"

        if adaptive_time_mi > 2.0 * best_null_mi:
            return "adaptive_time_is_unique"

        return "simpler_explanations_exist"
