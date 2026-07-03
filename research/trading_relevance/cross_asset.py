from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numba
from numpy.typing import NDArray

from research.information_discovery.mi_estimator import _fast_mutual_info


# ---------------------------------------------------------------------------
# Numba-accelerated helpers
# ---------------------------------------------------------------------------


@numba.jit(nopython=True, cache=True)
def _numba_ks_statistic(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    """Two-sample Kolmogorov-Smirnov statistic (maximum ECDF divergence)."""
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
def _numba_forward_volatility(
    indices: NDArray[np.int64],
    returns: NDArray[np.float64],
    h: int,
) -> NDArray[np.float64]:
    """Forward-looking volatility (ddof=1 std) over *h* steps at each index.

    Parameters
    ----------
    indices : NDArray[np.int64]
        Starting positions in *returns*.
    returns : NDArray[np.float64]
        1-D return series.
    h : int
        Horizon over which to compute volatility.

    Returns
    -------
    NDArray[np.float64]
        Volatility estimate per index (NaN if insufficient data).
    """
    n = len(returns)
    out = np.empty(len(indices), dtype=np.float64)
    for k in range(len(indices)):
        start = int(indices[k])
        end = min(start + h, n)
        window = returns[start:end]
        m = len(window)
        if m < 2:
            out[k] = np.nan
            continue
        mean = 0.0
        for i in range(m):
            mean += window[i]
        mean /= m
        var = 0.0
        for i in range(m):
            diff = window[i] - mean
            var += diff * diff
        out[k] = np.sqrt(var / (m - 1))
    return out


@numba.jit(nopython=True, cache=True)
def _numba_survivability_counts(
    indices: NDArray[np.int64],
    returns: NDArray[np.float64],
    h: int,
) -> tuple[int, int]:
    """Count positive and negative forward returns at horizon *h*.

    Parameters
    ----------
    indices : NDArray[np.int64]
        Starting positions in *returns*.
    returns : NDArray[np.float64]
        1-D return series.
    h : int
        Forward horizon to check sign.

    Returns
    -------
    tuple[int, int]
        (positive_count, negative_count).
    """
    n = len(returns)
    pos = 0
    neg = 0
    for k in range(len(indices)):
        idx = int(indices[k]) + h
        if idx < n:
            r = returns[idx]
            if r > 0.0:
                pos += 1
            elif r < 0.0:
                neg += 1
    return pos, neg


@numba.jit(nopython=True, cache=True)
def _numba_extreme_indices(
    adaptive_time: NDArray[np.float64],
    h: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Return sorted indices of the bottom-*h* and top-*h* values.

    Returns
    -------
    tuple[NDArray[np.int64], NDArray[np.int64]]
        (low_indices_sorted_by_time_position,
         high_indices_sorted_by_time_position).
    """
    n = len(adaptive_time)
    order = np.argsort(adaptive_time)
    low = order[:h]
    high = order[-h:]
    return np.sort(low).astype(np.int64), np.sort(high).astype(np.int64)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class AssetRelevanceMetrics:
    """Per-asset relevance metrics for RQ9."""

    outcome_separation: float = 0.0
    """KS statistic between returns at very-low vs very-high adaptive_time."""

    risk_score: float = 1.0
    """Ratio of forward volatility at extreme vs very-low adaptive_time."""

    survivability_ratio: float = 1.0
    """Laplace-smoothed ratio P(positive)/P(negative) at horizon h."""

    information_gain: float = 0.0
    """Mutual information between adaptive_time bucket and future returns."""


@dataclass
class CrossAssetRelevanceReport:
    """Cross-asset operational-relevance report for RQ9."""

    assets: list[str] = field(default_factory=list)
    per_asset: dict[str, AssetRelevanceMetrics] = field(default_factory=dict)
    outcome_separation_consistency: float = 0.0
    """Standard deviation of outcome_separation across assets (lower=more universal)."""
    risk_consistency: float = 0.0
    """Standard deviation of risk_score across assets."""
    survivability_consistency: float = 0.0
    """Standard deviation of survivability_ratio across assets."""
    verdict: str = ""
    """One of ``"operational_usefulness_transfers"`` / ``"asset_specific"`` /
    ``"mixed"``."""


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class CrossAssetRelevanceAnalyzer:
    """Analyze whether adaptive-time operational usefulness transfers across
    trading assets (RQ9).

    Runs identical outcome-distribution, risk, and survivability analyses
    on multiple assets and measures cross-asset consistency.

    Parameters
    ----------
    assets : list[str] | None
        Ordered list of asset names to analyse.  Defaults to the standard
        Phase 5 asset universe (EURJPY, USDJPY, GBPJPY, XAUUSD).
    """

    SUPPORTED_ASSETS: list[str] = [
        "EURJPY",
        "USDJPY",
        "GBPJPY",
        "XAUUSD",
    ]

    def __init__(self, assets: list[str] | None = None) -> None:
        self.assets = assets or list(self.SUPPORTED_ASSETS)

    # ------------------------------------------------------------------
    # Public entry-point
    # ------------------------------------------------------------------

    def compute(
        self,
        asset_data: dict[str, dict[str, Any]],
    ) -> CrossAssetRelevanceReport:
        """Compute cross-asset relevance metrics.

        Parameters
        ----------
        asset_data : dict[str, dict]
            Nested dict keyed by asset name.  Each inner dict **must**
            contain at least the keys ``"adaptive_time"``, ``"returns"``,
            and ``"states"``, each mapping to a 1-D :class:`numpy.ndarray`.

        Returns
        -------
        CrossAssetRelevanceReport
        """
        per_asset: dict[str, AssetRelevanceMetrics] = {}

        outcome_seps: list[float] = []
        risk_scores: list[float] = []
        surv_ratios: list[float] = []

        for asset in self.assets:
            if asset not in asset_data:
                continue
            d = asset_data[asset]
            at = np.asarray(d["adaptive_time"], dtype=np.float64).ravel()
            ret = np.asarray(d["returns"], dtype=np.float64).ravel()

            metrics = AssetRelevanceMetrics(
                outcome_separation=self._outcome_separation(at, ret),
                risk_score=self._risk_score(at, ret),
                survivability_ratio=self._survivability_ratio(at, ret),
                information_gain=self._information_gain(at, ret),
            )
            per_asset[asset] = metrics
            outcome_seps.append(metrics.outcome_separation)
            risk_scores.append(metrics.risk_score)
            surv_ratios.append(metrics.survivability_ratio)

        n = len(per_asset)
        report = CrossAssetRelevanceReport(
            assets=list(per_asset.keys()),
            per_asset=per_asset,
            outcome_separation_consistency=(
                float(np.std(outcome_seps)) if n > 1 else 0.0
            ),
            risk_consistency=float(np.std(risk_scores)) if n > 1 else 0.0,
            survivability_consistency=float(np.std(surv_ratios)) if n > 1 else 0.0,
            verdict="",
        )
        report.verdict = self._determine_verdict(
            outcome_seps, risk_scores, surv_ratios
        )
        return report

    # ------------------------------------------------------------------
    # Metric methods
    # ------------------------------------------------------------------

    @staticmethod
    def _outcome_separation(
        adaptive_time: NDArray,
        returns: NDArray,
        h: int = 20,
    ) -> float:
        """KS statistic between returns at very-low vs very-high adaptive_time.

        Parameters
        ----------
        adaptive_time : NDArray
            1-D adaptive-time coordinate.
        returns : NDArray
            1-D aligned return series.
        h : int
            Number of extreme tail observations per side.

        Returns
        -------
        float
            KS statistic in [0, 1].
        """
        at = np.asarray(adaptive_time, dtype=np.float64).ravel()
        r = np.asarray(returns, dtype=np.float64).ravel()
        n = len(at)
        k = min(h, n // 2)
        if k < 2:
            return 0.0

        low_idx, high_idx = _numba_extreme_indices(at, k)
        low_ret = r[low_idx]
        high_ret = r[high_idx]
        return float(_numba_ks_statistic(low_ret, high_ret))

    @staticmethod
    def _risk_score(
        adaptive_time: NDArray,
        returns: NDArray,
        h: int = 20,
    ) -> float:
        """Ratio of future volatility at extreme vs very-low adaptive_time.

        Parameters
        ----------
        adaptive_time : NDArray
            1-D adaptive-time coordinate.
        returns : NDArray
            1-D aligned return series.
        h : int
            Number of extreme tail observations / forward horizon.

        Returns
        -------
        float
            Volatility ratio (extreme / very_low).  >1 means higher risk at
            extreme adaptive-time levels.
        """
        at = np.asarray(adaptive_time, dtype=np.float64).ravel()
        r = np.asarray(returns, dtype=np.float64).ravel()
        n = len(at)
        k = min(h, n // 2)
        if k < 2:
            return 1.0

        low_idx, high_idx = _numba_extreme_indices(at, k)
        low_vols = _numba_forward_volatility(low_idx, r, h)
        high_vols = _numba_forward_volatility(high_idx, r, h)

        low_mean = float(np.nanmean(low_vols)) if low_vols.size > 0 else 0.0
        high_mean = float(np.nanmean(high_vols)) if high_vols.size > 0 else 0.0

        if low_mean < 1e-15:
            return 1.0
        return high_mean / low_mean

    @staticmethod
    def _survivability_ratio(
        adaptive_time: NDArray,
        returns: NDArray,
        h: int = 20,
    ) -> float:
        """Ratio of positive to negative probability at horizon *h*.

        A value >1 means positive outcomes are more likely when adaptive_time
        is at extreme levels.

        Parameters
        ----------
        adaptive_time : NDArray
            1-D adaptive-time coordinate.
        returns : NDArray
            1-D aligned return series.
        h : int
            Forward horizon (in steps) for outcome check.

        Returns
        -------
        float
            Laplace-smoothed ratio (positive + 1) / (negative + 1).
        """
        at = np.asarray(adaptive_time, dtype=np.float64).ravel()
        r = np.asarray(returns, dtype=np.float64).ravel()
        n = len(at)
        k = min(h, n // 2)
        if k < 2:
            return 1.0

        low_idx, high_idx = _numba_extreme_indices(at, k)
        combined = np.concatenate([low_idx, high_idx]).astype(np.int64)
        pos, neg = _numba_survivability_counts(combined, r, h)
        return float(pos + 1) / float(neg + 1)

    @staticmethod
    def _information_gain(
        adaptive_time: NDArray,
        returns: NDArray,
        n_bins: int = 10,
    ) -> float:
        """Mutual information between adaptive-time bucket and future returns.

        Parameters
        ----------
        adaptive_time : NDArray
            1-D adaptive-time coordinate.
        returns : NDArray
            1-D aligned return series.
        n_bins : int
            Number of bins for discretisation.

        Returns
        -------
        float
            Mutual information in nats.
        """
        at = np.asarray(adaptive_time, dtype=np.float64).ravel()
        r = np.asarray(returns, dtype=np.float64).ravel()
        return float(_fast_mutual_info(at, r, n_bins))

    # ------------------------------------------------------------------
    # Verdict logic
    # ------------------------------------------------------------------

    @staticmethod
    def _determine_verdict(
        outcome_seps: list[float],
        risk_scores: list[float],
        surv_ratios: list[float],
    ) -> str:
        """Determine whether adaptive-time operational usefulness transfers.

        Rules (applied in order):

        1. If *all three* consistency measures are low (std < 0.15 across
           assets) → ``"operational_usefulness_transfers"``.
        2. If the *mean* outcome separation is very weak (< 0.1) and mean
           risk score is near unity (< 1.1) across all assets →
           ``"asset_specific"``.
        3. Otherwise → ``"mixed"``.
        """
        if not outcome_seps:
            return "mixed"

        n = len(outcome_seps)
        os_std = float(np.std(outcome_seps)) if n > 1 else 0.0
        rs_std = float(np.std(risk_scores)) if n > 1 else 0.0
        sr_std = float(np.std(surv_ratios)) if n > 1 else 0.0

        # Rule 1 — consistency across assets implies transferability
        if n > 1 and os_std < 0.15 and rs_std < 0.15 and sr_std < 0.15:
            return "operational_usefulness_transfers"

        # Rule 2 — universally weak signal => asset-specific behaviour
        mean_os = float(np.mean(outcome_seps))
        mean_rs = float(np.mean(risk_scores))
        if mean_os < 0.1 and abs(mean_rs - 1.0) < 0.1:
            return "asset_specific"

        return "mixed"
