"""Universality analysis for temporal reality across assets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numba import jit


@dataclass
class UniversalityReport:
    """Report of universality analysis across trading assets.

    Attributes
    ----------
    assets : list[str]
        Assets that were compared.
    distribution_similarity : dict[str, float]
        Mapping ``"assetA__assetB"`` → KS-based similarity (1 - KS statistic).
    mutation_similarity : dict[str, float]
        Mapping ``"assetA__assetB"`` → correlation of rolling mutation rates.
    evolution_similarity : dict[str, float]
        Mapping ``"assetA__assetB"`` → correlation of evolution profile vectors.
    verdict : str
        One of ``"universal"``, ``"fx_specific"``, ``"equity_specific"``,
        ``"mixed"``.
    """

    assets: list[str]
    distribution_similarity: dict[str, float] = field(default_factory=dict)
    mutation_similarity: dict[str, float] = field(default_factory=dict)
    evolution_similarity: dict[str, float] = field(default_factory=dict)
    verdict: str = ""


# ---------------------------------------------------------------------------
# Numba-accelerated helpers
# ---------------------------------------------------------------------------


@jit(nopython=True, cache=True)
def _ks_statistic_impl(x: np.ndarray, y: np.ndarray) -> float:
    """Kolmogorov–Smirnov statistic between two 1-D samples.

    Returns the maximum absolute difference between the empirical cumulative
    distribution functions of *x* and *y*.
    """
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


@jit(nopython=True, cache=True)
def _percentile_impl(sorted_x: np.ndarray, q: int, n: int) -> float:
    """Linear-interpolation percentile on a sorted array."""
    idx = q / 100.0 * (n - 1)
    lo = int(np.floor(idx))
    hi = int(np.ceil(idx))
    if lo == hi:
        return sorted_x[lo]
    frac = idx - lo
    return sorted_x[lo] * (1.0 - frac) + sorted_x[hi] * frac


@jit(nopython=True, cache=True)
def _distribution_params_impl(x: np.ndarray) -> tuple[float, ...]:
    """Return (mean, std, skew, excess_kurtosis, p25, p50, p75)."""
    n = x.shape[0]
    mean = np.mean(x)

    # Sample standard deviation (ddof=1)
    if n > 1:
        std = np.sqrt(np.sum((x - mean) ** 2) / (n - 1))
    else:
        std = 0.0

    # Skewness (moment-based, adjusted for sample bias)
    if std > 0 and n > 2:
        skew = np.mean(((x - mean) / std) ** 3)
    else:
        skew = 0.0

    # Excess kurtosis (moment-based)
    if std > 0 and n > 3:
        kurt = np.mean(((x - mean) / std) ** 4) - 3.0
    else:
        kurt = 0.0

    s = np.sort(x)
    p25 = _percentile_impl(s, 25, n)
    p50 = _percentile_impl(s, 50, n)
    p75 = _percentile_impl(s, 75, n)

    return (mean, std, skew, kurt, p25, p50, p75)


@jit(nopython=True, cache=True)
def _rolling_mutation_rate_impl(states: np.ndarray, window: int) -> np.ndarray:
    """Rolling fraction of state transitions over a sliding window."""
    n = len(states)
    if n < window:
        return np.empty(0, dtype=np.float64)

    out = np.empty(n - window + 1, dtype=np.float64)
    for i in range(out.shape[0]):
        changes = 0.0
        for j in range(i + 1, i + window):
            if abs(states[j] - states[j - 1]) > 1e-12:
                changes += 1.0
        out[i] = changes / window
    return out


@jit(nopython=True, cache=True)
def _autocorr_impl(x: np.ndarray, lag: int) -> float:
    """Pearson autocorrelation at a given lag."""
    n = len(x)
    if n <= lag + 1:
        return 0.0
    m = n - lag
    x0 = x[:m]
    x1 = x[lag:]
    mu0 = np.mean(x0)
    mu1 = np.mean(x1)
    num = np.sum((x0 - mu0) * (x1 - mu1))
    den = np.sqrt(np.sum((x0 - mu0) ** 2) * np.sum((x1 - mu1) ** 2))
    return num / den if den > 0 else 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class UniversalityAnalyzer:
    """Analyze cross-asset universality of temporal-reality dynamics.

    Parameters
    ----------
    assets : list[str] | None
        Ordered list of asset names to analyse.  Defaults to the standard
        Proxima X Phase 4 asset universe.
    """

    FX_ASSETS = frozenset({"EURJPY", "USDJPY", "GBPJPY"})
    EQUITY_ASSETS = frozenset({"XAUUSD"})

    def __init__(self, assets: list[str] | None = None) -> None:
        self.assets = assets or [
            "EURJPY",
            "USDJPY",
            "GBPJPY",
            "XAUUSD",
        ]

    # ------------------------------------------------------------------
    # Public entry-point
    # ------------------------------------------------------------------

    def compute(self, asset_data: dict[str, dict[str, Any]]) -> UniversalityReport:
        """Compute pairwise universality metrics and produce a report.

        Parameters
        ----------
        asset_data : dict[str, dict]
            Nested dict keyed by asset name.  Each inner dict **must**
            contain at least the keys ``"adaptive_time"`` and ``"states"``,
            each mapping to a 1-D :class:`numpy.ndarray`.

        Returns
        -------
        UniversalityReport
        """
        report = UniversalityReport(assets=list(self.assets))

        pairs = [
            (self.assets[i], self.assets[j])
            for i in range(len(self.assets))
            for j in range(i + 1, len(self.assets))
        ]

        d_sim: dict[str, float] = {}
        m_sim: dict[str, float] = {}
        e_sim: dict[str, float] = {}

        for a, b in pairs:
            da = asset_data[a]
            db = asset_data[b]

            at_a = np.asarray(da["adaptive_time"], dtype=np.float64).ravel()
            at_b = np.asarray(db["adaptive_time"], dtype=np.float64).ravel()
            st_a = np.asarray(da["states"], dtype=np.float64).ravel()
            st_b = np.asarray(db["states"], dtype=np.float64).ravel()

            key = f"{a}__{b}"

            # 1. Distribution similarity (1 - KS statistic)
            d_sim[key] = 1.0 - self._ks_statistic(at_a, at_b)

            # 2. Mutation similarity – correlation of rolling mutation rates
            window = max(3, min(50, len(st_a) // 10, len(st_b) // 10))
            mr_a = _rolling_mutation_rate_impl(st_a, window)
            mr_b = _rolling_mutation_rate_impl(st_b, window)
            min_len = min(len(mr_a), len(mr_b))
            if min_len > 1:
                cc = np.corrcoef(mr_a[:min_len], mr_b[:min_len])[0, 1]
                m_sim[key] = float(cc) if not np.isnan(cc) else 0.0
            else:
                m_sim[key] = 0.0

            # 3. Evolution similarity – correlation of evolution profile vectors
            ep_a = self._evolution_profile(at_a)
            ep_b = self._evolution_profile(at_b)
            e_sim[key] = self._profile_correlation(ep_a, ep_b)

        report.distribution_similarity = d_sim
        report.mutation_similarity = m_sim
        report.evolution_similarity = e_sim
        report.verdict = self._determine_verdict(d_sim, m_sim, e_sim)

        return report

    # ------------------------------------------------------------------
    # Static helpers (delegated to numba for speed)
    # ------------------------------------------------------------------

    @staticmethod
    def _ks_statistic(x: np.ndarray, y: np.ndarray) -> float:
        """Kolmogorov–Smirnov statistic between two 1-D samples.

        Delegates to a :func:`numba.jit` compiled routine.
        """
        return float(_ks_statistic_impl(x, y))

    @staticmethod
    def _distribution_params(x: np.ndarray) -> dict[str, float]:
        """Return distribution summary statistics for the array *x*.

        Returns
        -------
        dict[str, float]
            Keys: ``mean``, ``std``, ``skew``, ``kurtosis`` (excess),
            ``p25``, ``p50``, ``p75``.
        """
        mean, std, skew, kurt, p25, p50, p75 = _distribution_params_impl(x)
        return {
            "mean": mean,
            "std": std,
            "skew": skew,
            "kurtosis": kurt,
            "p25": p25,
            "p50": p50,
            "p75": p75,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _evolution_profile(x: np.ndarray) -> np.ndarray:
        """Construct a feature vector describing the evolution of *x*.

        The profile concatenates distribution parameters (mean, std, skew,
        kurtosis, quartiles) with autocorrelation at lags 1, 5, 10, 20.
        """
        params = np.array(_distribution_params_impl(x), dtype=np.float64)
        acf = np.array(
            [_autocorr_impl(x, lag) for lag in (1, 5, 10, 20)],
            dtype=np.float64,
        )
        return np.concatenate((params, acf))

    @staticmethod
    def _profile_correlation(pa: np.ndarray, pb: np.ndarray) -> float:
        """Pearson correlation between two evolution profile vectors."""
        cc = np.corrcoef(pa, pb)[0, 1]
        return float(cc) if not np.isnan(cc) else 0.0

    # ------------------------------------------------------------------
    # Verdict logic
    # ------------------------------------------------------------------

    def _determine_verdict(
        self,
        dist_sim: dict[str, float],
        mut_sim: dict[str, float],
        evo_sim: dict[str, float],
    ) -> str:
        """Classify the cross-asset universality pattern.

        Rules (applied in order):

        1. If the composite (mean of all three metrics) exceeds 0.9 for
           **every** pair → ``"universal"``.
        2. If FX–FX pairs average > 0.85 and FX–equity pairs average < 0.5
           → ``"fx_specific"``.
        3. If equity–equity pairs average > 0.85 and FX–equity pairs average
           < 0.5 → ``"equity_specific"``.
        4. Otherwise → ``"mixed"``.
        """
        keys = list(dist_sim.keys())
        if not keys:
            return "mixed"

        composites: list[tuple[str, str, float]] = []
        for k in keys:
            avg = (dist_sim[k] + mut_sim[k] + evo_sim[k]) / 3.0
            a, b = k.split("__", 1)
            composites.append((a, b, avg))

        # Rule 1: all pairs universal
        if all(s >= 0.9 for _, _, s in composites):
            return "universal"

        # Categorise pairs
        fx_fx: list[float] = []
        fx_eq: list[float] = []
        eq_eq: list[float] = []

        for a, b, s in composites:
            a_fx = a in self.FX_ASSETS
            b_fx = b in self.FX_ASSETS
            a_eq = a in self.EQUITY_ASSETS
            b_eq = b in self.EQUITY_ASSETS

            if a_fx and b_fx:
                fx_fx.append(s)
            elif a_eq and b_eq:
                eq_eq.append(s)
            elif (a_fx and b_eq) or (a_eq and b_fx):
                fx_eq.append(s)

        mean_fx_fx = float(np.mean(fx_fx)) if fx_fx else 0.0
        mean_eq_eq = float(np.mean(eq_eq)) if eq_eq else 0.0
        mean_fx_eq = float(np.mean(fx_eq)) if fx_eq else 0.0

        # Rule 2
        if fx_fx and mean_fx_fx > 0.85 and mean_fx_eq < 0.5:
            return "fx_specific"

        # Rule 3
        if eq_eq and mean_eq_eq > 0.85 and mean_fx_eq < 0.5:
            return "equity_specific"

        return "mixed"
