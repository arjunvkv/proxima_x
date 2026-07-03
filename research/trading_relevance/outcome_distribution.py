from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numba
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Numba-accelerated helpers
# ---------------------------------------------------------------------------


@numba.jit(nopython=True, cache=True)
def _sample_std(x: NDArray[np.float64]) -> float:
    n = len(x)
    if n < 2:
        return 0.0
    mu = np.mean(x)
    var = np.sum((x - mu) ** 2) / (n - 1)
    return float(np.sqrt(var))


@numba.jit(nopython=True, cache=True)
def _skew(x: NDArray[np.float64]) -> float:
    n = len(x)
    if n < 3:
        return 0.0
    mu = np.mean(x)
    s = _sample_std(x)
    if s < 1e-15:
        return 0.0
    return float(np.mean(((x - mu) / s) ** 3))


@numba.jit(nopython=True, cache=True)
def _kurtosis(x: NDArray[np.float64]) -> float:
    n = len(x)
    if n < 4:
        return 0.0
    mu = np.mean(x)
    s = _sample_std(x)
    if s < 1e-15:
        return 0.0
    return float(np.mean(((x - mu) / s) ** 4) - 3.0)


@numba.jit(nopython=True, cache=True)
def _bucket_future_return_stats(
    returns: NDArray[np.float64],
    start_indices: NDArray[np.int64],
    h: int,
) -> tuple[float, float, float, float]:
    n = len(start_indices)
    if n < 2:
        return (0.0, 0.0, 0.0, 0.0)
    vals = np.zeros(n, dtype=np.float64)
    for i in range(n):
        idx = start_indices[i]
        vals[i] = np.sum(returns[idx : idx + h])
    mean = float(np.mean(vals))
    std = _sample_std(vals)
    return (mean, std, _skew(vals), _kurtosis(vals))


@numba.jit(nopython=True, cache=True)
def _drawdowns_batch(
    prices: NDArray[np.float64],
    start_indices: NDArray[np.int64],
    h: int,
) -> NDArray[np.float64]:
    n = len(start_indices)
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        idx = start_indices[i]
        if idx + h > len(prices):
            continue
        peak = prices[idx]
        best = 0.0
        for j in range(1, h):
            v = prices[idx + j]
            if v > peak:
                peak = v
            dd = (v - peak) / peak
            if dd < best:
                best = dd
        out[i] = best
    return out


@numba.jit(nopython=True, cache=True)
def _runups_batch(
    prices: NDArray[np.float64],
    start_indices: NDArray[np.int64],
    h: int,
) -> NDArray[np.float64]:
    n = len(start_indices)
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        idx = start_indices[i]
        if idx + h > len(prices):
            continue
        trough = prices[idx]
        best = 0.0
        for j in range(1, h):
            v = prices[idx + j]
            if v < trough:
                trough = v
            ru = (v - trough) / trough
            if ru > best:
                best = ru
        out[i] = best
    return out


@numba.jit(nopython=True, cache=True)
def _range_batch(
    prices: NDArray[np.float64],
    start_indices: NDArray[np.int64],
    h: int,
) -> NDArray[np.float64]:
    n = len(start_indices)
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        idx = start_indices[i]
        if idx + h > len(prices):
            continue
        lo = prices[idx]
        hi = prices[idx]
        for j in range(1, h):
            v = prices[idx + j]
            if v < lo:
                lo = v
            if v > hi:
                hi = v
        out[i] = (hi - lo) / prices[idx]
    return out


@numba.jit(nopython=True, cache=True)
def _volatility_batch(
    returns: NDArray[np.float64],
    start_indices: NDArray[np.int64],
    h: int,
) -> NDArray[np.float64]:
    n = len(start_indices)
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        idx = start_indices[i]
        if idx + h > len(returns):
            continue
        seg = returns[idx : idx + h]
        out[i] = np.std(seg) if len(seg) > 1 else 0.0
    return out


@numba.jit(nopython=True, cache=True)
def _time_to_return_threshold_batch(
    returns: NDArray[np.float64],
    start_indices: NDArray[np.int64],
    threshold: float,
    max_h: int,
) -> NDArray[np.float64]:
    n = len(start_indices)
    out = np.full(n, float(max_h), dtype=np.float64)
    N = len(returns)
    for i in range(n):
        idx = start_indices[i]
        cum = 0.0
        end = min(idx + max_h, N)
        for j in range(idx, end):
            cum += returns[j]
            if cum >= threshold:
                out[i] = float(j - idx + 1)
                break
    return out


@numba.jit(nopython=True, cache=True)
def _time_to_drawdown_threshold_batch(
    prices: NDArray[np.float64],
    start_indices: NDArray[np.int64],
    threshold: float,
    max_h: int,
) -> NDArray[np.float64]:
    n = len(start_indices)
    out = np.full(n, float(max_h), dtype=np.float64)
    N = len(prices)
    for i in range(n):
        idx = start_indices[i]
        peak = prices[idx]
        end = min(idx + max_h, N)
        for j in range(idx + 1, end):
            if prices[j] > peak:
                peak = prices[j]
            dd = (prices[j] - peak) / peak
            if dd <= threshold:
                out[i] = float(j - idx)
                break
    return out


@numba.jit(nopython=True, cache=True)
def _time_to_state_change_batch(
    states: NDArray[np.int64],
    start_indices: NDArray[np.int64],
    max_h: int,
) -> NDArray[np.float64]:
    n = len(start_indices)
    out = np.full(n, float(max_h), dtype=np.float64)
    N = len(states)
    for i in range(n):
        idx = start_indices[i]
        cur = states[idx]
        end = min(idx + max_h, N)
        for j in range(idx + 1, end):
            if states[j] != cur:
                out[i] = float(j - idx)
                break
    return out


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class BucketOutcomes:
    count: int
    future_returns: dict[int, dict[str, float]] = field(default_factory=dict)
    drawdowns: dict[int, float] = field(default_factory=dict)
    runups: dict[int, float] = field(default_factory=dict)
    expected_range: float = 0.0
    expected_drawdown: float = 0.0
    expected_volatility: float = 0.0
    time_to_target: float = 0.0
    time_to_drawdown: float = 0.0
    time_to_state_change: float = 0.0
    distribution_shift: float = 0.0
    outcome_separation: float = 0.0
    information_gain: float = 0.0


@dataclass
class OutcomeDistributionReport:
    asset: str
    n_samples: int
    buckets: dict[str, BucketOutcomes] = field(default_factory=dict)
    outcome_separation_avg: float = 0.0
    information_gain_avg: float = 0.0
    verdict: str = "outcomes_do_not_change"


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class OutcomeDistributionAnalyzer:
    """Investigate RQ1, RQ4, RQ7, RQ8.

    RQ1 — Outcome Distribution Conditioning:
        For each adaptive-time bucket compute forward-looking return
        moments (mean, std, skew, kurtosis), drawdown and runup at
        multiple horizons.

    RQ4 — Direction-Neutral Test:
        Measure distribution shift (KS), outcome separation (mean
        distance between adjacent buckets) and information gain (MI
        between bucket label and outcome magnitude) — all direction
        agnostic.

    RQ7 — Position Sizing Relevance:
        Expected range, drawdown and volatility per bucket.

    RQ8 — Holding Period Analysis:
        Average steps to reach a target return, a drawdown threshold,
        or a state change, per bucket.
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
        states: NDArray[np.int64],
        price: NDArray[np.float64],
        asset: str = "unknown",
    ) -> OutcomeDistributionReport:
        at = np.asarray(adaptive_time, dtype=np.float64).ravel()
        r = np.asarray(returns, dtype=np.float64).ravel()
        s = np.asarray(states, dtype=np.int64).ravel()
        p = np.asarray(price, dtype=np.float64).ravel()

        n = len(at)
        if n == 0:
            return self._empty_report(asset)

        # 1. Bucket adaptive time
        bucket_idx = self._bucket(at)

        # 2. Pre-compute cumulative returns for fast horizon sums
        cum_r = np.zeros(n + 1, dtype=np.float64)
        cum_r[1:] = np.cumsum(r)

        max_h = max(self.horizons)
        global_valid = np.arange(n, dtype=np.int64)
        global_valid = global_valid[global_valid + max_h < n]

        # Per-bucket storage
        raw_outcomes: dict[str, NDArray[np.float64]] = {}
        buckets: dict[str, BucketOutcomes] = {}

        for b in range(self.n_buckets):
            label = self._bucket_label(b)
            mask = bucket_idx == b
            count = int(mask.sum())
            starts = np.where(mask)[0].astype(np.int64)

            if count < 3:
                buckets[label] = BucketOutcomes(count=count)
                raw_outcomes[label] = np.zeros(0, dtype=np.float64)
                continue

            fr_dict: dict[int, dict[str, float]] = {}
            dd_dict: dict[int, float] = {}
            ru_dict: dict[int, float] = {}

            all_outcomes: list[NDArray[np.float64]] = []

            for h in self.horizons:
                h_valid = starts[starts + h < n]
                if len(h_valid) < 2:
                    continue

                # Future returns via cumsum
                fr_vals = cum_r[h_valid + h] - cum_r[h_valid]
                mean, std, skew, kurt = _bucket_future_return_stats(r, h_valid, h)
                fr_dict[h] = {"mean": mean, "std": std, "skew": skew, "kurt": kurt}
                all_outcomes.append(fr_vals)

                # Drawdowns / runups
                dd_arr = _drawdowns_batch(p, h_valid, h)
                ru_arr = _runups_batch(p, h_valid, h)
                dd_dict[h] = float(np.mean(dd_arr))
                ru_dict[h] = float(np.mean(ru_arr))

            if all_outcomes:
                raw_outcomes[label] = np.concatenate(all_outcomes)
            else:
                raw_outcomes[label] = np.zeros(0, dtype=np.float64)

            # RQ7 — position sizing metrics
            range_vals: list[float] = []
            dd_vals: list[float] = []
            vol_vals: list[float] = []

            for h in self.horizons:
                h_valid = starts[starts + h < n]
                if len(h_valid) < 2:
                    continue
                rng_arr = _range_batch(p, h_valid, h)
                d_arr = _drawdowns_batch(p, h_valid, h)
                v_arr = _volatility_batch(r, h_valid, h)
                range_vals.append(float(np.mean(rng_arr)))
                dd_vals.append(float(np.mean(d_arr)))
                vol_vals.append(float(np.mean(v_arr)))

            expected_range = float(np.mean(range_vals)) if range_vals else 0.0
            expected_drawdown = float(np.mean(dd_vals)) if dd_vals else 0.0
            expected_volatility = float(np.mean(vol_vals)) if vol_vals else 0.0

            # RQ8 — holding period metrics
            tt_target = _time_to_return_threshold_batch(r, starts, 0.01, max_h)
            tt_dd = _time_to_drawdown_threshold_batch(p, starts, -0.01, max_h)
            tt_sc = _time_to_state_change_batch(s, starts, max_h)
            time_to_target = float(np.mean(tt_target))
            time_to_drawdown = float(np.mean(tt_dd))
            time_to_state_change = float(np.mean(tt_sc))

            buckets[label] = BucketOutcomes(
                count=count,
                future_returns=fr_dict,
                drawdowns=dd_dict,
                runups=ru_dict,
                expected_range=expected_range,
                expected_drawdown=expected_drawdown,
                expected_volatility=expected_volatility,
                time_to_target=time_to_target,
                time_to_drawdown=time_to_drawdown,
                time_to_state_change=time_to_state_change,
            )

        # 3. Cross-bucket RQ4 metrics
        labels = list(buckets.keys())
        median_label = "medium" if "medium" in buckets else (labels[len(labels) // 2] if labels else None)

        for i, label in enumerate(labels):
            bkt = buckets[label]

            # Distribution shift — KS from median bucket
            if median_label is not None and label != median_label:
                med = buckets[median_label]
                x = raw_outcomes.get(label, np.zeros(0))
                y = raw_outcomes.get(median_label, np.zeros(0))
                if len(x) > 1 and len(y) > 1:
                    bkt.distribution_shift = self._ks_statistic(x, y)

            # Outcome separation — distance from adjacent bucket means
            adj_dists: list[float] = []
            if i > 0:
                prev = buckets[labels[i - 1]]
                d_prev = self._mean_outcome_distance(bkt, prev)
                if d_prev >= 0.0:
                    adj_dists.append(d_prev)
            if i < len(labels) - 1:
                nxt = buckets[labels[i + 1]]
                d_nxt = self._mean_outcome_distance(bkt, nxt)
                if d_nxt >= 0.0:
                    adj_dists.append(d_nxt)
            bkt.outcome_separation = float(np.mean(adj_dists)) if adj_dists else 0.0

            # Information gain — MI(bucket_label, outcome_value)
            outcomes_all = raw_outcomes.get(label, np.zeros(0))
            if len(outcomes_all) > 1:
                other_labels = [l for l in labels if l != label]
                if other_labels:
                    other_outcomes = np.concatenate([raw_outcomes.get(l, np.zeros(0)) for l in other_labels])
                    if len(other_outcomes) > 1:
                        n_total = len(outcomes_all) + len(other_outcomes)
                        bucket_ids = np.concatenate([
                            np.full(len(outcomes_all), i, dtype=np.int64),
                            np.full(len(other_outcomes), -1, dtype=np.int64),
                        ])
                        combined = np.concatenate([outcomes_all, other_outcomes])
                        bkt.information_gain = self._discrete_mi(
                            bucket_ids.astype(np.float64),
                            np.abs(combined),
                            n_bins=10,
                        )

        # 4. Report-level averages
        sep_vals = [b.outcome_separation for b in buckets.values()]
        ig_vals = [b.information_gain for b in buckets.values()]

        outcome_separation_avg = float(np.mean(sep_vals)) if sep_vals else 0.0
        information_gain_avg = float(np.mean(ig_vals)) if ig_vals else 0.0

        report = OutcomeDistributionReport(
            asset=asset,
            n_samples=n,
            buckets=buckets,
            outcome_separation_avg=outcome_separation_avg,
            information_gain_avg=information_gain_avg,
            verdict="",
        )
        report.verdict = self._determine_verdict(report)
        return report

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
    # Outcome helpers
    # ------------------------------------------------------------------

    def _future_outcomes(
        self,
        returns: NDArray[np.float64],
        idx: int,
        h: int,
    ) -> dict[str, float]:
        if idx + h > len(returns):
            return {"cumulative_return": 0.0}
        return {"cumulative_return": float(np.sum(returns[idx: idx + h]))}

    @staticmethod
    def _compute_drawdown(prices: NDArray[np.float64]) -> float:
        p = np.asarray(prices, dtype=np.float64).ravel()
        if len(p) < 2:
            return 0.0
        peak = p[0]
        best = 0.0
        for i in range(1, len(p)):
            if p[i] > peak:
                peak = p[i]
            dd = (p[i] - peak) / peak
            if dd < best:
                best = dd
        return best

    @staticmethod
    def _compute_runup(prices: NDArray[np.float64]) -> float:
        p = np.asarray(prices, dtype=np.float64).ravel()
        if len(p) < 2:
            return 0.0
        trough = p[0]
        best = 0.0
        for i in range(1, len(p)):
            if p[i] < trough:
                trough = p[i]
            ru = (p[i] - trough) / trough
            if ru > best:
                best = ru
        return best

    # ------------------------------------------------------------------
    # Cross-bucket comparison helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ks_statistic(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
        from research.temporal_reality.universality import _ks_statistic_impl
        return float(_ks_statistic_impl(x, y))

    @staticmethod
    def _discrete_mi(
        buckets: NDArray[np.float64],
        outcomes: NDArray[np.float64],
        n_bins: int = 10,
    ) -> float:
        from research.information_discovery.mi_estimator import _fast_mutual_info
        valid = ~(np.isnan(outcomes) | np.isinf(outcomes))
        if valid.sum() < 2:
            return 0.0
        return float(_fast_mutual_info(
            outcomes[valid].astype(np.float64),
            buckets[valid].astype(np.float64),
            n_bins,
        ))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _bucket_label(self, b: int) -> str:
        if b < len(self.bucket_labels):
            return self.bucket_labels[b]
        return f"bucket_{b}"

    @staticmethod
    def _mean_outcome_distance(
        a: BucketOutcomes,
        b: BucketOutcomes,
    ) -> float:
        shared = [h for h in a.future_returns if h in b.future_returns]
        if not shared:
            return -1.0
        dists = []
        for h in shared:
            ma = a.future_returns[h].get("mean", 0.0)
            mb = b.future_returns[h].get("mean", 0.0)
            dists.append(abs(ma - mb))
        return float(np.mean(dists)) if dists else -1.0

    def _determine_verdict(
        self,
        report: OutcomeDistributionReport,
    ) -> str:
        buckets = report.buckets
        if len(buckets) < 2:
            return "outcomes_do_not_change"

        # Check if outcome_separation is consistently positive
        sep_scores = [b.outcome_separation for b in buckets.values()]
        ig_scores = [b.information_gain for b in buckets.values()]
        shift_scores = [b.distribution_shift for b in buckets.values() if b.distribution_shift > 0.0]

        avg_sep = float(np.mean(sep_scores)) if sep_scores else 0.0
        avg_ig = float(np.mean(ig_scores)) if ig_scores else 0.0
        avg_shift = float(np.mean(shift_scores)) if shift_scores else 0.0

        evidence = 0
        if avg_sep > 0.01:
            evidence += 1
        if avg_ig > 0.05:
            evidence += 1
        if avg_shift > 0.15:
            evidence += 1

        if evidence >= 2:
            return "outcomes_change_materially"
        if evidence >= 1:
            return "mixed"
        return "outcomes_do_not_change"

    def _empty_report(self, asset: str) -> OutcomeDistributionReport:
        labels = [self._bucket_label(b) for b in range(self.n_buckets)]
        empty = {lb: BucketOutcomes(count=0) for lb in labels}
        return OutcomeDistributionReport(
            asset=asset,
            n_samples=0,
            buckets=empty,
            verdict="outcomes_do_not_change",
        )
