from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numba
import numpy as np
from numpy.typing import NDArray

from research.information_discovery.mi_estimator import (
    _fast_digitize,
    _fast_entropy_digitized,
    _fast_mutual_info,
    _fast_percentile,
)


@dataclass
class BucketRisk:
    count: int
    future_volatility: dict  # horizon -> float
    future_entropy: dict  # horizon -> float
    future_state_mutation: dict  # horizon -> float
    future_regime_change: dict  # horizon -> float
    risk_score: float  # composite risk score (higher = riskier)


@dataclass
class RiskProfileReport:
    asset: str
    n_samples: int
    buckets: dict  # bucket_name -> BucketRisk
    verdict: str  # "adaptive_time_alters_risk", "risk_does_not_change", "monotonic_risk"


@numba.jit(nopython=True, cache=True)
def _bucket_assign(adaptive_time: NDArray[np.float64], n_buckets: int) -> NDArray[np.int32]:
    n = len(adaptive_time)
    out = np.zeros(n, dtype=np.int32)
    sorted_at = np.sort(adaptive_time)
    edges = np.zeros(n_buckets - 1)
    for b in range(n_buckets - 1):
        idx = int((b + 1) * n // n_buckets)
        if idx >= n:
            idx = n - 1
        edges[b] = sorted_at[idx]
    for i in range(n):
        v = adaptive_time[i]
        assigned = n_buckets - 1
        for b in range(n_buckets - 1):
            if v <= edges[b]:
                assigned = b
                break
        out[i] = assigned
    return out


@numba.jit(nopython=True, cache=True)
def _forward_metrics(
    returns: NDArray[np.float64],
    states: NDArray[np.int32],
    horizons: NDArray[np.int32],
    entropy_bins: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    n = len(returns)
    n_h = len(horizons)
    vol = np.zeros((n_h, n))
    ent = np.zeros((n_h, n))
    mut = np.zeros((n_h, n))
    reg = np.zeros((n_h, n))
    for h_idx in range(n_h):
        h = horizons[h_idx]
        for i in range(n):
            start = min(i + 1, n)
            end = min(i + 1 + h, n)
            seg_r = returns[start:end]
            seg_s = states[start:end]
            seg_len = end - start
            if seg_len < 1:
                continue
            vol[h_idx, i] = _seg_volatility(seg_r)
            ent[h_idx, i] = _seg_entropy(seg_r, entropy_bins)
            mut[h_idx, i] = _seg_mutation(seg_s)
            reg[h_idx, i] = _seg_regime_change(seg_s, states[i])
    return vol, ent, mut, reg


@numba.jit(nopython=True, cache=True)
def _seg_volatility(seg: NDArray[np.float64]) -> float:
    seg_len = len(seg)
    if seg_len < 2:
        return 0.0
    m = 0.0
    for j in range(seg_len):
        m += seg[j]
    m /= seg_len
    var = 0.0
    for j in range(seg_len):
        d = seg[j] - m
        var += d * d
    return np.sqrt(var / (seg_len - 1))


@numba.jit(nopython=True, cache=True)
def _seg_entropy(seg: NDArray[np.float64], n_bins: int) -> float:
    seg_len = len(seg)
    if seg_len < 2:
        return 0.0
    lo = seg[0]
    hi = seg[0]
    for j in range(1, seg_len):
        if seg[j] < lo:
            lo = seg[j]
        if seg[j] > hi:
            hi = seg[j]
    if hi - lo < 1e-12:
        return 0.0
    q = np.linspace(0.0, 1.0, n_bins + 1)
    bins = _fast_percentile(seg, q)
    uq = np.zeros(n_bins + 1, dtype=np.float64)
    uq_len = 0
    for j in range(n_bins + 1):
        is_dup = False
        for k in range(uq_len):
            if abs(bins[j] - uq[k]) < 1e-12:
                is_dup = True
                break
        if not is_dup:
            uq[uq_len] = bins[j]
            uq_len += 1
    if uq_len < 2:
        return 0.0
    useful_bins = uq[:uq_len]
    dig = _fast_digitize(seg, useful_bins)
    return _fast_entropy_digitized(dig, uq_len - 1)


@numba.jit(nopython=True, cache=True)
def _seg_mutation(seg: NDArray[np.int32]) -> float:
    seg_len = len(seg)
    if seg_len < 2:
        return 0.0
    trans = 0
    for j in range(1, seg_len):
        if seg[j] != seg[j - 1] and seg[j] >= 0 and seg[j - 1] >= 0:
            trans += 1
    return trans / (seg_len - 1)


@numba.jit(nopython=True, cache=True)
def _seg_regime_change(seg: NDArray[np.int32], current_state: np.int32) -> float:
    seg_len = len(seg)
    if seg_len < 1 or current_state < 0:
        return 0.0
    changes = 0
    for j in range(seg_len):
        if seg[j] != current_state and seg[j] >= 0:
            changes += 1
    return changes / seg_len


@numba.jit(nopython=True, cache=True)
def _aggregate_by_bucket(
    bucket_ids: NDArray[np.int32],
    metrics: NDArray[np.float64],
    n_buckets: int,
) -> NDArray[np.float64]:
    out = np.zeros(n_buckets)
    counts = np.zeros(n_buckets, dtype=np.int32)
    for i in range(len(bucket_ids)):
        b = bucket_ids[i]
        if b >= 0 and b < n_buckets:
            out[b] += metrics[i]
            counts[b] += 1
    for b in range(n_buckets):
        if counts[b] > 0:
            out[b] /= counts[b]
    return out


class RiskProfileAnalyzer:
    def __init__(self, n_buckets: int = 5, horizons: list | None = None):
        self.n_buckets = n_buckets
        self.horizons = horizons or [1, 5, 20, 50, 100, 500]
        self.bucket_labels = ["very_low", "low", "medium", "high", "extreme"]

    def compute(
        self,
        adaptive_time: NDArray,
        returns: NDArray,
        states: NDArray,
        state_mutation_rate: NDArray = None,
    ) -> RiskProfileReport:
        at = np.asarray(adaptive_time, dtype=np.float64)
        r = np.asarray(returns, dtype=np.float64)
        s = np.asarray(states, dtype=np.int32)

        n = len(at)
        if n < 10:
            return RiskProfileReport(
                asset="unknown", n_samples=n, buckets={}, verdict="insufficient_data"
            )

        bucket_ids = _bucket_assign(at, self.n_buckets)
        horizon_arr = np.array(self.horizons, dtype=np.int32)
        vol_all, ent_all, mut_all, reg_all = _forward_metrics(
            r, s, horizon_arr, entropy_bins=10
        )

        per_horizon_vol = np.zeros((self.n_buckets, len(self.horizons)))
        per_horizon_ent = np.zeros((self.n_buckets, len(self.horizons)))
        per_horizon_mut = np.zeros((self.n_buckets, len(self.horizons)))
        per_horizon_reg = np.zeros((self.n_buckets, len(self.horizons)))
        bucket_counts = np.zeros(self.n_buckets, dtype=np.int32)
        for i in range(self.n_buckets):
            bucket_counts[i] = np.sum(bucket_ids == i)

        for h_idx in range(len(self.horizons)):
            per_horizon_vol[:, h_idx] = _aggregate_by_bucket(
                bucket_ids, vol_all[h_idx], self.n_buckets
            )
            per_horizon_ent[:, h_idx] = _aggregate_by_bucket(
                bucket_ids, ent_all[h_idx], self.n_buckets
            )
            per_horizon_mut[:, h_idx] = _aggregate_by_bucket(
                bucket_ids, mut_all[h_idx], self.n_buckets
            )
            per_horizon_reg[:, h_idx] = _aggregate_by_bucket(
                bucket_ids, reg_all[h_idx], self.n_buckets
            )

        buckets: dict[str, BucketRisk] = {}
        for b in range(self.n_buckets):
            label = self.bucket_labels[b] if b < len(self.bucket_labels) else str(b)
            fv = {str(h): float(per_horizon_vol[b, h_idx]) for h_idx, h in enumerate(self.horizons)}
            fe = {str(h): float(per_horizon_ent[b, h_idx]) for h_idx, h in enumerate(self.horizons)}
            fm = {str(h): float(per_horizon_mut[b, h_idx]) for h_idx, h in enumerate(self.horizons)}
            fr = {str(h): float(per_horizon_reg[b, h_idx]) for h_idx, h in enumerate(self.horizons)}

            all_vals = np.array([
                per_horizon_vol[b, :].mean(),
                per_horizon_ent[b, :].mean(),
                per_horizon_mut[b, :].mean(),
                per_horizon_reg[b, :].mean(),
            ])
            risk_score = float(np.mean(all_vals))

            buckets[label] = BucketRisk(
                count=int(bucket_counts[b]),
                future_volatility=fv,
                future_entropy=fe,
                future_state_mutation=fm,
                future_regime_change=fr,
                risk_score=risk_score,
            )

        report = RiskProfileReport(
            asset="unknown",
            n_samples=n,
            buckets=buckets,
            verdict="unknown",
        )
        report.verdict = self._determine_verdict(report)
        return report

    def _bucket(self, adaptive_time: NDArray) -> NDArray:
        return _bucket_assign(
            np.asarray(adaptive_time, dtype=np.float64), self.n_buckets
        )

    def _determine_verdict(self, report: RiskProfileReport) -> str:
        scores = []
        for label in self.bucket_labels[: self.n_buckets]:
            if label in report.buckets:
                scores.append(report.buckets[label].risk_score)

        if len(scores) < 2:
            return "insufficient_data"

        score_arr = np.array(scores)
        min_s = score_arr.min()
        max_s = score_arr.max()
        mean_s = score_arr.mean()
        if mean_s < 1e-12:
            mean_s = 1.0
        relative_range = (max_s - min_s) / mean_s if mean_s > 0 else 0.0

        ranks = np.arange(len(score_arr), dtype=np.float64)
        rho = np.corrcoef(ranks, score_arr)[0, 1] if len(score_arr) > 2 else 0.0
        if np.isnan(rho):
            rho = 0.0

        if relative_range < 0.05:
            return "risk_does_not_change"
        if abs(rho) > 0.7 and relative_range > 0.1:
            return "monotonic_risk"
        return "adaptive_time_alters_risk"

    @staticmethod
    @numba.jit(nopython=True, cache=True)
    def _rolling_volatility(returns: NDArray, window: int) -> NDArray:
        n = len(returns)
        out = np.zeros(n)
        for i in range(n):
            start = max(0, i - window + 1)
            seg = returns[start : i + 1]
            seg_len = len(seg)
            if seg_len < 2:
                out[i] = 0.0
                continue
            seg = seg[~np.isnan(seg)]
            if len(seg) < 2:
                out[i] = 0.0
                continue
            m = 0.0
            for j in range(len(seg)):
                m += seg[j]
            m /= len(seg)
            var = 0.0
            for j in range(len(seg)):
                d = seg[j] - m
                var += d * d
            out[i] = np.sqrt(var / (len(seg) - 1))
        return out

    @staticmethod
    @numba.jit(nopython=True, cache=True)
    def _rolling_entropy(arr: NDArray, window: int, n_bins: int) -> NDArray:
        n = len(arr)
        out = np.zeros(n)
        for i in range(n):
            start = max(0, i - window + 1)
            seg = arr[start : i + 1]
            seg = seg[~np.isnan(seg)]
            if len(seg) < 2:
                out[i] = 0.0
                continue
            lo = seg[0]
            hi = seg[0]
            for j in range(1, len(seg)):
                if seg[j] < lo:
                    lo = seg[j]
                if seg[j] > hi:
                    hi = seg[j]
            if hi - lo < 1e-12:
                out[i] = 0.0
                continue
            q = np.linspace(0.0, 1.0, n_bins + 1)
            bins = _fast_percentile(seg, q)
            uq = np.zeros(n_bins + 1, dtype=np.float64)
            uq_len = 0
            for j in range(n_bins + 1):
                is_dup = False
                for k in range(uq_len):
                    if abs(bins[j] - uq[k]) < 1e-12:
                        is_dup = True
                        break
                if not is_dup:
                    uq[uq_len] = bins[j]
                    uq_len += 1
            if uq_len < 2:
                out[i] = 0.0
                continue
            useful = uq[:uq_len]
            dig = _fast_digitize(seg, useful)
            out[i] = _fast_entropy_digitized(dig, uq_len - 1)
        return out

    @staticmethod
    @numba.jit(nopython=True, cache=True)
    def _rolling_state_mutation(states: NDArray, window: int) -> NDArray:
        n = len(states)
        out = np.zeros(n)
        for i in range(n):
            start = max(0, i - window + 1)
            seg = states[start : i + 1]
            seg_len = len(seg)
            if seg_len < 2:
                out[i] = 0.0
                continue
            trans = 0
            for j in range(1, seg_len):
                if seg[j] != seg[j - 1] and seg[j] >= 0 and seg[j - 1] >= 0:
                    trans += 1
            out[i] = trans / (seg_len - 1)
        return out
