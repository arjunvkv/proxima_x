from __future__ import annotations

from typing import Any

import numpy as np
import numba
from numpy.typing import NDArray
from scipy.stats import skew, kurtosis, f_oneway
from sklearn.metrics import mutual_info_score


@numba.jit(nopython=True, cache=True)
def _forward_returns_numba(price: NDArray[np.float64], horizon: int) -> NDArray[np.float64]:
    n = len(price)
    result = np.empty(n, dtype=np.float64)
    for i in range(n):
        j = i + horizon
        if j < n:
            result[i] = np.log(price[j] / price[i])
        else:
            result[i] = np.nan
    return result


@numba.jit(nopython=True, cache=True)
def _forward_volatility_numba(returns: NDArray[np.float64], horizon: int, vol_window: int) -> NDArray[np.float64]:
    n = len(returns)
    result = np.empty(n, dtype=np.float64)
    for i in range(n):
        end = min(i + horizon, n)
        wlen = end - i
        if wlen < 2:
            result[i] = np.nan
        else:
            seg = returns[i:end]
            s = 0.0
            for k in range(wlen):
                s += seg[k] * seg[k]
            result[i] = np.sqrt(s / (wlen - 1))
    return result


@numba.jit(nopython=True, cache=True)
def _forward_entropy_numba(returns: NDArray[np.float64], horizon: int, bins: int) -> NDArray[np.float64]:
    n = len(returns)
    result = np.empty(n, dtype=np.float64)
    for i in range(n):
        end = min(i + horizon, n)
        seg = returns[i:end]
        if len(seg) < 2:
            result[i] = np.nan
            continue
        lo = np.min(seg)
        hi = np.max(seg)
        if hi - lo < 1e-15:
            result[i] = 0.0
            continue
        bin_edges = np.linspace(lo, hi, bins + 1)
        counts = np.zeros(bins, dtype=np.int64)
        for k in range(len(seg)):
            v = seg[k]
            if v < bin_edges[0] or v >= bin_edges[-1]:
                continue
            for b in range(bins):
                if bin_edges[b] <= v < bin_edges[b + 1]:
                    counts[b] += 1
                    break
        total = np.sum(counts)
        if total < 1:
            result[i] = 0.0
            continue
        h = 0.0
        for b in range(bins):
            if counts[b] > 0:
                p = counts[b] / total
                h -= p * np.log(p)
        result[i] = h
    return result


@numba.jit(nopython=True, cache=True)
def _forward_max_drawdown_numba(price: NDArray[np.float64], horizon: int) -> NDArray[np.float64]:
    n = len(price)
    result = np.empty(n, dtype=np.float64)
    for i in range(n):
        end = min(i + horizon, n)
        seg = price[i:end]
        if len(seg) < 2:
            result[i] = np.nan
            continue
        peak = seg[0]
        max_dd = 0.0
        for k in range(1, len(seg)):
            if seg[k] > peak:
                peak = seg[k]
            dd = (peak - seg[k]) / peak
            if dd > max_dd:
                max_dd = dd
        result[i] = max_dd
    return result


@numba.jit(nopython=True, cache=True)
def _forward_sharpe_numba(price: NDArray[np.float64], horizon: int) -> NDArray[np.float64]:
    n = len(price)
    result = np.empty(n, dtype=np.float64)
    for i in range(n):
        end = min(i + horizon, n)
        seg = price[i:end]
        if len(seg) < 3:
            result[i] = np.nan
            continue
        log_rets = np.empty(len(seg) - 1, dtype=np.float64)
        for k in range(len(seg) - 1):
            log_rets[k] = np.log(seg[k + 1] / seg[k])
        mean_r = 0.0
        for k in range(len(log_rets)):
            mean_r += log_rets[k]
        mean_r /= len(log_rets)
        var_r = 0.0
        for k in range(len(log_rets)):
            d = log_rets[k] - mean_r
            var_r += d * d
        var_r /= (len(log_rets) - 1)
        std_r = np.sqrt(var_r)
        if std_r < 1e-15:
            result[i] = 0.0
        else:
            result[i] = (mean_r / std_r) * np.sqrt(252.0)
    return result


@numba.jit(nopython=True, cache=True)
def _conditional_stats_numba(
    forward_vals: NDArray[np.float64],
    states: NDArray[np.int32],
    unique_states: NDArray[np.int32],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    n_states = len(unique_states)
    n = len(forward_vals)
    means = np.zeros(n_states, dtype=np.float64)
    stds = np.zeros(n_states, dtype=np.float64)
    skws = np.zeros(n_states, dtype=np.float64)
    kurts = np.zeros(n_states, dtype=np.float64)
    p5 = np.zeros(n_states, dtype=np.float64)
    p25 = np.zeros(n_states, dtype=np.float64)
    p50 = np.zeros(n_states, dtype=np.float64)
    p75 = np.zeros(n_states, dtype=np.float64)
    p95 = np.zeros(n_states, dtype=np.float64)
    for s_idx in range(n_states):
        s = unique_states[s_idx]
        count = 0
        for i in range(n):
            if states[i] == s and not np.isnan(forward_vals[i]):
                count += 1
        if count < 2:
            means[s_idx] = np.nan
            stds[s_idx] = np.nan
            skws[s_idx] = np.nan
            kurts[s_idx] = np.nan
            p5[s_idx] = np.nan
            p25[s_idx] = np.nan
            p50[s_idx] = np.nan
            p75[s_idx] = np.nan
            p95[s_idx] = np.nan
            continue
        vals = np.empty(count, dtype=np.float64)
        j = 0
        for i in range(n):
            if states[i] == s and not np.isnan(forward_vals[i]):
                vals[j] = forward_vals[i]
                j += 1
        sorted_vals = np.sort(vals)
        m = np.mean(vals)
        means[s_idx] = m
        d = vals - m
        v = np.sum(d * d) / (len(vals) - 1)
        stds[s_idx] = np.sqrt(v)
        if v > 1e-15:
            s3 = np.sum(d * d * d) / len(vals)
            skws[s_idx] = s3 / (v ** 1.5)
            s4 = np.sum(d * d * d * d) / len(vals)
            kurts[s_idx] = s4 / (v * v) - 3.0
        nv = len(sorted_vals)
        p5[s_idx] = sorted_vals[int(nv * 0.05)]
        p25[s_idx] = sorted_vals[int(nv * 0.25)]
        p50[s_idx] = sorted_vals[int(nv * 0.50)]
        p75[s_idx] = sorted_vals[int(nv * 0.75)]
        p95[s_idx] = sorted_vals[int(nv * 0.95)]
    return means, stds, skws, kurts, p5, p25, p50, p75, p95


@numba.jit(nopython=True, cache=True)
def _eta_squared_numba(
    forward_vals: NDArray[np.float64],
    states: NDArray[np.int32],
    unique_states: NDArray[np.int32],
) -> float:
    n = len(forward_vals)
    grand_mean = 0.0
    count = 0
    for i in range(n):
        if not np.isnan(forward_vals[i]):
            grand_mean += forward_vals[i]
            count += 1
    if count < 2:
        return 0.0
    grand_mean /= count
    ss_between = 0.0
    ss_total = 0.0
    for s_idx in range(len(unique_states)):
        s = unique_states[s_idx]
        s_count = 0
        s_sum = 0.0
        for i in range(n):
            if states[i] == s and not np.isnan(forward_vals[i]):
                s_sum += forward_vals[i]
                s_count += 1
        if s_count > 0:
            s_mean = s_sum / s_count
            ss_between += s_count * (s_mean - grand_mean) ** 2
    for i in range(n):
        if not np.isnan(forward_vals[i]):
            ss_total += (forward_vals[i] - grand_mean) ** 2
    if ss_total < 1e-15:
        return 0.0
    return ss_between / ss_total


class ForwardAnalyzer:
    def __init__(self, horizons: list[int] | None = None):
        if horizons is None:
            horizons = [1, 5, 20, 50, 100, 500]
        self.horizons = horizons

    def compute_forward_returns(self, price: NDArray[np.float64], horizon: int) -> NDArray[np.float64]:
        return _forward_returns_numba(price.astype(np.float64), horizon)

    def compute_forward_volatility(self, returns: NDArray[np.float64], horizon: int, vol_window: int = 20) -> NDArray[np.float64]:
        return _forward_volatility_numba(returns.astype(np.float64), horizon, vol_window)

    def compute_forward_entropy(self, returns: NDArray[np.float64], horizon: int, bins: int = 20) -> NDArray[np.float64]:
        return _forward_entropy_numba(returns.astype(np.float64), horizon, bins)

    def compute_forward_max_drawdown(self, price: NDArray[np.float64], horizon: int) -> NDArray[np.float64]:
        return _forward_max_drawdown_numba(price.astype(np.float64), horizon)

    def compute_forward_sharpe(self, price: NDArray[np.float64], horizon: int) -> NDArray[np.float64]:
        return _forward_sharpe_numba(price.astype(np.float64), horizon)

    def compute_all_forward_metrics(self, price: NDArray[np.float64], returns: NDArray[np.float64]) -> dict:
        metrics: dict[str, NDArray[np.float64]] = {}
        for h in self.horizons:
            metrics[f"forward_return_{h}"] = self.compute_forward_returns(price, h)
            metrics[f"forward_vol_{h}"] = self.compute_forward_volatility(returns, h)
            metrics[f"forward_entropy_{h}"] = self.compute_forward_entropy(returns, h)
            metrics[f"forward_max_dd_{h}"] = self.compute_forward_max_drawdown(price, h)
            metrics[f"forward_sharpe_{h}"] = self.compute_forward_sharpe(price, h)
        return metrics

    def compute_conditional_distribution(
        self,
        states: NDArray[np.int32],
        forward_metrics: dict,
        horizon: int,
    ) -> dict:
        h_key = None
        for k in forward_metrics:
            if f"_{horizon}" in k:
                h_key = k
                break
        if h_key is None:
            raise ValueError(f"No forward metric found for horizon {horizon}")
        fwd = forward_metrics[h_key]
        unique_states = np.unique(states)
        valid = ~np.isnan(fwd)
        s_valid = states[valid]
        f_valid = fwd[valid].astype(np.float64)
        uq = np.unique(s_valid)
        means, stds, skws, kurts, p5, p25, p50, p75, p95 = _conditional_stats_numba(f_valid, s_valid, uq)
        result: dict[int, dict[str, float]] = {}
        for i in range(len(uq)):
            sid = int(uq[i])
            result[sid] = {
                "mean": float(means[i]) if not np.isnan(means[i]) else 0.0,
                "std": float(stds[i]) if not np.isnan(stds[i]) else 0.0,
                "skew": float(skws[i]) if not np.isnan(skws[i]) else 0.0,
                "kurtosis": float(kurts[i]) if not np.isnan(kurts[i]) else 0.0,
                "p5": float(p5[i]) if not np.isnan(p5[i]) else 0.0,
                "p25": float(p25[i]) if not np.isnan(p25[i]) else 0.0,
                "p50": float(p50[i]) if not np.isnan(p50[i]) else 0.0,
                "p75": float(p75[i]) if not np.isnan(p75[i]) else 0.0,
                "p95": float(p95[i]) if not np.isnan(p95[i]) else 0.0,
            }
        return result

    def analyze_state_separation(
        self,
        states: NDArray[np.int32],
        forward_metrics: dict,
        horizon: int,
    ) -> dict:
        h_key = None
        for k in forward_metrics:
            if f"_{horizon}" in k:
                h_key = k
                break
        if h_key is None:
            raise ValueError(f"No forward metric found for horizon {horizon}")
        fwd = forward_metrics[h_key]
        valid = ~np.isnan(fwd)
        s_valid = states[valid]
        f_valid = fwd[valid]
        unique_states = np.unique(s_valid)
        if len(unique_states) < 2:
            return {
                "f_statistic": 0.0,
                "p_value": 1.0,
                "eta_squared": 0.0,
                "mutual_information": 0.0,
            }
        groups: list[NDArray[np.float64]] = []
        for s in unique_states:
            mask = s_valid == s
            if np.sum(mask) > 1:
                groups.append(f_valid[mask].astype(np.float64))
        if len(groups) < 2:
            return {
                "f_statistic": 0.0,
                "p_value": 1.0,
                "eta_squared": 0.0,
                "mutual_information": 0.0,
            }
        f_stat, p_val = f_oneway(*groups)
        eta_sq = float(_eta_squared_numba(f_valid.astype(np.float64), s_valid, unique_states))
        mi = float(mutual_info_score(s_valid, np.digitize(f_valid, np.percentile(f_valid[~np.isnan(f_valid)], np.linspace(0, 100, 21))[1:-1])))
        return {
            "f_statistic": float(f_stat),
            "p_value": float(p_val),
            "eta_squared": eta_sq,
            "mutual_information": mi,
        }
