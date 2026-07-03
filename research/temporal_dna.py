from __future__ import annotations

import numpy as np
import numba
from numpy.typing import NDArray

from config.settings import settings


@numba.jit(nopython=True, cache=True)
def _rolling_std(x: NDArray[np.float64], window: int) -> NDArray[np.float32]:
    n = len(x)
    result = np.zeros(n, dtype=np.float32)
    for i in range(window - 1, n):
        seg = x[i - window + 1 : i + 1]
        m = np.mean(seg)
        v = 0.0
        for j in range(window):
            d = seg[j] - m
            v += d * d
        result[i] = np.sqrt(v / window)
    return result


@numba.jit(nopython=True, cache=True)
def _rolling_entropy(x: NDArray[np.float64], window: int, bins: int) -> NDArray[np.float32]:
    n = len(x)
    result = np.zeros(n, dtype=np.float32)
    eps = 1e-10
    for i in range(window - 1, n):
        seg = x[i - window + 1 : i + 1]
        lo = seg.min()
        hi = seg.max()
        if hi - lo < 1e-10:
            result[i] = 0.0
            continue
        bin_edges = np.linspace(lo, hi, bins + 1)
        counts = np.zeros(bins, dtype=np.int64)
        for j in range(window):
            for b in range(bins):
                if bin_edges[b] <= seg[j] < bin_edges[b + 1]:
                    counts[b] += 1
                    break
        ent = 0.0
        for b in range(bins):
            if counts[b] > 0:
                p = counts[b] / window
                ent -= p * np.log(p + eps)
        result[i] = ent / np.log(bins + 1)
    return result


@numba.jit(nopython=True, cache=True)
def _rolling_compression(high: NDArray[np.float64], low: NDArray[np.float64], window: int) -> NDArray[np.float32]:
    n = len(high)
    result = np.zeros(n, dtype=np.float32)
    for i in range(window - 1, n):
        ranges = high[i - window + 1 : i + 1] - low[i - window + 1 : i + 1]
        cur = ranges[-1]
        avg = np.mean(ranges)
        if avg > 1e-10:
            result[i] = 1.0 - cur / avg
    return result


@numba.jit(nopython=True, cache=True)
def _hurst_variance_ratio(returns: NDArray[np.float64], window: int, max_lag: int) -> NDArray[np.float32]:
    n = len(returns)
    result = np.full(n, 0.5, dtype=np.float32)
    lags = []
    l = 2
    while l <= max_lag and l < window:
        lags.append(l)
        l *= 2
    if len(lags) < 2:
        return result
    for i in range(window, n):
        seg = returns[i - window : i]
        vars_list = np.zeros(len(lags), dtype=np.float64)
        for li, lag in enumerate(lags):
            diff = seg[lag:] - seg[:-lag]
            vars_list[li] = np.var(diff)
        log_lags = np.zeros(len(lags), dtype=np.float64)
        log_vars = np.zeros(len(lags), dtype=np.float64)
        for li in range(len(lags)):
            log_lags[li] = np.log(lags[li])
            log_vars[li] = np.log(vars_list[li] + 1e-15)
        sum_l = 0.0
        sum_l2 = 0.0
        sum_lv = 0.0
        sum_v = 0.0
        k = len(lags)
        for li in range(k):
            sum_l += log_lags[li]
            sum_l2 += log_lags[li] * log_lags[li]
            sum_lv += log_lags[li] * log_vars[li]
            sum_v += log_vars[li]
        slope = (k * sum_lv - sum_l * sum_v) / (k * sum_l2 - sum_l * sum_l + 1e-15)
        h = slope / 2.0
        result[i] = max(0.0, min(1.0, h))
    return result


class TemporalDNAResearch:
    def __init__(self, sequence_length: int = settings.research.dna_sequence_length):
        self.sequence_length = sequence_length

    def compute_volatility(self, returns: NDArray[np.float64], window: int = 20) -> NDArray[np.float32]:
        return _rolling_std(returns.astype(np.float64), min(window, len(returns)))

    def compute_acceleration(self, returns: NDArray[np.float64], window: int = 10) -> NDArray[np.float32]:
        vol = self.compute_volatility(returns, window)
        acc = np.diff(vol, prepend=vol[0:1])
        return acc

    def compute_entropy(self, returns: NDArray[np.float64], window: int = 20, bins: int = 10) -> NDArray[np.float32]:
        return _rolling_entropy(returns.astype(np.float64), min(window, len(returns)), bins)

    def compute_compression(self, high: NDArray[np.float64], low: NDArray[np.float64], window: int = 20) -> NDArray[np.float32]:
        return _rolling_compression(high.astype(np.float64), low.astype(np.float64), min(window, len(high)))

    def compute_expansion(self, high: NDArray[np.float64], low: NDArray[np.float64], window: int = 20) -> NDArray[np.float32]:
        comp = self.compute_compression(high, low, window)
        return (1.0 - comp).astype(np.float32)

    def compute_persistence(self, returns: NDArray[np.float64], window: int = 20) -> NDArray[np.float32]:
        max_lag = min(16, window // 2)
        return _hurst_variance_ratio(returns.astype(np.float64), min(window, len(returns)), max_lag)

    def build_dna_vector(
        self,
        volatility: NDArray[np.float32],
        acceleration: NDArray[np.float32],
        entropy: NDArray[np.float32],
        compression: NDArray[np.float32],
        expansion: NDArray[np.float32],
        persistence: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        n = min(len(volatility), len(acceleration), len(entropy), len(compression), len(expansion), len(persistence))
        cols = 6
        result = np.zeros((n, cols), dtype=np.float32)
        result[:, 0] = volatility[:n]
        result[:, 1] = acceleration[:n]
        result[:, 2] = entropy[:n]
        result[:, 3] = compression[:n]
        result[:, 4] = expansion[:n]
        result[:, 5] = persistence[:n]
        return result

    def build_behavioral_signature(self, dna_matrix: NDArray[np.float32], window: int) -> NDArray[np.float32]:
        n = dna_matrix.shape[0]
        cols = dna_matrix.shape[1]
        result = np.zeros((n, cols), dtype=np.float32)
        for i in range(window - 1, n):
            result[i] = np.mean(dna_matrix[i - window + 1 : i + 1], axis=0)
        return result

    def compute_all(self, ohlc: dict) -> dict:
        returns = ohlc.get("returns", np.array([], dtype=np.float64))
        high = ohlc.get("high", np.array([], dtype=np.float64))
        low = ohlc.get("low", np.array([], dtype=np.float64))
        vol = self.compute_volatility(returns) if len(returns) > 0 else np.array([], dtype=np.float32)
        acc = self.compute_acceleration(returns) if len(returns) > 0 else np.array([], dtype=np.float32)
        ent = self.compute_entropy(returns) if len(returns) > 0 else np.array([], dtype=np.float32)
        comp = self.compute_compression(high, low) if len(high) > 0 and len(low) > 0 else np.array([], dtype=np.float32)
        exp = self.compute_expansion(high, low) if len(high) > 0 and len(low) > 0 else np.array([], dtype=np.float32)
        pers = self.compute_persistence(returns) if len(returns) > 0 else np.array([], dtype=np.float32)
        return {
            "volatility": vol,
            "acceleration": acc,
            "entropy": ent,
            "compression": comp,
            "expansion": exp,
            "persistence": pers,
        }
