from __future__ import annotations

import numba
import numpy as np
from numpy.typing import NDArray


@numba.jit(nopython=True, cache=True)
def rolling_zscore(arr: NDArray[np.float64], window: int) -> NDArray[np.float32]:
    result = np.full(len(arr), np.nan, dtype=np.float32)
    for i in range(window - 1, len(arr)):
        chunk = arr[i - window + 1 : i + 1]
        mean = np.mean(chunk)
        std = np.std(chunk)
        if std > 0:
            result[i] = (arr[i] - mean) / std
    return result


@numba.jit(nopython=True, cache=True)
def rolling_skew(arr: NDArray[np.float64], window: int) -> NDArray[np.float32]:
    result = np.full(len(arr), np.nan, dtype=np.float32)
    for i in range(window - 1, len(arr)):
        chunk = arr[i - window + 1 : i + 1]
        mean = np.mean(chunk)
        std = np.std(chunk)
        if std > 0:
            n = len(chunk)
            skew = np.sum((chunk - mean) ** 3) / n / (std ** 3)
            result[i] = skew
    return result


@numba.jit(nopython=True, cache=True)
def rolling_kurtosis(arr: NDArray[np.float64], window: int) -> NDArray[np.float32]:
    result = np.full(len(arr), np.nan, dtype=np.float32)
    for i in range(window - 1, len(arr)):
        chunk = arr[i - window + 1 : i + 1]
        mean = np.mean(chunk)
        std = np.std(chunk)
        if std > 0:
            n = len(chunk)
            kurt = np.sum((chunk - mean) ** 4) / n / (std ** 4) - 3.0
            result[i] = kurt
    return result


@numba.jit(nopython=True, cache=True)
def rolling_quantile(arr: NDArray[np.float64], window: int, q: float) -> NDArray[np.float32]:
    result = np.full(len(arr), np.nan, dtype=np.float32)
    for i in range(window - 1, len(arr)):
        chunk = np.sort(arr[i - window + 1 : i + 1])
        idx = int(q * (len(chunk) - 1))
        result[i] = chunk[idx]
    return result


@numba.jit(nopython=True, cache=True)
def rolling_rank(arr: NDArray[np.float64], window: int) -> NDArray[np.float32]:
    result = np.full(len(arr), np.nan, dtype=np.float32)
    for i in range(window - 1, len(arr)):
        chunk = arr[i - window + 1 : i + 1]
        rank = np.sum(chunk <= arr[i]) - 1
        result[i] = rank / (len(chunk) - 1)
    return result


@numba.jit(nopython=True, cache=True)
def rolling_autocorr(arr: NDArray[np.float64], window: int, lag: int = 1) -> NDArray[np.float32]:
    result = np.full(len(arr), np.nan, dtype=np.float32)
    for i in range(window - 1 + lag, len(arr)):
        a = arr[i - window + 1 : i + 1 - lag]
        b = arr[i - window + 1 + lag : i + 1]
        ma = np.mean(a)
        mb = np.mean(b)
        num = np.sum((a - ma) * (b - mb))
        den = np.sqrt(np.sum((a - ma) ** 2) * np.sum((b - mb) ** 2))
        if den > 0:
            result[i] = num / den
    return result


@numba.jit(nopython=True, cache=True)
def rolling_entropy(arr: NDArray[np.float64], window: int, bins: int = 10) -> NDArray[np.float32]:
    result = np.full(len(arr), np.nan, dtype=np.float32)
    for i in range(window - 1, len(arr)):
        chunk = arr[i - window + 1 : i + 1]
        min_val = np.min(chunk)
        max_val = np.max(chunk)
        if max_val == min_val:
            result[i] = 0.0
            continue
        bin_edges = np.linspace(min_val, max_val, bins + 1)
        hist = np.zeros(bins)
        for v in chunk:
            idx = int((v - min_val) / (max_val - min_val) * bins)
            if idx >= bins:
                idx = bins - 1
            hist[idx] += 1
        probs = hist / len(chunk)
        entropy = 0.0
        for p in probs:
            if p > 0:
                entropy -= p * np.log2(p)
        result[i] = entropy
    return result


@numba.jit(nopython=True, cache=True)
def rolling_corr(a: NDArray[np.float64], b: NDArray[np.float64], window: int) -> NDArray[np.float32]:
    result = np.full(len(a), np.nan, dtype=np.float32)
    for i in range(window - 1, len(a)):
        ca = a[i - window + 1 : i + 1]
        cb = b[i - window + 1 : i + 1]
        ma = np.mean(ca)
        mb = np.mean(cb)
        num = np.sum((ca - ma) * (cb - mb))
        den = np.sqrt(np.sum((ca - ma) ** 2) * np.sum((cb - mb) ** 2))
        if den > 0:
            result[i] = num / den
    return result


@numba.jit(nopython=True, cache=True)
def rolling_hurst(returns: NDArray[np.float64], window: int) -> NDArray[np.float32]:
    result = np.full(len(returns), np.nan, dtype=np.float32)
    lags = [2, 4, 8, 16]
    for i in range(window - 1, len(returns)):
        chunk = returns[i - window + 1 : i + 1]
        tau = []
        used_lags = []
        for lag in lags:
            if lag >= len(chunk):
                continue
            diff = chunk[lag:] - chunk[:-lag]
            var = np.var(diff)
            if var > 0:
                tau.append(var)
                used_lags.append(lag)
        if len(tau) >= 3:
            n_pts = len(tau)
            log_lags = np.zeros(n_pts, dtype=np.float64)
            log_tau = np.zeros(n_pts, dtype=np.float64)
            for j in range(n_pts):
                log_lags[j] = np.log(used_lags[j])
                log_tau[j] = np.log(tau[j])
            xm = np.mean(log_lags)
            ym = np.mean(log_tau)
            num = 0.0
            den = 0.0
            for j in range(n_pts):
                dx = log_lags[j] - xm
                num += dx * (log_tau[j] - ym)
                den += dx * dx
            if den > 0:
                result[i] = (num / den) / 2.0
    return result
