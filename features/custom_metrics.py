from __future__ import annotations

import numba
import numpy as np
from numpy.typing import NDArray


@numba.jit(nopython=True, cache=True)
def price_position(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
) -> NDArray[np.float32]:
    result = np.zeros(len(close), dtype=np.float32)
    for i in range(len(close)):
        r = high[i] - low[i]
        if r > 0:
            result[i] = (close[i] - low[i]) / r
    return result


@numba.jit(nopython=True, cache=True)
def wvap(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    volume: NDArray[np.float64],
) -> NDArray[np.float32]:
    result = np.zeros(len(close), dtype=np.float32)
    cum_pv = 0.0
    cum_v = 0.0
    for i in range(len(close)):
        typical = (high[i] + low[i] + close[i]) / 3.0
        cum_pv += typical * volume[i]
        cum_v += volume[i]
        if cum_v > 0:
            result[i] = cum_pv / cum_v
    return result


@numba.jit(nopython=True, cache=True)
def cumulative_delta(
    buy_volume: NDArray[np.float64],
    sell_volume: NDArray[np.float64],
) -> NDArray[np.float32]:
    result = np.zeros(len(buy_volume), dtype=np.float32)
    cum = 0.0
    for i in range(len(buy_volume)):
        cum += buy_volume[i] - sell_volume[i]
        result[i] = cum
    return result


@numba.jit(nopython=True, cache=True)
def efficiency_ratio(
    close: NDArray[np.float64],
    window: int,
) -> NDArray[np.float32]:
    result = np.full(len(close), np.nan, dtype=np.float32)
    for i in range(window - 1, len(close)):
        net_change = abs(close[i] - close[i - window + 1])
        sum_changes = np.sum(np.abs(np.diff(close[i - window + 1 : i + 1])))
        if sum_changes > 0:
            result[i] = net_change / sum_changes
    return result


@numba.jit(nopython=True, cache=True)
def rolling_regression_slope(
    close: NDArray[np.float64],
    window: int,
) -> NDArray[np.float32]:
    result = np.full(len(close), np.nan, dtype=np.float32)
    x = np.arange(window, dtype=np.float64)
    x_mean = np.mean(x)
    x_ss = np.sum((x - x_mean) ** 2)
    for i in range(window - 1, len(close)):
        y = close[i - window + 1 : i + 1]
        y_mean = np.mean(y)
        slope = np.sum((x - x_mean) * (y - y_mean)) / x_ss
        result[i] = slope
    return result


@numba.jit(nopython=True, cache=True)
def rolling_regression_r2(
    close: NDArray[np.float64],
    window: int,
) -> NDArray[np.float32]:
    result = np.full(len(close), np.nan, dtype=np.float32)
    x = np.arange(window, dtype=np.float64)
    x_mean = np.mean(x)
    x_ss = np.sum((x - x_mean) ** 2)
    for i in range(window - 1, len(close)):
        y = close[i - window + 1 : i + 1]
        y_mean = np.mean(y)
        slope = np.sum((x - x_mean) * (y - y_mean)) / x_ss
        intercept = y_mean - slope * x_mean
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y_mean) ** 2)
        if ss_tot > 0:
            result[i] = 1.0 - ss_res / ss_tot
    return result


@numba.jit(nopython=True, cache=True)
def atr(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    window: int,
) -> NDArray[np.float32]:
    result = np.full(len(high), np.nan, dtype=np.float32)
    tr = np.zeros(len(high), dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, len(high)):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
    result[window - 1] = np.mean(tr[:window])
    for i in range(window, len(high)):
        result[i] = (result[i - 1] * (window - 1) + tr[i]) / window
    return result


@numba.jit(nopython=True, cache=True)
def super_smoother(
    value: NDArray[np.float64],
    cutoff: float = 10.0,
) -> NDArray[np.float32]:
    result = np.zeros(len(value), dtype=np.float32)
    a = np.exp(-1.414 * np.pi / cutoff)
    b = 2.0 * a * np.cos(1.414 * np.pi / cutoff)
    c2 = b
    c3 = -a * a
    c1 = 1.0 - c2 - c3
    result[0] = value[0]
    result[1] = value[1]
    for i in range(2, len(value)):
        result[i] = c1 * (value[i] + value[i - 1]) / 2.0 + c2 * result[i - 1] + c3 * result[i - 2]
    return result
