from __future__ import annotations

from typing import Optional

import numpy as np
import numba
from numpy.typing import NDArray


@numba.jit(nopython=True, cache=True)
def _acc_dist_inner(price: NDArray[np.float64], volume: NDArray[np.float64], window: int) -> NDArray[np.int32]:
    n = len(price)
    result = np.zeros(n, dtype=np.int32)
    for i in range(window, n):
        price_seg = price[i - window:i]
        vol_seg = volume[i - window:i]
        p_slope = (price_seg[-1] - price_seg[0]) / price_seg[0]
        v_slope = (vol_seg[-1] - vol_seg[0]) / (vol_seg[0] + 1e-15)
        p_flat = abs(p_slope) < 0.02
        v_rising = v_slope > 0.05
        v_falling = v_slope < -0.05
        v_strong_rising = v_slope > 0.15
        v_strong_falling = v_slope < -0.15
        if p_flat and v_strong_rising:
            result[i] = 2
        elif p_flat and v_rising:
            result[i] = 1
        elif p_flat and v_strong_falling:
            result[i] = -2
        elif p_flat and v_falling:
            result[i] = -1
        else:
            result[i] = 0
    return result


@numba.jit(nopython=True, cache=True)
def _micro_regime_inner(returns: NDArray[np.float64], tick_vol: NDArray[np.float64], window: int) -> NDArray[np.int32]:
    n = len(returns)
    result = np.zeros(n, dtype=np.int32)
    for i in range(window, n):
        ret_seg = returns[i - window:i]
        vol_seg = tick_vol[i - window:i]
        r_mean = np.mean(ret_seg)
        r_std = np.std(ret_seg) + 1e-15
        autocorr = 0.0
        for j in range(1, len(ret_seg)):
            autocorr += (ret_seg[j - 1] - r_mean) * (ret_seg[j] - r_mean)
        autocorr /= (r_std * r_std * len(ret_seg))
        avg_vol = np.mean(vol_seg)
        high_vol = avg_vol > np.percentile(tick_vol[max(0, n - 1000):], 75) if n > 1000 else avg_vol > np.mean(tick_vol)
        if abs(autocorr) < 0.2 and not high_vol:
            result[i] = 0
        elif autocorr > 0.3:
            result[i] = 1
        elif autocorr < -0.3:
            result[i] = 2
        else:
            result[i] = 3
    return result


@numba.jit(nopython=True, cache=True)
def _info_efficiency_inner(price: NDArray[np.float64], returns: NDArray[np.float64], window: int) -> NDArray[np.int32]:
    n = len(price)
    result = np.zeros(n, dtype=np.int32)
    for i in range(window, n):
        ret_seg = returns[i - window:i]
        short_lag = max(1, window // 10)
        long_lag = max(2, window // 2)
        if len(ret_seg) <= long_lag:
            continue
        var_short = np.var(ret_seg[short_lag:] - ret_seg[:-short_lag])
        var_long = np.var(ret_seg[long_lag:] - ret_seg[:-long_lag])
        vr = var_long / (var_short * (long_lag / short_lag) + 1e-15)
        if 0.8 <= vr <= 1.2:
            result[i] = 0
        elif vr < 0.8:
            result[i] = 1
        else:
            result[i] = 2
    return result


@numba.jit(nopython=True, cache=True)
def _vol_sync_inner(price: NDArray[np.float64], volume: NDArray[np.float64], window: int) -> NDArray[np.int32]:
    n = len(price)
    result = np.zeros(n, dtype=np.int32)
    for i in range(window, n):
        price_seg = price[i - window:i]
        vol_seg = volume[i - window:i]
        p_ret = (price_seg[-1] - price_seg[0]) / price_seg[0]
        vol_corr = 0.0
        p_mean = np.mean(price_seg)
        v_mean = np.mean(vol_seg)
        p_std = np.std(price_seg) + 1e-15
        v_std = np.std(vol_seg) + 1e-15
        for j in range(window):
            vol_corr += (price_seg[j] - p_mean) * (vol_seg[j] - v_mean)
        vol_corr /= (p_std * v_std * window)
        if vol_corr > 0.3 and p_ret > 0:
            result[i] = 1
        elif vol_corr > 0.3 and p_ret < 0:
            result[i] = 2
        else:
            result[i] = 0
    return result


@numba.jit(nopython=True, cache=True)
def _cross_asset_inner(primary: NDArray[np.float64], secondary: NDArray[np.float64], window: int) -> NDArray[np.int32]:
    n = min(len(primary), len(secondary))
    result = np.zeros(n, dtype=np.int32)
    for i in range(window, n):
        p_seg = primary[i - window:i]
        s_seg = secondary[i - window:i]
        p_mean = np.mean(p_seg)
        s_mean = np.mean(s_seg)
        p_std = np.std(p_seg) + 1e-15
        s_std = np.std(s_seg) + 1e-15
        corr = 0.0
        for j in range(window):
            corr += (p_seg[j] - p_mean) * (s_seg[j] - s_mean)
        corr /= (p_std * s_std * window)
        p_trend = p_seg[-1] - p_seg[0]
        s_trend = s_seg[-1] - s_seg[0]
        if abs(corr) < 0.2:
            result[i] = 0
        elif corr > 0.5 and p_trend * s_trend > 0:
            result[i] = 1
        elif corr > 0.3 and p_trend * s_trend < 0:
            result[i] = 3
        elif corr > 0.3:
            result[i] = 2
        else:
            result[i] = 0
    return result


@numba.jit(nopython=True, cache=True)
def _normalize_minmax(arr: NDArray[np.float64]) -> NDArray[np.float32]:
    n = len(arr)
    result = np.zeros(n, dtype=np.float32)
    if n == 0:
        return result
    lo = np.min(arr)
    hi = np.max(arr)
    if hi - lo < 1e-15:
        return result
    for i in range(n):
        result[i] = (arr[i] - lo) / (hi - lo)
    return result


class NovelStateGenerator:

    def __init__(self) -> None:
        pass

    def compute_accumulation_distribution_state(
        self, price: NDArray[np.float64], volume: NDArray[np.float64], window: int = 100
    ) -> NDArray[np.int32]:
        return _acc_dist_inner(price, volume, window)

    def compute_microstructure_regime(
        self, returns: NDArray[np.float64], tick_volatility: NDArray[np.float64], window: int = 50
    ) -> NDArray[np.int32]:
        return _micro_regime_inner(returns, tick_volatility, window)

    def compute_information_efficiency_state(
        self, price: NDArray[np.float64], returns: NDArray[np.float64], window: int = 200
    ) -> NDArray[np.int32]:
        return _info_efficiency_inner(price, returns, window)

    def compute_volume_synchronization_state(
        self, price: NDArray[np.float64], volume: NDArray[np.float64], window: int = 50
    ) -> NDArray[np.int32]:
        return _vol_sync_inner(price, volume, window)

    def compute_cross_asset_state(
        self, primary_returns: NDArray[np.float64], secondary_returns: NDArray[np.float64], window: int = 50
    ) -> NDArray[np.int32]:
        return _cross_asset_inner(primary_returns, secondary_returns, window)

    def compute_novel_state_vector(
        self,
        price: NDArray[np.float64],
        high: NDArray[np.float64],
        low: NDArray[np.float64],
        returns: NDArray[np.float64],
        volume: NDArray[np.float64],
        tick_vol: Optional[NDArray[np.float64]] = None,
    ) -> NDArray[np.float32]:
        n = len(price)
        acc_dist = self.compute_accumulation_distribution_state(price, volume).astype(np.float64)
        if tick_vol is not None:
            micro = self.compute_microstructure_regime(returns, tick_vol).astype(np.float64)
        else:
            tick_vol_arr = np.abs(returns) * np.std(returns) + 1e-15
            micro = self.compute_microstructure_regime(returns, tick_vol_arr).astype(np.float64)
        info_eff = self.compute_information_efficiency_state(price, returns).astype(np.float64)
        vol_sync = self.compute_volume_synchronization_state(price, volume).astype(np.float64)
        high_low_range = high - low
        features = np.zeros((n, 5), dtype=np.float64)
        features[:, 0] = acc_dist
        features[:, 1] = micro
        features[:, 2] = info_eff
        features[:, 3] = vol_sync
        features[:, 4] = high_low_range
        result = np.zeros((n, 5), dtype=np.float32)
        for col in range(5):
            normalized = _normalize_minmax(features[:, col])
            for row in range(n):
                result[row, col] = normalized[row]
        return result
