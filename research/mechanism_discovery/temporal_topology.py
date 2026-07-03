from __future__ import annotations

from typing import Any, Optional

import numpy as np
import numba
from numpy.typing import NDArray

from research.mechanism_discovery.base import BaseMechanism, MechanismScore
from research.information_discovery.mi_estimator import (
    MIEstimator,
    _fast_percentile,
    _fast_digitize,
    _fast_entropy_digitized,
    _fast_joint_entropy_digitized,
)


@numba.jit(nopython=True, cache=True)
def _numba_rolling_mutual_info(
    feature: NDArray[np.float64],
    target: NDArray[np.float64],
    window: int,
    n_bins: int,
) -> NDArray[np.float64]:
    n = min(len(feature), len(target))
    mi_values = np.zeros(n)
    q = np.linspace(0.0, 1.0, n_bins + 1)

    for i in range(window, n):
        f_slice = feature[i - window : i]
        t_slice = target[i - window : i]

        n_valid = 0
        for k in range(len(f_slice)):
            if not np.isnan(f_slice[k]) and not np.isnan(t_slice[k]):
                n_valid += 1

        if n_valid < 2:
            mi_values[i] = 0.0
            continue

        f_clean = np.zeros(n_valid)
        t_clean = np.zeros(n_valid)
        idx = 0
        for k in range(len(f_slice)):
            if not np.isnan(f_slice[k]) and not np.isnan(t_slice[k]):
                f_clean[idx] = f_slice[k]
                t_clean[idx] = t_slice[k]
                idx += 1

        f_bins = _fast_percentile(f_clean, q)
        t_bins = _fast_percentile(t_clean, q)

        ux = np.unique(f_bins)
        uy = np.unique(t_bins)
        if len(ux) < 2 or len(uy) < 2:
            mi_values[i] = 0.0
            continue

        dig_x = _fast_digitize(f_clean, f_bins)
        dig_y = _fast_digitize(t_clean, t_bins)

        hx = _fast_entropy_digitized(dig_x, n_bins)
        hy = _fast_entropy_digitized(dig_y, n_bins)
        hxy = _fast_joint_entropy_digitized(dig_x, dig_y, n_bins)

        mi_values[i] = max(0.0, hx + hy - hxy)

    return mi_values


@numba.jit(nopython=True, cache=True)
def _numba_rolling_autocorr_lag1(returns: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    n = len(returns)
    rolling_ac = np.zeros(n)
    for i in range(window + 1, n):
        seg = returns[i - window : i]
        seg1 = seg[:-1]
        seg2 = seg[1:]
        s1 = np.std(seg1)
        s2 = np.std(seg2)
        if s1 > 1e-12 and s2 > 1e-12:
            mean1 = np.mean(seg1)
            mean2 = np.mean(seg2)
            cov = np.mean((seg1 - mean1) * (seg2 - mean2))
            rolling_ac[i] = cov / (s1 * s2)
    return rolling_ac


@numba.jit(nopython=True, cache=True)
def _numba_rolling_std(returns: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    n = len(returns)
    rolling_vol = np.zeros(n)
    for i in range(window, n):
        rolling_vol[i] = np.std(returns[i - window : i])
    return rolling_vol


@numba.jit(nopython=True, cache=True)
def _numba_rolling_sum(x: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    N = len(x)
    result = np.zeros(N, dtype=np.float64)
    cum = np.cumsum(x)
    result[window - 1] = cum[window - 1]
    result[window:] = cum[window:] - cum[:-window]
    return result


class TemporalTopology(BaseMechanism):
    WINDOW: int = 20
    PRICE_MOVE_THRESHOLD: float = 0.5
    VOLUME_SPIKE_THRESHOLD: float = 1.5
    EVENT_PRICE_STD: float = 1.0
    EVENT_VOLUME_STD: float = 2.0
    N_REGIME_BINS: int = 3

    def __init__(self) -> None:
        super().__init__(name="temporal_topology", category="mechanism_class_3")
        self._mi_estimator = MIEstimator()
        self._state_contribution: NDArray = np.array([], dtype=np.float64)

    def compute(self, data: dict[str, NDArray], states: Optional[NDArray] = None) -> dict[str, Any]:
        price = np.asarray(data.get("price", []), dtype=np.float64)
        returns = np.asarray(data.get("returns", []), dtype=np.float64)
        volume = np.asarray(data.get("volume", []), dtype=np.float64)
        high = np.asarray(data.get("high", []), dtype=np.float64)
        low = np.asarray(data.get("low", []), dtype=np.float64)

        N = len(price)
        if N < self.WINDOW + 1:
            return self._empty_result(N)

        if len(returns) < N:
            returns = np.diff(price, prepend=price[0])
        if len(volume) < N:
            volume = np.ones(N, dtype=np.float64)
        if len(high) < N:
            high = price.copy()
        if len(low) < N:
            low = price.copy()

        time_density = self._compute_time_density(price, returns, volume, N)
        event_density = self._compute_event_density(returns, volume, N)
        information_density = self._compute_information_density(returns, volume, N)
        behavior_density = self._compute_behavior_density(returns, N)

        combined_density = (time_density + event_density + information_density + behavior_density) / 4.0
        combined_density = np.nan_to_num(combined_density)

        adaptive_time = np.cumsum(combined_density)
        adaptive_time = adaptive_time / max(1e-12, adaptive_time[-1])

        clock_time = np.arange(N, dtype=np.float64)
        clock_time_norm = clock_time / max(1.0, N - 1.0)

        warp_factor = float(adaptive_time[-1] / max(1e-12, clock_time_norm[-1]))

        clock_vs_event_variance = self._compare_time_variances(returns, clock_time_norm, adaptive_time, N)

        time_regime = self._compute_time_regime(combined_density)

        self._state_contribution = time_density / max(1e-12, np.max(time_density))

        result = {
            "clock_time": clock_time,
            "event_time": adaptive_time.copy(),
            "time_density": time_density,
            "event_density": event_density,
            "information_density": information_density,
            "behavior_density": behavior_density,
            "adaptive_time_coordinate": adaptive_time.copy(),
            "time_warp_factor": warp_factor,
            "clock_vs_event_variance": clock_vs_event_variance,
            "time_regime": time_regime,
        }
        self._state.update(result)
        return result

    def get_state_contribution(self) -> NDArray:
        return self._state_contribution

    def _compute_time_density(self, price: NDArray, returns: NDArray, volume: NDArray, N: int) -> NDArray:
        price_move = np.abs(np.diff(price, prepend=price[0]))
        ret_std = float(np.nanstd(returns))
        threshold = self.PRICE_MOVE_THRESHOLD * ret_std if ret_std > 1e-12 else 0.0
        price_exceed = (price_move > threshold).astype(np.float64)

        vol_mean = float(np.nanmean(volume))
        vol_std = float(np.nanstd(volume))
        vol_spike = np.zeros(N, dtype=np.float64)
        if vol_std > 1e-12:
            vol_spike = (volume > vol_mean + self.VOLUME_SPIKE_THRESHOLD * vol_std).astype(np.float64)

        density = _numba_rolling_sum(price_exceed + vol_spike, self.WINDOW) / float(self.WINDOW)
        return np.nan_to_num(density)

    def _compute_event_density(self, returns: NDArray, volume: NDArray, N: int) -> NDArray:
        ret_std = float(np.nanstd(returns))
        vol_std = float(np.nanstd(volume))

        event_mask = np.zeros(N, dtype=np.float64)
        if ret_std > 1e-12:
            ret_event = (np.abs(returns) > self.EVENT_PRICE_STD * ret_std).astype(np.float64)
            event_mask = ret_event
        if vol_std > 1e-12:
            vol_event = (volume > float(np.nanmean(volume)) + self.EVENT_VOLUME_STD * vol_std).astype(np.float64)
            event_mask = np.maximum(event_mask, vol_event)

        density = _numba_rolling_sum(event_mask, self.WINDOW) / float(self.WINDOW)
        density = np.nan_to_num(density)
        dmax = float(np.max(density))
        if dmax > 1e-12:
            density = density / dmax
        return density

    def _compute_information_density(self, returns: NDArray, volume: NDArray, N: int) -> NDArray:
        density = _numba_rolling_mutual_info(
            returns.astype(np.float64),
            volume.astype(np.float64),
            self.WINDOW,
            self._mi_estimator.n_bins,
        )

        dmax = float(np.max(density))
        if dmax > 1e-12:
            density = density / dmax
        return density

    def _compute_behavior_density(self, returns: NDArray, N: int) -> NDArray:
        returns_double = returns.astype(np.float64)
        rolling_vol = _numba_rolling_std(returns_double, self.WINDOW)
        rolling_vol[:self.WINDOW] = rolling_vol[self.WINDOW] if self.WINDOW < N else 0.0
        vol_rate = np.abs(np.diff(rolling_vol, prepend=rolling_vol[0]))
        vmax = float(np.max(vol_rate))
        if vmax > 1e-12:
            vol_rate = vol_rate / vmax

        rolling_ac = _numba_rolling_autocorr_lag1(returns_double, self.WINDOW)
        ac_rate = np.abs(np.diff(rolling_ac, prepend=rolling_ac[0]))
        amax = float(np.max(ac_rate))
        if amax > 1e-12:
            ac_rate = ac_rate / amax

        density = (vol_rate + ac_rate) / 2.0
        dmax = float(np.max(density))
        if dmax > 1e-12:
            density = density / dmax
        return np.nan_to_num(density)

    def _compare_time_variances(self, returns: NDArray, clock_time: NDArray, event_time: NDArray, N: int) -> float:
        # Use contemporaneous returns to avoid lookahead bias
        n_bins = max(2, N // self.WINDOW)
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

        clock_bins = np.digitize(clock_time, bin_edges) - 1
        event_bins = np.digitize(event_time, bin_edges) - 1

        clock_vars = []
        event_vars = []
        for b in range(n_bins):
            c_mask = clock_bins == b
            e_mask = event_bins == b
            c_vals = returns[c_mask]
            e_vals = returns[e_mask]
            if np.sum(c_mask) > 1:
                clock_vars.append(float(np.nanvar(c_vals)))
            if np.sum(e_mask) > 1:
                event_vars.append(float(np.nanvar(e_vals)))

        clock_var = float(np.mean(clock_vars)) if clock_vars else 0.0
        event_var = float(np.mean(event_vars)) if event_vars else 0.0

        if clock_var > 1e-12:
            return (clock_var - event_var) / clock_var
        return 0.0

    def _compute_time_regime(self, density: NDArray) -> NDArray:
        p33 = float(np.percentile(density, 33))
        p66 = float(np.percentile(density, 66))
        regime = np.zeros(len(density), dtype=np.int64)
        regime[density > p66] = 2
        regime[(density > p33) & (density <= p66)] = 1
        return regime

    @staticmethod
    def _rolling_sum(x: NDArray, window: int) -> NDArray:
        return _numba_rolling_sum(x.astype(np.float64), window)

    def _empty_result(self, N: int) -> dict[str, Any]:
        self._state_contribution = np.zeros(max(1, N), dtype=np.float64)
        return {
            "clock_time": np.arange(max(1, N), dtype=np.float64),
            "event_time": np.zeros(max(1, N), dtype=np.float64),
            "time_density": np.zeros(max(1, N), dtype=np.float64),
            "event_density": np.zeros(max(1, N), dtype=np.float64),
            "information_density": np.zeros(max(1, N), dtype=np.float64),
            "behavior_density": np.zeros(max(1, N), dtype=np.float64),
            "adaptive_time_coordinate": np.zeros(max(1, N), dtype=np.float64),
            "time_warp_factor": 1.0,
            "clock_vs_event_variance": 0.0,
            "time_regime": np.ones(max(1, N), dtype=np.int64),
        }
