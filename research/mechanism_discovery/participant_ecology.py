from __future__ import annotations

from typing import Any, Optional

import numpy as np
import numba
from numpy.typing import NDArray

from research.mechanism_discovery.base import BaseMechanism, MechanismScore


@numba.jit(nopython=True, cache=True)
def _numba_rolling_mean(x: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    T = len(x)
    result = np.full(T, np.nan, dtype=np.float64)
    if T < window:
        return result
    cum = np.cumsum(x)
    result[window - 1] = cum[window - 1]
    for t in range(window, T):
        result[t] = cum[t] - cum[t - window]
    result[window - 1:] /= float(window)
    return result


@numba.jit(nopython=True, cache=True)
def _numba_rolling_std(x: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    T = len(x)
    result = np.full(T, np.nan, dtype=np.float64)
    if T < window:
        return result
    for t in range(window - 1, T):
        result[t] = np.std(x[t - window + 1 : t + 1])
    return result


@numba.jit(nopython=True, cache=True)
def _numba_rolling_autocorr(x: NDArray[np.float64], lag: int, window: int) -> NDArray[np.float64]:
    T = len(x)
    result = np.full(T, np.nan, dtype=np.float64)
    if T <= window or window <= lag:
        return result
    n_pairs = window - lag
    if n_pairs < 3:
        return result
    for t in range(window, T + 1):
        seg1 = x[t - window : t - lag]
        seg2 = x[t - window + lag : t]
        s1 = np.std(seg1)
        s2 = np.std(seg2)
        if s1 < 1e-12 or s2 < 1e-12:
            result[t - 1] = 0.0
        else:
            mean1 = np.mean(seg1)
            mean2 = np.mean(seg2)
            cov = np.mean((seg1 - mean1) * (seg2 - mean2))
            result[t - 1] = cov / (s1 * s2)
    return result


@numba.jit(nopython=True, cache=True)
def _numba_rolling_corr(x: NDArray[np.float64], y: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    T = min(len(x), len(y))
    result = np.full(T, np.nan, dtype=np.float64)
    if T < window:
        return result
    for t in range(window - 1, T):
        sx = x[t - window + 1 : t + 1]
        sy = y[t - window + 1 : t + 1]
        s1 = np.std(sx)
        s2 = np.std(sy)
        if s1 < 1e-12 or s2 < 1e-12:
            result[t] = 0.0
        else:
            mean1 = np.mean(sx)
            mean2 = np.mean(sy)
            cov = np.mean((sx - mean1) * (sy - mean2))
            result[t] = cov / (s1 * s2)
    return result


@numba.jit(nopython=True, cache=True)
def _numba_linregress_slope(x_vals: NDArray[np.float64], y_vals: NDArray[np.float64]) -> float:
    mean_x = np.mean(x_vals)
    mean_y = np.mean(y_vals)
    var_x = np.var(x_vals)
    if var_x < 1e-12:
        return 0.0
    cov_xy = np.mean((x_vals - mean_x) * (y_vals - mean_y))
    return cov_xy / var_x


@numba.jit(nopython=True, cache=True)
def _numba_medium_activity_slopes(price: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    T = len(price)
    slope_signal = np.full(T, np.nan, dtype=np.float64)
    x_vals = np.arange(window, dtype=np.float64)
    for t in range(window - 1, T):
        seg = price[t - window + 1 : t + 1]
        s = np.std(seg)
        if s > 1e-12:
            slope_signal[t] = _numba_linregress_slope(x_vals, seg)
        else:
            slope_signal[t] = 0.0
    return slope_signal


@numba.jit(nopython=True, cache=True)
def _numba_rolling_vwap(price: NDArray[np.float64], volume: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    T = len(price)
    result = np.full(T, np.nan, dtype=np.float64)
    if T < window:
        return result
    pv = price * volume
    for t in range(window - 1, T):
        vol_sum = float(np.sum(volume[t - window + 1 : t + 1]))
        if vol_sum > 1e-12:
            result[t] = float(np.sum(pv[t - window + 1 : t + 1])) / vol_sum
        else:
            result[t] = float(np.mean(price[t - window + 1 : t + 1]))
    return result


@numba.jit(nopython=True, cache=True)
def _numba_compute_alignment(activity_matrix: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    T = activity_matrix.shape[0]
    result = np.full(T, np.nan, dtype=np.float64)
    if T < window:
        return np.zeros(T, dtype=np.float64)
    for t in range(window - 1, T):
        cors = np.zeros(3)
        idx = 0
        for i in range(3):
            for j in range(i + 1, 3):
                xi = activity_matrix[t - window + 1 : t + 1, i]
                xj = activity_matrix[t - window + 1 : t + 1, j]
                si = np.std(xi)
                sj = np.std(xj)
                if si < 1e-12 or sj < 1e-12:
                    cors[idx] = 0.0
                else:
                    mean_i = np.mean(xi)
                    mean_j = np.mean(xj)
                    cov = np.mean((xi - mean_i) * (xj - mean_j))
                    cors[idx] = cov / (si * sj)
                idx += 1
        result[t] = np.mean(cors)
    
    val_sum = 0.0
    val_cnt = 0
    for t in range(T):
        if not np.isnan(result[t]):
            val_sum += result[t]
            val_cnt += 1
    fill = val_sum / val_cnt if val_cnt > 0 else 0.0
    for t in range(window - 1):
        result[t] = fill
        
    return result


class ParticipantEcology(BaseMechanism):
    FAST_WINDOW: int = 15
    MEDIUM_WINDOW: int = 35
    SLOW_WINDOW: int = 100
    AUTO_LAGS: tuple[int, ...] = (1, 2, 3, 4, 5)
    ROLLING_ALIGNMENT_WINDOW: int = 30

    def __init__(self) -> None:
        super().__init__(name="participant_ecology", category="market_microstructure")

    def compute(self, data: dict[str, NDArray], states: Optional[NDArray] = None) -> dict[str, Any]:
        price = np.asarray(data.get("price", []), dtype=np.float64)
        returns = np.asarray(data.get("returns", []), dtype=np.float64)
        volume = np.asarray(data.get("volume", []), dtype=np.float64)
        high = np.asarray(data.get("high", []), dtype=np.float64)
        low = np.asarray(data.get("low", []), dtype=np.float64)

        T = len(price)
        if T < max(self.FAST_WINDOW, 2):
            return self._empty_result(T)

        if len(returns) < T:
            returns = np.diff(price, prepend=price[0])
        if len(volume) < T:
            volume = np.ones(T, dtype=np.float64)
        if len(high) < T:
            high = price.copy()
        if len(low) < T:
            low = price.copy()

        fast_activity = self._fast_activity(returns, volume, T)
        medium_activity = self._medium_activity(price, returns, volume, T)
        slow_activity = self._slow_activity(price, returns, volume, high, low, T)

        activity_matrix = np.column_stack([fast_activity, medium_activity, slow_activity])

        cohort_dominance = self._compute_dominance(activity_matrix)
        cohort_alignment = self._compute_alignment(activity_matrix)
        cohort_alignment_mean = float(np.nanmean(np.abs(cohort_alignment)))

        labels = ["fast", "medium", "slow"]
        dominant_idx = int(np.argmax(list(cohort_dominance.values())))
        dominant_cohort = labels[dominant_idx]
        cohort_regime = dominant_idx

        cohort_rotation = self._compute_rotation(activity_matrix)
        cohort_stress = self._compute_stress(activity_matrix)
        cohort_conflict = self._compute_conflict(cohort_alignment)

        result = {
            "fast_cohort_activity": fast_activity.tolist(),
            "medium_cohort_activity": medium_activity.tolist(),
            "slow_cohort_activity": slow_activity.tolist(),
            "cohort_dominance": cohort_dominance,
            "cohort_conflict": cohort_conflict,
            "cohort_rotation": cohort_rotation,
            "cohort_stress": cohort_stress,
            "cohort_alignment": cohort_alignment,
            "cohort_alignment_mean": cohort_alignment_mean,
            "dominant_cohort": dominant_cohort,
            "cohort_regime": cohort_regime,
        }
        self._state.update(result)
        return result

    def get_state_contribution(self) -> NDArray:
        return self._state.get("cohort_alignment", np.array([], dtype=np.float64))

    @staticmethod
    def _rolling_mean(x: NDArray, window: int) -> NDArray:
        return _numba_rolling_mean(x.astype(np.float64), window)

    @staticmethod
    def _rolling_std(x: NDArray, window: int) -> NDArray:
        return _numba_rolling_std(x.astype(np.float64), window)

    @staticmethod
    def _rolling_autocorr(x: NDArray, lag: int, window: int) -> NDArray:
        return _numba_rolling_autocorr(x.astype(np.float64), lag, window)

    @staticmethod
    def _rolling_corr(x: NDArray, y: NDArray, window: int) -> NDArray:
        return _numba_rolling_corr(x.astype(np.float64), y.astype(np.float64), window)

    @staticmethod
    def _zscore(x: NDArray) -> NDArray:
        mean = np.nanmean(x)
        std = np.nanstd(x)
        if std < 1e-12:
            return np.zeros_like(x)
        return (x - mean) / std

    @staticmethod
    def _normalize(x: NDArray) -> NDArray:
        xmin = np.nanmin(x)
        xmax = np.nanmax(x)
        if xmax - xmin < 1e-12:
            return np.zeros_like(x)
        return (x - xmin) / (xmax - xmin)

    def _fast_activity(self, returns: NDArray, volume: NDArray, T: int) -> NDArray:
        window = self.FAST_WINDOW
        signals = []

        auto_features = []
        for lag in self.AUTO_LAGS:
            ac = self._rolling_autocorr(returns, lag, window)
            if len(ac) > 0:
                auto_features.append(np.abs(ac))
        if auto_features:
            stacked = np.column_stack(auto_features)
            valid_mask = ~np.all(np.isnan(stacked), axis=1)
            af_mean = np.zeros(stacked.shape[0], dtype=np.float64)
            if np.any(valid_mask):
                af_mean[valid_mask] = np.nanmean(stacked[valid_mask], axis=1)
            signals.append(np.nan_to_num(af_mean))
        else:
            signals.append(np.zeros(len(returns), dtype=np.float64))

        vc = self._rolling_std(returns, min(5, window))
        signals.append(vc)

        ofi = volume * np.sign(returns)
        ofi_smooth = self._rolling_mean(ofi, window)
        signals.append(np.abs(ofi_smooth))

        norm_signals = []
        for sig in signals:
            norm_signals.append(np.nan_to_num(self._zscore(sig)))

        combined = np.nanmean(np.column_stack(norm_signals), axis=1)
        return np.nan_to_num(self._normalize(combined))

    def _medium_activity(self, price: NDArray, returns: NDArray, volume: NDArray, T: int) -> NDArray:
        window = self.MEDIUM_WINDOW
        signals = []

        slope_signal = _numba_medium_activity_slopes(price.astype(np.float64), window)
        signals.append(slope_signal)

        vp_corr = self._rolling_corr(volume, returns, window)
        signals.append(vp_corr)

        momentum = self._rolling_mean(returns, window)
        signals.append(momentum)

        norm_signals = []
        for sig in signals:
            norm_signals.append(np.nan_to_num(self._zscore(sig)))

        combined = np.nanmean(np.column_stack(norm_signals), axis=1)
        return np.nan_to_num(self._normalize(combined))

    def _slow_activity(
        self,
        price: NDArray,
        returns: NDArray,
        volume: NDArray,
        high: NDArray,
        low: NDArray,
        T: int,
    ) -> NDArray:
        window = self.SLOW_WINDOW
        signals = []

        long_ma = self._rolling_mean(price, window)
        position_accum = price - long_ma
        signals.append(position_accum)

        vol_z = self._zscore(volume)
        atr = self._rolling_mean(high - low, max(10, window // 10))
        atr_z = self._zscore(atr)
        large_trade = np.zeros(T, dtype=np.float64)
        for t in range(T):
            if not (np.isnan(vol_z[t]) or np.isnan(atr_z[t])):
                if vol_z[t] > 1.0 and atr_z[t] < 0.0:
                    large_trade[t] = 1.0
        signals.append(large_trade)

        vwap = self._rolling_vwap(price, volume, window)
        vwap_dev = price - vwap
        cum_vwap_dev = np.nancumsum(np.nan_to_num(vwap_dev))
        denom = np.arange(T, dtype=np.float64) + 1.0
        signals.append(cum_vwap_dev / denom)

        norm_signals = []
        for sig in signals:
            norm_signals.append(np.nan_to_num(self._zscore(sig)))

        combined = np.nanmean(np.column_stack(norm_signals), axis=1)
        return np.nan_to_num(self._normalize(combined))

    @staticmethod
    def _rolling_vwap(price: NDArray, volume: NDArray, window: int) -> NDArray:
        return _numba_rolling_vwap(price.astype(np.float64), volume.astype(np.float64), window)

    @staticmethod
    def _compute_dominance(activity_matrix: NDArray) -> dict[str, float]:
        variances = []
        for i in range(3):
            col = activity_matrix[:, i]
            valid = col[~np.isnan(col)]
            if len(valid) < 2:
                variances.append(0.0)
            else:
                variances.append(float(np.var(valid)))
        total = sum(variances)
        if total < 1e-12:
            return {"fast": 1.0 / 3.0, "medium": 1.0 / 3.0, "slow": 1.0 / 3.0}
        return {
            "fast": variances[0] / total,
            "medium": variances[1] / total,
            "slow": variances[2] / total,
        }

    def _compute_alignment(self, activity_matrix: NDArray) -> NDArray:
        T = activity_matrix.shape[0]
        window = min(self.ROLLING_ALIGNMENT_WINDOW, max(10, T // 4))
        return _numba_compute_alignment(activity_matrix.astype(np.float64), window)

    @staticmethod
    def _compute_rotation(activity_matrix: NDArray) -> float:
        T = activity_matrix.shape[0]
        if T < 3:
            return 0.0
        dominant = np.nanargmax(activity_matrix, axis=1)
        changes = float(np.sum(np.diff(dominant) != 0))
        return changes / float(T - 1)

    def _compute_stress(self, activity_matrix: NDArray) -> float:
        T = activity_matrix.shape[0]
        baseline_window = min(50, max(2, T // 2))
        if baseline_window < 2:
            return 0.0
        stresses = []
        for i in range(3):
            col = activity_matrix[:, i]
            baseline_col = self._rolling_mean(col, baseline_window)
            dev = np.abs(col - baseline_col)
            stresses.append(float(np.nanmean(dev)))
        return float(np.nanmean(stresses))

    @staticmethod
    def _compute_conflict(alignment: NDArray) -> float:
        return float(np.nanmean(np.abs(alignment)))

    def _empty_result(self, T: int) -> dict[str, Any]:
        empty = np.zeros(max(1, T), dtype=np.float64).tolist()
        alignment_empty = np.zeros(max(1, T), dtype=np.float64)
        self._state["cohort_alignment"] = alignment_empty
        return {
            "fast_cohort_activity": empty,
            "medium_cohort_activity": empty,
            "slow_cohort_activity": empty,
            "cohort_dominance": {"fast": 1.0 / 3.0, "medium": 1.0 / 3.0, "slow": 1.0 / 3.0},
            "cohort_conflict": 0.0,
            "cohort_rotation": 0.0,
            "cohort_stress": 0.0,
            "cohort_alignment": alignment_empty,
            "cohort_alignment_mean": 0.0,
            "dominant_cohort": "fast",
            "cohort_regime": 0,
        }
