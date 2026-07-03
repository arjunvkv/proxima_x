from __future__ import annotations

from typing import Any, Optional

import numpy as np
import numba
from numpy.typing import NDArray

from research.mechanism_discovery.base import BaseMechanism, MechanismScore


@numba.jit(nopython=True, cache=True)
def _numba_trailing_mean_std(price: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    n = len(price)
    trailing_mean = np.zeros(n)
    trailing_std = np.zeros(n)
    s = 0.0
    sq = 0.0
    cnt = 0
    for i in range(n):
        val = price[i]
        if not np.isnan(val):
            s += val
            sq += val * val
            cnt += 1
        if cnt == 0:
            trailing_mean[i] = 0.0
            trailing_std[i] = 0.0
        else:
            mean = s / cnt
            trailing_mean[i] = mean
            var = (sq / cnt) - mean * mean
            if var < 0.0:
                var = 0.0
            trailing_std[i] = np.sqrt(var)
    return trailing_mean, trailing_std


@numba.jit(nopython=True, cache=True)
def _numba_energy_dissipation(combined: NDArray[np.float64], peak_mask: NDArray[np.bool_], decay_fit_window: int) -> tuple[NDArray[np.float64], float]:
    N = len(combined)
    dissipation_rate = np.zeros(N, dtype=np.float64)
    
    max_peaks = np.sum(peak_mask)
    decay_rates = np.zeros(max_peaks, dtype=np.float64)
    decay_cnt = 0
    
    for i in range(N):
        if peak_mask[i]:
            end_idx = min(N, i + decay_fit_window)
            post_len = end_idx - i
            if post_len >= 5:
                y_log = np.zeros(post_len)
                for k in range(post_len):
                    val = combined[i + k]
                    if val < 1e-12:
                        val = 1e-12
                    y_log[k] = np.log(val)
                
                mean_y_log = np.mean(y_log)
                var_y_log = np.var(y_log)
                if var_y_log > 1e-24:
                    t_vals = np.arange(post_len, dtype=np.float64)
                    mean_t = np.mean(t_vals)
                    var_t = np.var(t_vals)
                    if var_t > 1e-12:
                        cov_ty = np.mean((t_vals - mean_t) * (y_log - mean_y_log))
                        slope = cov_ty / var_t
                        decay_rates[decay_cnt] = -slope
                        decay_cnt += 1
            
            for j in range(i + 1, end_idx):
                decay = combined[i] * np.exp(-0.1 * (j - i))
                dissipation_rate[j] = max(dissipation_rate[j], float(combined[i] - decay))
                
    dissipation_coeff = 0.0
    if decay_cnt > 0:
        valid_rates = np.sort(decay_rates[:decay_cnt])
        mid = decay_cnt // 2
        if decay_cnt % 2 == 1:
            dissipation_coeff = valid_rates[mid]
        else:
            dissipation_coeff = 0.5 * (valid_rates[mid - 1] + valid_rates[mid])
            
    return dissipation_rate, dissipation_coeff


@numba.jit(nopython=True, cache=True)
def _numba_estimate_half_life(energy: NDArray[np.float64], depth: int) -> float:
    n = len(energy)
    ac = np.zeros(depth, dtype=np.float64)
    for k in range(1, depth + 1):
        if n > k:
            e_lag = energy[:-k]
            e_lead = energy[k:]
            s_lag = np.std(e_lag)
            s_lead = np.std(e_lead)
            if s_lag > 1e-12 and s_lead > 1e-12:
                mean_lag = np.mean(e_lag)
                mean_lead = np.mean(e_lead)
                cov = np.mean((e_lag - mean_lag) * (e_lead - mean_lead))
                ac[k - 1] = cov / (s_lag * s_lead)
            else:
                ac[k - 1] = 0.0
        else:
            ac[k - 1] = 0.0
            
    for i in range(depth):
        if ac[i] < 0.0:
            ac[i] = 0.0
            
    val_sum = 0.0
    for i in range(depth):
        val_sum += ac[i]
        
    if val_sum > 1e-12:
        decays = np.zeros(depth - 1)
        decay_cnt = 0
        for i in range(depth - 1):
            if ac[i] > 1e-12:
                decays[decay_cnt] = ac[i + 1] / ac[i]
                decay_cnt += 1
        if decay_cnt > 0:
            avg_decay = np.mean(decays[:decay_cnt])
            if avg_decay < 1.0:
                if avg_decay < 1e-12:
                    avg_decay = 1e-12
                return -np.log(2.0) / np.log(avg_decay)
                
    return float("inf")


@numba.jit(nopython=True, cache=True)
def _numba_rolling_mean(x: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    N = len(x)
    result = np.zeros(N, dtype=np.float64)
    if N == 0:
        return result
    cum = np.cumsum(x)
    result[window - 1] = cum[window - 1] / float(window)
    result[window:] = (cum[window:] - cum[:-window]) / float(window)
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
def _numba_rolling_sum(x: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    N = len(x)
    result = np.zeros(N, dtype=np.float64)
    if N == 0:
        return result
    cum = np.cumsum(x)
    result[window - 1] = cum[window - 1]
    result[window:] = cum[window:] - cum[:-window]
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


class EnergyDynamics(BaseMechanism):
    STORAGE_WINDOW: int = 50
    SHORT_WINDOW: int = 5
    MEDIUM_WINDOW: int = 20
    RELEASE_STD_THRESHOLD: float = 2.0
    BREAKOUT_STD_THRESHOLD: float = 1.0
    DECAY_FIT_WINDOW: int = 30

    def __init__(self) -> None:
        super().__init__(name="energy_dynamics", category="mechanism_class_4")
        self._state_contribution: NDArray = np.array([], dtype=np.float64)
        self._release_events: list[dict[str, Any]] = []

    def compute(self, data: dict[str, NDArray], states: Optional[NDArray] = None) -> dict[str, Any]:
        price = np.asarray(data.get("price", []), dtype=np.float64)
        returns = np.asarray(data.get("returns", []), dtype=np.float64)
        volume = np.asarray(data.get("volume", []), dtype=np.float64)
        high = np.asarray(data.get("high", []), dtype=np.float64)
        low = np.asarray(data.get("low", []), dtype=np.float64)

        N = len(price)
        if N < self.MEDIUM_WINDOW + 1:
            return self._empty_result(N)

        if len(returns) < N:
            returns = np.diff(price, prepend=price[0])
        if len(volume) < N:
            volume = np.ones(N, dtype=np.float64)
        if len(high) < N:
            high = price.copy()
        if len(low) < N:
            low = price.copy()

        creation = self._compute_energy_creation(price, returns, volume, high, low, N)
        storage = self._compute_energy_storage(returns, volume, high, low, creation, N)
        transfer = self._compute_energy_transfer(returns, creation, N)
        release_dict, release_events = self._compute_energy_release(creation, storage, N)
        dissipation = self._compute_energy_dissipation(creation, release_dict.get("release_magnitude", np.zeros(N, dtype=np.float64)), N)

        balance = creation["combined"] - release_dict.get("release_magnitude", np.zeros(N, dtype=np.float64)) - dissipation["dissipation_rate"]
        regime = self._compute_energy_regime(balance, creation["combined"], N)
        efficiency = self._compute_energy_efficiency(returns, creation["combined"], release_dict.get("release_magnitude", np.zeros(N, dtype=np.float64)), N)

        state_vector = np.column_stack((
            creation["combined"],
            storage["stored_energy"],
            transfer.get("transfer_efficiency", np.zeros(N, dtype=np.float64)),
            release_dict.get("release_magnitude", np.zeros(N, dtype=np.float64)),
            dissipation.get("dissipation_rate", np.zeros(N, dtype=np.float64)),
            balance,
        ))

        self._state_contribution = balance / max(1e-12, float(np.max(np.abs(balance))))
        self._release_events = release_events

        depth = max(1, N // 10)
        half_life = self._estimate_half_life(creation["combined"], dissipation["dissipation_coefficient"], depth)

        result = {
            "energy_creation": creation["combined"],
            "return_energy": creation["return_energy"],
            "volume_energy": creation["volume_energy"],
            "breakout_energy": creation["breakout_energy"],
            "energy_storage": storage["stored_energy"],
            "accumulated_energy": storage["accumulated_energy"],
            "range_expansion_energy": storage["range_expansion"],
            "energy_transfer": transfer.get("transfer_strength", np.zeros(N, dtype=np.float64)),
            "transfer_correlation": transfer.get("correlation", np.zeros(N, dtype=np.float64)),
            "transfer_efficiency": transfer.get("transfer_efficiency", np.zeros(N, dtype=np.float64)),
            "energy_release": release_dict.get("release_magnitude", np.zeros(N, dtype=np.float64)),
            "release_entropy": release_dict.get("release_entropy", 0.0),
            "energy_dissipation": dissipation["dissipation_rate"],
            "dissipation_coefficient": dissipation["dissipation_coefficient"],
            "dissipation_half_life": half_life,
            "market_energy_state": state_vector,
            "energy_balance": balance,
            "energy_regime": regime,
            "energy_efficiency": efficiency,
            "release_events": release_events,
        }
        self._state.update(result)
        return result

    def get_state_contribution(self) -> NDArray:
        return self._state_contribution

    def _compute_energy_creation(
        self, price: NDArray, returns: NDArray, volume: NDArray,
        high: NDArray, low: NDArray, N: int,
    ) -> dict[str, NDArray]:
        return_energy = returns ** 2
        volume_energy = volume * np.abs(returns)

        trailing_mean, trailing_std = _numba_trailing_mean_std(price)
        trailing_std = np.maximum(trailing_std, 1e-12)
        z = (price - trailing_mean) / trailing_std
        breakout_raw = np.maximum(0.0, np.abs(z) - self.BREAKOUT_STD_THRESHOLD)
        breakout_energy = breakout_raw * volume

        w_max = min(50, N)
        ret_vol = self._rolling_mean(np.abs(returns), w_max) + 1e-12
        vol_vol = self._rolling_mean(volume, w_max) + 1e-12
        w_ret = 1.0 / ret_vol
        w_vol = 1.0 / vol_vol
        w_break = np.ones(N, dtype=np.float64)
        w_sum = w_ret + w_vol + w_break + 1e-12
        w_ret = w_ret / w_sum
        w_vol = w_vol / w_sum
        w_break = w_break / w_sum
        combined = w_ret * return_energy + w_vol * volume_energy + w_break * breakout_energy
        combined = np.nan_to_num(combined)

        return {
            "return_energy": return_energy,
            "volume_energy": volume_energy,
            "breakout_energy": breakout_energy,
            "combined": combined,
        }

    def _compute_energy_storage(
        self, returns: NDArray, volume: NDArray, high: NDArray, low: NDArray,
        creation: dict[str, NDArray], N: int,
    ) -> dict[str, NDArray]:
        return_energy = creation["return_energy"]
        acc = self._rolling_sum(return_energy, self.STORAGE_WINDOW)

        vol_weights = self._rolling_mean(volume, self.STORAGE_WINDOW) * self.STORAGE_WINDOW + 1e-12
        vol_weighted_acc = self._rolling_sum(return_energy * volume, self.STORAGE_WINDOW) / vol_weights

        range_exp = (high - low) ** 2
        range_exp = np.nan_to_num(range_exp)

        release_mag = np.zeros(N, dtype=np.float64)
        rel_mean = float(np.mean(acc)) + 1e-12
        rel_std = float(np.std(acc)) + 1e-12
        release_mask = acc > rel_mean + self.RELEASE_STD_THRESHOLD * rel_std
        release_mag[release_mask] = acc[release_mask] - rel_mean

        stored = acc - release_mag
        stored = np.maximum(stored, 0.0)
        stored = np.nan_to_num(stored)

        return {
            "accumulated_energy": acc,
            "volume_weighted_accumulation": vol_weighted_acc,
            "range_expansion": range_exp,
            "stored_energy": stored,
        }

    def _compute_energy_transfer(
        self, returns: NDArray, creation: dict[str, NDArray], N: int,
    ) -> dict[str, NDArray]:
        short_energy = self._rolling_mean(creation["combined"], self.SHORT_WINDOW)
        medium_energy = self._rolling_mean(creation["combined"], self.MEDIUM_WINDOW)

        min_obs = self.MEDIUM_WINDOW * 2
        correlation = _numba_rolling_corr(short_energy, medium_energy, min_obs)

        correlation = np.nan_to_num(correlation)
        transfer_strength = np.abs(correlation)
        transfer_efficiency = np.zeros(N, dtype=np.float64)
        for i in range(1, N):
            if medium_energy[i - 1] > 1e-12:
                flow = (medium_energy[i] - medium_energy[i - 1]) / medium_energy[i - 1]
                transfer_efficiency[i] = float(correlation[i]) * float(flow)
        transfer_efficiency = np.nan_to_num(transfer_efficiency)

        return {
            "correlation": correlation,
            "transfer_strength": transfer_strength,
            "transfer_efficiency": transfer_efficiency,
        }

    def _compute_energy_release(
        self, creation: dict[str, NDArray], storage: dict[str, NDArray], N: int,
    ) -> tuple[dict[str, NDArray], list[dict[str, Any]]]:
        combined = creation["combined"]
        roll_mean = self._rolling_mean(combined, self.STORAGE_WINDOW)
        roll_std = _numba_rolling_std(combined, self.STORAGE_WINDOW)
        
        # Fill leading std NaNs safely
        roll_std = np.nan_to_num(roll_std)
        roll_std = np.maximum(roll_std, 1e-12)
        roll_mean = np.maximum(roll_mean, 0.0)

        threshold = roll_mean + self.RELEASE_STD_THRESHOLD * roll_std
        is_release = combined > threshold

        release_magnitude = np.where(is_release, combined - roll_mean, 0.0)
        release_magnitude = np.nan_to_num(release_magnitude)

        release_sizes = release_magnitude[release_magnitude > 1e-12]
        release_entropy = 0.0
        if len(release_sizes) > 1:
            hist, _ = np.histogram(release_sizes, bins=max(2, len(release_sizes) // 5))
            hist = hist.astype(np.float64)
            hist = hist / max(1e-12, np.sum(hist))
            release_entropy = float(-np.sum(hist * np.log(hist + 1e-12)))

        events: list[dict[str, Any]] = []
        in_event = False
        event_start = 0
        event_peak_idx = 0
        event_peak_val = 0.0
        for i in range(N):
            if is_release[i] and not in_event:
                in_event = True
                event_start = i
                event_peak_idx = i
                event_peak_val = release_magnitude[i]
            elif is_release[i] and in_event:
                if release_magnitude[i] > event_peak_val:
                    event_peak_val = release_magnitude[i]
                    event_peak_idx = i
            elif not is_release[i] and in_event:
                in_event = False
                duration = i - event_start
                recovery = 0
                for j in range(i, min(N, i + self.STORAGE_WINDOW)):
                    if combined[j] <= roll_mean[j]:
                        recovery += 1
                    else:
                        break
                events.append({
                    "start": event_start,
                    "peak": event_peak_idx,
                    "magnitude": float(event_peak_val),
                    "duration": duration,
                    "recovery": recovery,
                })

        top_k = sorted(events, key=lambda e: e["magnitude"], reverse=True)[:10]

        return {
            "release_magnitude": release_magnitude,
            "release_entropy": release_entropy,
            "is_release": is_release.astype(np.float64),
        }, top_k

    def _compute_energy_dissipation(
        self, creation: dict[str, NDArray], release_magnitude: NDArray, N: int,
    ) -> dict[str, Any]:
        combined = creation["combined"]

        peak_mask = np.zeros(N, dtype=bool)
        for i in range(1, N - 1):
            if combined[i] > combined[i - 1] and combined[i] >= combined[i + 1]:
                peak_mask[i] = True

        dissipation_rate, dissipation_coeff = _numba_energy_dissipation(
            combined, peak_mask, self.DECAY_FIT_WINDOW
        )

        dissipation_rate = np.nan_to_num(dissipation_rate)
        dissipation_rate = np.maximum(dissipation_rate, 0.0)

        return {
            "dissipation_rate": dissipation_rate,
            "dissipation_coefficient": dissipation_coeff,
        }

    def _compute_energy_regime(self, balance: NDArray, creation: NDArray, N: int) -> NDArray:
        regime = np.ones(N, dtype=np.int64) * 2
        creation_mean = float(np.mean(creation)) + 1e-12
        creation_std = float(np.std(creation)) + 1e-12
        for i in range(N):
            if creation[i] > creation_mean + 0.5 * creation_std and balance[i] > 0:
                regime[i] = 0
            elif creation[i] > creation_mean + 0.5 * creation_std and balance[i] < 0:
                regime[i] = 1
            else:
                regime[i] = 2
        return regime

    def _compute_energy_efficiency(
        self, returns: NDArray, creation: NDArray, release: NDArray, N: int,
    ) -> NDArray:
        # NOTE: uses contemporaneous return, not forward return, to avoid lookahead
        energy_consumed = creation + release + 1e-12
        efficiency = returns / energy_consumed
        efficiency = np.nan_to_num(efficiency)
        emax = float(np.max(np.abs(efficiency)))
        if emax > 1e-12:
            efficiency = efficiency / emax
        return efficiency

    def _estimate_half_life(self, energy: NDArray, dissipation_coeff: float, depth: int) -> float:
        if dissipation_coeff <= 1e-12:
            return float("inf")
        return _numba_estimate_half_life(energy, depth)

    @staticmethod
    def _rolling_sum(x: NDArray, window: int) -> NDArray:
        return _numba_rolling_sum(x.astype(np.float64), window)

    @staticmethod
    def _rolling_mean(x: NDArray, window: int) -> NDArray:
        return _numba_rolling_mean(x.astype(np.float64), window)

    def _empty_result(self, N: int) -> dict[str, Any]:
        size = max(1, N)
        self._state_contribution = np.zeros(size, dtype=np.float64)
        return {
            "energy_creation": np.zeros(size, dtype=np.float64),
            "return_energy": np.zeros(size, dtype=np.float64),
            "volume_energy": np.zeros(size, dtype=np.float64),
            "breakout_energy": np.zeros(size, dtype=np.float64),
            "energy_storage": np.zeros(size, dtype=np.float64),
            "accumulated_energy": np.zeros(size, dtype=np.float64),
            "range_expansion_energy": np.zeros(size, dtype=np.float64),
            "energy_transfer": np.zeros(size, dtype=np.float64),
            "transfer_correlation": np.zeros(size, dtype=np.float64),
            "transfer_efficiency": np.zeros(size, dtype=np.float64),
            "energy_release": np.zeros(size, dtype=np.float64),
            "release_entropy": 0.0,
            "energy_dissipation": np.zeros(size, dtype=np.float64),
            "dissipation_coefficient": 0.0,
            "dissipation_half_life": float("inf"),
            "market_energy_state": np.zeros((size, 6), dtype=np.float64),
            "energy_balance": np.zeros(size, dtype=np.float64),
            "energy_regime": np.ones(size, dtype=np.int64) * 2,
            "energy_efficiency": np.zeros(size, dtype=np.float64),
            "release_events": [],
        }
