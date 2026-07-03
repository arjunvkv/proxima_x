from __future__ import annotations

from typing import Optional

import polars as pl
import numpy as np
import numba
from numpy.typing import NDArray


@numba.jit(nopython=True, cache=True)
def _market_memory_residue_numba(
    price: NDArray[np.float64], window: int
) -> NDArray[np.float32]:
    n = len(price)
    result = np.zeros((n, 5), dtype=np.float32)
    degree = 3
    w = window
    t = np.arange(w, dtype=np.float64)
    X = np.empty((w, degree + 1), dtype=np.float64)
    for d in range(degree + 1):
        for i in range(w):
            X[i, d] = t[i] ** d
    XtX = np.zeros((degree + 1, degree + 1), dtype=np.float64)
    for i in range(degree + 1):
        for j in range(degree + 1):
            s = 0.0
            for k in range(w):
                s += X[k, i] * X[k, j]
            XtX[i, j] = s
    XtX_inv = np.linalg.inv(XtX)
    M = XtX_inv @ X.T
    for i in range(w - 1, n):
        y = price[i - w + 1 : i + 1]
        beta = M @ y
        pred = X @ beta
        residuals = y - pred
        r_mean = np.mean(residuals)
        r_std = np.std(residuals)
        result[i, 0] = residuals[-1]
        if r_std > 1e-10:
            result[i, 1] = r_std
            skew = np.mean(((residuals - r_mean) / r_std) ** 3)
            result[i, 2] = skew
        r_min = np.min(residuals)
        r_max = np.max(residuals)
        if r_max - r_min > 1e-10:
            nb = 10
            bin_edges = np.linspace(r_min, r_max, nb + 1)
            hist = np.zeros(nb)
            for j in range(w):
                idx = int((residuals[j] - r_min) / (r_max - r_min) * nb)
                if idx >= nb:
                    idx = nb - 1
                hist[idx] += 1
            probs = hist / w
            ent = 0.0
            for j in range(nb):
                if probs[j] > 0:
                    ent -= probs[j] * np.log2(probs[j])
            result[i, 3] = ent
        a = residuals[:-1]
        b = residuals[1:]
        ma = np.mean(a)
        mb = np.mean(b)
        num = np.sum((a - ma) * (b - mb))
        den = np.sqrt(np.sum((a - ma) ** 2) * np.sum((b - mb) ** 2))
        if den > 1e-10:
            result[i, 4] = num / den
    return result


@numba.jit(nopython=True, cache=True)
def _behavioral_inertia_numba(
    returns: NDArray[np.float64], volume: NDArray[np.float64], window: int
) -> NDArray[np.float32]:
    n = min(len(returns), len(volume))
    result = np.zeros((n, 3), dtype=np.float32)
    for i in range(window - 1, n):
        r_seg = returns[i - window + 1 : i + 1]
        v_seg = volume[i - window + 1 : i + 1]
        v_sum = np.sum(v_seg)
        if v_sum > 1e-10:
            wts = v_seg / v_sum
        else:
            wts = np.ones(window, dtype=np.float64) / window
        wr = r_seg * wts
        a = wr[:-1]
        b = wr[1:]
        ma = np.mean(a)
        mb = np.mean(b)
        num = np.sum((a - ma) * (b - mb))
        den = np.sqrt(np.sum((a - ma) ** 2) * np.sum((b - mb) ** 2))
        if den > 1e-10:
            result[i, 0] = num / den
        pos_dir = 0.0
        neg_dir = 0.0
        for j in range(window):
            if wr[j] > 0:
                pos_dir += wr[j]
            else:
                neg_dir += wr[j]
        total_dir = pos_dir + abs(neg_dir)
        if total_dir > 1e-10:
            result[i, 1] = (pos_dir - abs(neg_dir)) / total_dir
        vol_ratio = 0.0
        if i >= 2 * window - 1:
            prev_vol = volume[i - 2 * window + 1 : i - window + 1]
            pv_mean = np.mean(prev_vol)
            cv_mean = np.mean(v_seg)
            if pv_mean > 1e-10:
                vol_ratio = cv_mean / pv_mean - 1.0
        r_vol = np.std(r_seg)
        if r_vol > 1e-10:
            result[i, 2] = abs(result[i, 0]) / r_vol + abs(vol_ratio)
    return result


@numba.jit(nopython=True, cache=True)
def _state_momentum_numba(
    returns: NDArray[np.float64], window: int
) -> NDArray[np.float32]:
    n = len(returns)
    result = np.zeros((n, 4), dtype=np.float32)
    vol_window = max(5, window // 2)
    entropy_window = max(10, window)
    vol_arr = np.zeros(n, dtype=np.float32)
    ent_arr = np.zeros(n, dtype=np.float32)
    for i in range(vol_window - 1, n):
        seg = returns[i - vol_window + 1 : i + 1]
        vol_arr[i] = np.std(seg)
    for i in range(entropy_window - 1, n):
        seg = returns[i - entropy_window + 1 : i + 1]
        lo = np.min(seg)
        hi = np.max(seg)
        if hi - lo > 1e-10:
            nb = 10
            bin_edges = np.linspace(lo, hi, nb + 1)
            hist = np.zeros(nb)
            for j in range(entropy_window):
                idx = int((seg[j] - lo) / (hi - lo) * nb)
                if idx >= nb:
                    idx = nb - 1
                hist[idx] += 1
            probs = hist / entropy_window
            e = 0.0
            for j in range(nb):
                if probs[j] > 0:
                    e -= probs[j] * np.log2(probs[j])
            ent_arr[i] = e
    for i in range(window, n):
        result[i, 0] = vol_arr[i] - vol_arr[i - 1]
        result[i, 1] = ent_arr[i] - ent_arr[i - 1]
    for i in range(window + 1, n):
        ret_diff1 = returns[i] - returns[i - 1]
        ret_diff2 = returns[i - 1] - returns[i - 2]
        result[i, 2] = ret_diff1 - ret_diff2
    for i in range(window + 2, n):
        dd1 = returns[i] - returns[i - 1]
        dd2 = returns[i - 1] - returns[i - 2]
        dd3 = returns[i - 2] - returns[i - 3]
        result[i, 3] = (dd1 - dd2) - (dd2 - dd3)
    return result


@numba.jit(nopython=True, cache=True)
def _information_compression_numba(
    price: NDArray[np.float64], returns: NDArray[np.float64], window: int
) -> NDArray[np.float32]:
    n = min(len(price), len(returns))
    result = np.zeros((n, 3), dtype=np.float32)
    for i in range(window - 1, n):
        r_seg = returns[i - window + 1 : i + 1]
        p_seg = price[i - window + 1 : i + 1]
        r_std = np.std(r_seg)
        actual_range = np.max(p_seg) - np.min(p_seg)
        if r_std > 1e-10:
            expected_range = r_std * np.sqrt(float(window))
            if expected_range > 1e-10:
                result[i, 0] = actual_range / expected_range
        net_chg = abs(p_seg[-1] - p_seg[0])
        sum_chg = np.sum(np.abs(np.diff(p_seg)))
        if sum_chg > 1e-10:
            er = net_chg / sum_chg
            if i >= window + 5:
                prev_er = 0.0
                prev_net = abs(price[i - 5] - price[i - window])
                prev_sum = np.sum(np.abs(np.diff(price[i - window : i - 4])))
                if prev_sum > 1e-10:
                    prev_er = prev_net / prev_sum
                result[i, 1] = er - prev_er
            result[i, 2] = 1.0 - er
    return result


@numba.jit(nopython=True, cache=True)
def _liquidity_vacuum_numba(
    price: NDArray[np.float64], volume: NDArray[np.float64], window: int
) -> NDArray[np.float32]:
    n = min(len(price), len(volume))
    result = np.zeros((n, 3), dtype=np.float32)
    for i in range(window - 1, n):
        v_seg = volume[i - window + 1 : i + 1]
        p_seg = price[i - window + 1 : i + 1]
        v_mean = np.mean(v_seg)
        recent_v = np.mean(v_seg[-5:]) if window >= 5 else v_mean
        if v_mean > 1e-10:
            result[i, 0] = max(0.0, 1.0 - recent_v / v_mean)
        prev_idx = max(0, i - window)
        ret = price[i] - price[max(0, i - 1)]
        v_cur = max(volume[i], 1e-10)
        result[i, 1] = abs(ret) / v_cur * v_mean if v_mean > 1e-10 else 0.0
        p_diffs = np.abs(np.diff(p_seg))
        max_gap = np.max(p_diffs) if len(p_diffs) > 0 else 0.0
        p_std = np.std(p_seg)
        if p_std > 1e-10:
            result[i, 2] = max_gap / p_std
    return result


@numba.jit(nopython=True, cache=True)
def _temporal_resonance_numba(
    returns: NDArray[np.float64], window: int
) -> NDArray[np.float32]:
    n = len(returns)
    result = np.zeros((n, 3), dtype=np.float32)
    fast_w = max(3, window // 5)
    med_w = max(5, window // 2)
    slow_w = max(10, window)
    for i in range(slow_w - 1, n):
        fast_seg = returns[i - fast_w + 1 : i + 1]
        med_seg = returns[i - med_w + 1 : i + 1]
        slow_seg = returns[i - slow_w + 1 : i + 1]
        f_mean = np.mean(fast_seg)
        m_mean = np.mean(med_seg)
        s_mean = np.mean(slow_seg)
        def _corr(a, b, ma, mb):
            num = np.sum((a - ma) * (b - mb))
            den = np.sqrt(np.sum((a - ma) ** 2) * np.sum((b - mb) ** 2))
            if den > 1e-10:
                return num / den
            return 0.0
        if fast_w <= med_w:
            med_aligned = med_seg[-fast_w:]
            result[i, 0] = _corr(fast_seg, med_aligned, f_mean, np.mean(med_aligned))
        if med_w <= slow_w:
            slow_aligned = slow_seg[-med_w:]
            result[i, 1] = _corr(med_seg, slow_aligned, m_mean, np.mean(slow_aligned))
        c = result[i, 0] * result[i, 1]
        if c > 1e-10:
            result[i, 2] = c * (1.0 - abs(result[i, 0] - result[i, 1]))
    return result


@numba.jit(nopython=True, cache=True)
def _market_fatigue_numba(
    returns: NDArray[np.float64], volume: NDArray[np.float64], window: int
) -> NDArray[np.float32]:
    n = min(len(returns), len(volume))
    result = np.zeros((n, 3), dtype=np.float32)
    for i in range(window - 1, n):
        r_seg = returns[i - window + 1 : i + 1]
        v_seg = volume[i - window + 1 : i + 1]
        v_sum = np.sum(v_seg)
        if v_sum > 1e-10:
            result[i, 0] = np.sum(np.abs(r_seg)) / v_sum
        half = window // 2
        first_half = r_seg[:half]
        second_half = r_seg[half:]
        fv = np.std(first_half)
        sv = np.std(second_half)
        if fv > 1e-10:
            result[i, 1] = sv / fv - 1.0
        cum_ret = np.sum(r_seg)
        ret_mag = np.sum(np.abs(r_seg))
        if ret_mag > 1e-10:
            recent = np.sum(np.abs(r_seg[-min(5, window):]))
            result[i, 2] = recent / ret_mag
    return result


@numba.jit(nopython=True, cache=True)
def _entropy_shock_numba(
    returns: NDArray[np.float64], window: int
) -> NDArray[np.float32]:
    n = len(returns)
    result = np.zeros((n, 3), dtype=np.float32)
    ent = np.zeros(n, dtype=np.float32)
    for i in range(window - 1, n):
        seg = returns[i - window + 1 : i + 1]
        lo = np.min(seg)
        hi = np.max(seg)
        if hi - lo > 1e-10:
            nb = 10
            bin_edges = np.linspace(lo, hi, nb + 1)
            hist = np.zeros(nb)
            for j in range(window):
                idx = int((seg[j] - lo) / (hi - lo) * nb)
                if idx >= nb:
                    idx = nb - 1
                hist[idx] += 1
            probs = hist / window
            e = 0.0
            for j in range(nb):
                if probs[j] > 0:
                    e -= probs[j] * np.log2(probs[j])
            ent[i] = e
    for i in range(window + 1, n):
        result[i, 0] = ent[i] - ent[i - 1]
    for i in range(window + 2, n):
        result[i, 1] = (ent[i] - ent[i - 1]) - (ent[i - 1] - ent[i - 2])
    baseline_start = window - 1
    baseline_len = n - baseline_start
    if baseline_len > 10:
        baseline = ent[baseline_start:]
        b_mean = np.mean(baseline)
        b_std = np.std(baseline)
        for i in range(window - 1, n):
            if b_std > 1e-10:
                result[i, 2] = (ent[i] - b_mean) / b_std
    return result


@numba.jit(nopython=True, cache=True)
def _state_gravity_numba(
    price: NDArray[np.float64],
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    window: int,
) -> NDArray[np.float32]:
    n = min(len(price), len(high), len(low))
    result = np.zeros((n, 3), dtype=np.float32)
    for i in range(window - 1, n):
        p_seg = price[i - window + 1 : i + 1]
        h_seg = high[i - window + 1 : i + 1]
        l_seg = low[i - window + 1 : i + 1]
        typical = (h_seg + l_seg + p_seg) / 3.0
        vwap = np.mean(typical)
        p_std = np.std(p_seg)
        if p_std > 1e-10:
            result[i, 0] = (price[i] - vwap) / p_std
        dist = abs(price[i] - vwap)
        if dist > 1e-10:
            vol_context = np.mean(h_seg - l_seg)
            if vol_context > 1e-10:
                result[i, 1] = 1.0 / (1.0 + dist / vol_context)
        if i >= 1:
            velocity = price[i] - price[i - 1]
            if p_std > 1e-10:
                result[i, 2] = abs(velocity) / p_std
    return result


@numba.jit(nopython=True, cache=True)
def _behavioral_potential_numba(
    returns: NDArray[np.float64], volume: NDArray[np.float64], window: int
) -> NDArray[np.float32]:
    n = min(len(returns), len(volume))
    result = np.zeros((n, 3), dtype=np.float32)
    for i in range(window - 1, n):
        r_seg = returns[i - window + 1 : i + 1]
        v_seg = volume[i - window + 1 : i + 1]
        v_sum = np.sum(v_seg)
        pressure = 0.0
        energy = 0.0
        for j in range(window):
            pressure += r_seg[j] * v_seg[j]
            energy += abs(r_seg[j]) * v_seg[j]
        if v_sum > 1e-10:
            result[i, 0] = pressure / v_sum
            result[i, 1] = energy / v_sum
        mean_vol = np.std(r_seg)
        if i >= 2 * window - 1:
            prev_r = returns[i - 2 * window + 1 : i - window + 1]
            prev_vol = np.std(prev_r)
            if mean_vol + prev_vol > 1e-10:
                result[i, 2] = mean_vol / (mean_vol + prev_vol)
        elif mean_vol > 1e-10:
            result[i, 2] = 0.5
    return result


class StateDiscoveryEngine:

    def __init__(self, n_components_target: int = 100) -> None:
        self.n_components_target = n_components_target

    def _compute_market_memory_residue(
        self,
        price: NDArray[np.float64],
        returns: NDArray[np.float64],
        window: int = 100,
    ) -> NDArray[np.float32]:
        w = min(window, len(price))
        return _market_memory_residue_numba(price.astype(np.float64), w)

    def _compute_behavioral_inertia(
        self,
        returns: NDArray[np.float64],
        volume: NDArray[np.float64],
        window: int = 50,
    ) -> NDArray[np.float32]:
        w = min(window, len(returns), len(volume))
        return _behavioral_inertia_numba(
            returns.astype(np.float64), volume.astype(np.float64), w
        )

    def _compute_state_momentum(
        self,
        returns: NDArray[np.float64],
        features: dict | None = None,
        window: int = 20,
    ) -> NDArray[np.float32]:
        w = min(window, len(returns))
        return _state_momentum_numba(returns.astype(np.float64), w)

    def _compute_information_compression(
        self,
        price: NDArray[np.float64],
        returns: NDArray[np.float64],
        window: int = 100,
    ) -> NDArray[np.float32]:
        w = min(window, len(price), len(returns))
        return _information_compression_numba(
            price.astype(np.float64), returns.astype(np.float64), w
        )

    def _compute_liquidity_vacuum(
        self,
        price: NDArray[np.float64],
        volume: NDArray[np.float64],
        window: int = 50,
    ) -> NDArray[np.float32]:
        w = min(window, len(price), len(volume))
        return _liquidity_vacuum_numba(
            price.astype(np.float64), volume.astype(np.float64), w
        )

    def _compute_temporal_resonance(
        self,
        returns: NDArray[np.float64],
        window: int = 100,
    ) -> NDArray[np.float32]:
        w = min(window, len(returns))
        return _temporal_resonance_numba(returns.astype(np.float64), w)

    def _compute_market_fatigue(
        self,
        returns: NDArray[np.float64],
        volume: NDArray[np.float64],
        window: int = 100,
    ) -> NDArray[np.float32]:
        w = min(window, len(returns), len(volume))
        return _market_fatigue_numba(
            returns.astype(np.float64), volume.astype(np.float64), w
        )

    def _compute_entropy_shock(
        self,
        returns: NDArray[np.float64],
        window: int = 50,
    ) -> NDArray[np.float32]:
        w = min(window, len(returns))
        return _entropy_shock_numba(returns.astype(np.float64), w)

    def _compute_state_gravity(
        self,
        price: NDArray[np.float64],
        high: NDArray[np.float64],
        low: NDArray[np.float64],
        window: int = 200,
    ) -> NDArray[np.float32]:
        w = min(window, len(price), len(high), len(low))
        return _state_gravity_numba(
            price.astype(np.float64), high.astype(np.float64), low.astype(np.float64), w
        )

    def _compute_behavioral_potential(
        self,
        returns: NDArray[np.float64],
        volume: NDArray[np.float64],
        window: int = 100,
    ) -> NDArray[np.float32]:
        w = min(window, len(returns), len(volume))
        return _behavioral_potential_numba(
            returns.astype(np.float64), volume.astype(np.float64), w
        )

    def compute_all_features(
        self,
        price: NDArray[np.float64],
        high: NDArray[np.float64],
        low: NDArray[np.float64],
        returns: NDArray[np.float64],
        volume: NDArray[np.float64],
    ) -> dict[str, NDArray[np.float32]]:
        price = np.asarray(price, dtype=np.float64)
        high = np.asarray(high, dtype=np.float64)
        low = np.asarray(low, dtype=np.float64)
        returns = np.asarray(returns, dtype=np.float64)
        volume = np.asarray(volume, dtype=np.float64)
        return {
            "market_memory_residue": self._compute_market_memory_residue(price, returns),
            "behavioral_inertia": self._compute_behavioral_inertia(returns, volume),
            "state_momentum": self._compute_state_momentum(returns),
            "information_compression": self._compute_information_compression(price, returns),
            "liquidity_vacuum": self._compute_liquidity_vacuum(price, volume),
            "temporal_resonance": self._compute_temporal_resonance(returns),
            "market_fatigue": self._compute_market_fatigue(returns, volume),
            "entropy_shock": self._compute_entropy_shock(returns),
            "state_gravity": self._compute_state_gravity(price, high, low),
            "behavioral_potential": self._compute_behavioral_potential(returns, volume),
        }

    def build_state_vector(
        self,
        price: NDArray[np.float64],
        high: NDArray[np.float64],
        low: NDArray[np.float64],
        returns: NDArray[np.float64],
        volume: NDArray[np.float64],
    ) -> NDArray[np.float32]:
        features = self.compute_all_features(price, high, low, returns, volume)
        arrays = list(features.values())
        n = min(a.shape[0] for a in arrays)
        cols = sum(a.shape[1] for a in arrays)
        result = np.zeros((n, cols), dtype=np.float32)
        offset = 0
        for a in arrays:
            c = a.shape[1]
            result[:, offset : offset + c] = a[:n]
            offset += c
        return result

    def build_hybrid_state_vector(
        self,
        price: NDArray[np.float64],
        high: NDArray[np.float64],
        low: NDArray[np.float64],
        returns: NDArray[np.float64],
        volume: NDArray[np.float64],
        existing_features: Optional[dict[str, NDArray[np.float32]]] = None,
    ) -> NDArray[np.float32]:
        discovery = self.build_state_vector(price, high, low, returns, volume)
        n = discovery.shape[0]
        if existing_features is None or not existing_features:
            return discovery
        ext_arrays = []
        for arr in existing_features.values():
            arr_np = np.asarray(arr, dtype=np.float32)
            if arr_np.ndim == 1:
                arr_np = arr_np.reshape(-1, 1)
            if arr_np.shape[0] == n:
                ext_arrays.append(arr_np)
        if not ext_arrays:
            return discovery
        ext_total = sum(a.shape[1] for a in ext_arrays)
        result = np.zeros((n, discovery.shape[1] + ext_total), dtype=np.float32)
        result[:, : discovery.shape[1]] = discovery
        offset = discovery.shape[1]
        for a in ext_arrays:
            c = a.shape[1]
            result[:, offset : offset + c] = a
            offset += c
        return result
