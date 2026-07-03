from __future__ import annotations

from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.stats import f as f_dist, linregress, pearsonr

from research.information_discovery.mi_estimator import MIEstimator
from research.mechanism_discovery.base import BaseMechanism, MechanismScore


class PropagationNetwork(BaseMechanism):
    MAX_LAGS: int = 20
    SIG_THRESHOLD: float = 0.05
    TE_THRESHOLD: float = 0.01

    def __init__(self) -> None:
        super().__init__(name="propagation_network", category="information_propagation")
        self._mi_estimator = MIEstimator()
        self._state_contribution: NDArray = np.array([], dtype=np.float64)
        self._fallback_price: NDArray = np.array([], dtype=np.float64)
        self._edges: list[dict[str, Any]] = []
        self._propagation_velocity: float = 0.0
        self._propagation_acceleration: NDArray = np.array([], dtype=np.float64)
        self._propagation_decay: float = 0.0
        self._propagation_entropy: float = 0.0
        self._propagation_reach: float = 0.0
        self._dominant_direction: str = ""

    def compute(self, data: dict[str, NDArray], states: Optional[NDArray] = None) -> dict[str, Any]:
        prices_raw = data.get("prices", {})
        if isinstance(prices_raw, np.ndarray):
            self._fallback_price = np.asarray(prices_raw, dtype=np.float64)
            return self._empty_result()
        prices_dict = prices_raw
        asset_ids = list(prices_dict.keys())
        n_assets = len(asset_ids)
        if n_assets < 2:
            if n_assets == 1:
                self._fallback_price = np.asarray(prices_dict[asset_ids[0]], dtype=np.float64)
            return self._empty_result()

        arrays = {aid: np.asarray(prices_dict[aid], dtype=np.float64) for aid in asset_ids}
        T = min(len(arr) for arr in arrays.values())
        aligned = {aid: arr[:T] for aid, arr in arrays.items()}

        edges: list[dict[str, Any]] = []
        n = n_assets
        te_matrix = np.zeros((n, n), dtype=np.float64)
        delay_matrix = np.zeros((n, n), dtype=np.float64)

        for i, src in enumerate(asset_ids):
            for j, dst in enumerate(asset_ids):
                if i == j:
                    continue
                x = aligned[src]
                y = aligned[dst]

                delay, xcorr_peak = self._cross_correlation_delay(x, y)
                p_value = self._granger_causality(x, y, max_lag=min(5, self.MAX_LAGS))
                te = self._mi_estimator.transfer_entropy(x, y, lag=max(1, int(abs(delay))))

                velocity = 1.0 / delay if delay > 0 else 0.0

                edges.append({
                    "source": src,
                    "dest": dst,
                    "delay": int(delay),
                    "velocity": velocity,
                    "info_flow": te,
                    "granger_pvalue": p_value,
                    "xcorr_peak": xcorr_peak,
                })
                te_matrix[i, j] = te
                delay_matrix[i, j] = delay

        velocities = np.array([e["velocity"] for e in edges if e["velocity"] > 0], dtype=np.float64)
        self._propagation_velocity = float(np.mean(velocities)) if len(velocities) > 0 else 0.0

        self._propagation_acceleration = self._compute_rolling_acceleration(aligned, asset_ids, T)

        self._propagation_decay = self._compute_propagation_decay(aligned, asset_ids)

        delays = delay_matrix[delay_matrix > 0]
        self._propagation_entropy = self._compute_delay_entropy(delays) if len(delays) > 0 else 0.0

        self._propagation_reach = self._compute_reach(te_matrix, delay_matrix, n)

        self._dominant_direction = self._compute_dominant_direction(te_matrix, asset_ids)

        self._edges = edges
        self._state_contribution = self._build_state_contribution(T)

        result = {
            "network_edges": edges,
            "propagation_velocity": self._propagation_velocity,
            "propagation_acceleration": self._propagation_acceleration.tolist(),
            "propagation_decay": self._propagation_decay,
            "propagation_entropy": self._propagation_entropy,
            "propagation_reach": self._propagation_reach,
            "dominant_direction": self._dominant_direction,
            "delay_matrix": delay_matrix.tolist(),
            "te_matrix": te_matrix.tolist(),
        }
        self._state.update(result)
        return result

    def get_state_contribution(self) -> NDArray:
        if len(self._state_contribution) == 0 and len(self._fallback_price) > 0:
            return self.propagate_forward(self._fallback_price)
        return self._state_contribution

    @staticmethod
    def propagate_forward(price: NDArray) -> NDArray:
        arr = np.asarray(price, dtype=np.float64)
        T = len(arr)
        sig = np.zeros(T, dtype=np.float64)
        if T < 2:
            return sig
        returns = np.diff(arr)
        window = min(10, T // 2) if T > 4 else max(1, T - 1)
        rolling_vol = np.zeros(T, dtype=np.float64)
        for t in range(T):
            start = max(0, t - window)
            end = min(T - 1, t + window)
            seg = arr[start:end + 1]
            if len(seg) > 1:
                rolling_vol[t] = float(np.std(seg))
        influence = np.abs(returns) * rolling_vol[1:]
        sig[1:] = np.cumsum(influence) / (np.arange(1, T, dtype=np.float64) + 1.0)
        max_abs = float(np.max(np.abs(sig)))
        if max_abs > 1e-12:
            sig = sig / max_abs
        return sig

    def _cross_correlation_delay(self, x: NDArray, y: NDArray) -> tuple[int, float]:
        x_clean = x - np.nanmean(x)
        y_clean = y - np.nanmean(y)
        x_clean = np.nan_to_num(x_clean)
        y_clean = np.nan_to_num(y_clean)
        n = len(x_clean)
        xcorr = np.correlate(x_clean, y_clean, mode="same")
        denom = np.sqrt(np.sum(x_clean ** 2) * np.sum(y_clean ** 2))
        if denom < 1e-12:
            return 0, 0.0
        xcorr = xcorr / denom
        center = n // 2
        half = min(self.MAX_LAGS, center, n - center - 1)
        if half < 1:
            return 0, 0.0
        lag_slice = xcorr[center - half: center + half + 1]
        max_idx = int(np.argmax(np.abs(lag_slice)))
        delay = max_idx - half
        peak_val = float(lag_slice[max_idx])
        return delay, peak_val

    def _granger_causality(self, x: NDArray, y: NDArray, max_lag: int = 5) -> float:
        n = len(y)
        if n <= max_lag + 2:
            return 1.0
        y_lags = np.column_stack([y[max_lag - k - 1: n - k - 1] for k in range(max_lag)])
        x_lags = np.column_stack([x[max_lag - k - 1: n - k - 1] for k in range(max_lag)])
        y_target = y[max_lag:]
        mask = ~(np.any(np.isnan(y_lags), axis=1) | np.any(np.isnan(x_lags), axis=1) | np.isnan(y_target))
        y_lags = y_lags[mask]
        x_lags = x_lags[mask]
        y_target = y_target[mask]
        n_obs = len(y_target)
        if n_obs <= max_lag + 2:
            return 1.0
        X_rest = np.column_stack([np.ones(n_obs, dtype=np.float64), y_lags])
        X_unrest = np.column_stack([np.ones(n_obs, dtype=np.float64), y_lags, x_lags])
        try:
            beta_rest = np.linalg.lstsq(X_rest, y_target, rcond=None)[0]
            beta_unrest = np.linalg.lstsq(X_unrest, y_target, rcond=None)[0]
            rss_rest = float(np.sum((y_target - X_rest @ beta_rest) ** 2))
            rss_unrest = float(np.sum((y_target - X_unrest @ beta_unrest) ** 2))
            if rss_unrest < 1e-15:
                return 0.0
            df_num = max_lag
            df_den = n_obs - 2 * max_lag - 1
            if df_den <= 0:
                return 1.0
            f_stat = ((rss_rest - rss_unrest) / df_num) / (rss_unrest / df_den)
            if f_stat < 0.0:
                return 1.0
            return float(1.0 - f_dist.cdf(f_stat, df_num, df_den))
        except np.linalg.LinAlgError:
            return 1.0

    def _compute_rolling_acceleration(
        self,
        aligned: dict[str, NDArray],
        asset_ids: list[str],
        T: int,
    ) -> NDArray:
        window = min(10, T // 4)
        if T < 2 * window:
            return np.array([], dtype=np.float64)
        velocities: list[float] = []
        for t in range(T - window):
            vels: list[float] = []
            n = len(asset_ids)
            for i, src in enumerate(asset_ids):
                for j, dst in enumerate(asset_ids):
                    if i == j:
                        continue
                    x_seg = aligned[src][t:t + window]
                    y_seg = aligned[dst][t:t + window]
                    if len(x_seg) < 3:
                        continue
                    delay, _ = self._cross_correlation_delay(x_seg, y_seg)
                    if delay > 0:
                        vels.append(1.0 / delay)
            if len(vels) > 0:
                velocities.append(float(np.mean(vels)))
        if len(velocities) < 2:
            return np.array([], dtype=np.float64)
        v_arr = np.array(velocities, dtype=np.float64)
        mean_v = float(np.mean(v_arr))
        if mean_v < 1e-12:
            return np.zeros(len(v_arr) - 1, dtype=np.float64)
        return np.diff(v_arr) / mean_v

    def _compute_propagation_decay(
        self,
        aligned: dict[str, NDArray],
        asset_ids: list[str],
    ) -> float:
        lags = np.arange(1, self.MAX_LAGS + 1, dtype=np.int64)
        n = len(asset_ids)
        n_pairs = n * (n - 1)
        n_total = n_pairs * self.MAX_LAGS
        corrs = np.zeros(n_total, dtype=np.float64)
        idx = 0
        for i, src in enumerate(asset_ids):
            for j, dst in enumerate(asset_ids):
                if i == j:
                    continue
                x = aligned[src]
                y = aligned[dst]
                for lag in lags:
                    if lag >= len(x) or lag >= len(y):
                        corrs[idx] = 0.0
                    else:
                        r, _ = pearsonr(x[:-lag], y[lag:])
                        corrs[idx] = abs(r)
                    idx += 1
        corrs = np.clip(corrs, 1e-12, None)
        log_corrs = np.log(corrs)
        lags_flat = np.tile(lags, n_pairs).astype(np.float64)
        if np.std(lags_flat) < 1e-12:
            return 0.0
        try:
            slope, _, _, _, _ = linregress(lags_flat, log_corrs)
            return float(-slope)
        except (ValueError, RuntimeError):
            return 0.0

    def _compute_delay_entropy(self, delays: NDArray) -> float:
        int_delays = delays.astype(np.int64)
        counts = np.bincount(int_delays)
        probs = counts[counts > 0].astype(np.float64) / float(len(int_delays))
        return -float(np.sum(probs * np.log(probs)))

    def _compute_reach(self, te_matrix: NDArray, delay_matrix: NDArray, n: int) -> float:
        if n < 2:
            return 0.0
        connected = (te_matrix > self.TE_THRESHOLD) | (delay_matrix != 0)
        affected = np.any(connected, axis=0)
        return float(np.sum(affected)) / float(n)

    def _compute_dominant_direction(self, te_matrix: NDArray, asset_ids: list[str]) -> str:
        if len(asset_ids) == 0:
            return ""
        outflows = np.sum(te_matrix, axis=1)
        max_val = float(np.max(outflows))
        if max_val < 1e-12:
            return str(asset_ids[0])
        return str(asset_ids[int(np.argmax(outflows))])

    def _build_state_contribution(self, T: int) -> NDArray:
        contrib = np.full(T, self._propagation_velocity, dtype=np.float64)
        if len(self._propagation_acceleration) > 0:
            accel_padded = np.pad(
                self._propagation_acceleration,
                (0, max(0, T - len(self._propagation_acceleration))),
                mode="edge",
            )[:T]
            contrib += accel_padded
        return contrib

    def _empty_result(self) -> dict[str, Any]:
        return {
            "network_edges": [],
            "propagation_velocity": 0.0,
            "propagation_acceleration": [],
            "propagation_decay": 0.0,
            "propagation_entropy": 0.0,
            "propagation_reach": 0.0,
            "dominant_direction": "",
            "delay_matrix": [],
            "te_matrix": [],
        }
