from __future__ import annotations

from typing import Any

import numpy as np
import numba
from numpy.typing import NDArray

from research.temporal_reality.causality_analysis import AdaptiveTimeCausality
from research.information_discovery.mi_estimator import (
    _fast_mutual_info,
    _fast_conditional_mutual_info,
)


@numba.jit(nopython=True, cache=True)
def _numba_rolling_abs_sum(x: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    n = len(x)
    abs_x = np.abs(x)
    result = np.zeros(n, dtype=np.float64)
    cum = np.cumsum(abs_x)
    result[window - 1] = cum[window - 1]
    result[window:] = cum[window:] - cum[:-window]
    return result


@numba.jit(nopython=True, cache=True)
def _numba_rolling_mean(x: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    n = len(x)
    result = np.zeros(n, dtype=np.float64)
    cum = np.cumsum(x)
    result[window - 1] = cum[window - 1] / float(window)
    result[window:] = (cum[window:] - cum[:-window]) / float(window)
    return result


@numba.jit(nopython=True, cache=True)
def _numba_rolling_std(x: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    n = len(x)
    result = np.zeros(n, dtype=np.float64)
    for i in range(window, n):
        s = 0.0
        c = 0
        for j in range(i - window, i):
            v = x[j]
            if not np.isnan(v):
                s += v
                c += 1
        mean = s / max(c, 1)
        var = 0.0
        for j in range(i - window, i):
            v = x[j]
            if not np.isnan(v):
                var += (v - mean) ** 2
        result[i] = np.sqrt(var / max(c, 1))
    return result


class LatentDriverSearch:
    def __init__(self, n_latent: int = 3, max_lag: int = 50, pca_components: int = 2):
        self.n_latent = n_latent
        self.max_lag = max_lag
        self.pca_components = pca_components

    def search(self, data: dict) -> dict:
        signals = self._extract_signals(data)
        if not signals:
            return self._empty_result()

        n = min(len(v) for v in signals.values())
        keys = list(signals.keys())
        mat = np.column_stack([signals[k][:n].astype(np.float64) for k in keys])

        means = np.mean(mat, axis=0)
        stds = np.maximum(np.std(mat, axis=0), 1e-12)
        mat_z = (mat - means) / stds

        U, S, Vt = np.linalg.svd(mat_z, full_matrices=False)
        n_gen = min(self.pca_components, mat_z.shape[1] - 1)
        predicted = (U[:, :n_gen] * S[:n_gen]) @ Vt[:n_gen, :]
        residuals = mat_z - predicted

        U_r, S_r, _ = np.linalg.svd(residuals, full_matrices=False)
        n_latent = min(self.n_latent, residuals.shape[1])
        latent_factors = U_r[:, :n_latent]

        total_var = np.sum(S_r ** 2)
        explained = np.sum(S_r[:n_latent] ** 2)
        residual_variance = 1.0 - explained / max(total_var, 1e-12)

        factor_correlations = []
        for i in range(n_latent):
            corrs = []
            for j, key in enumerate(keys):
                c_arr = AdaptiveTimeCausality._cross_correlate(
                    latent_factors[:, i], mat_z[:, j], self.max_lag
                )
                peak_idx = int(np.argmax(np.abs(c_arr)))
                corrs.append({"feature": key, "correlation": float(c_arr[peak_idx])})
            corrs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
            factor_correlations.append(corrs)

        identified_drivers = self._compute_candidate_drivers(data, n)

        return {
            "latent_factors": latent_factors,
            "factor_correlations": factor_correlations,
            "identified_drivers": identified_drivers,
            "residual_variance": residual_variance,
        }

    def _extract_signals(self, data: dict) -> dict[str, NDArray[np.float64]]:
        signals: dict[str, NDArray[np.float64]] = {}
        for key, val in data.items():
            if isinstance(val, np.ndarray) and val.ndim == 1 and len(val) > 0:
                signals[key] = np.asarray(val, dtype=np.float64)
        return signals

    def _compute_candidate_drivers(
        self, data: dict, n: int
    ) -> list[dict[str, Any]]:
        drivers: list[dict[str, Any]] = []

        returns = np.asarray(data.get("returns", np.zeros(n)), dtype=np.float64)
        info_acc = _numba_rolling_abs_sum(returns, 50)
        imax = float(np.max(info_acc))
        if imax > 1e-12:
            info_acc = info_acc / imax
        drivers.append({
            "name": "Information Accumulation",
            "formula": "rolling_sum(abs(returns), 50)",
            "series": info_acc,
        })

        adaptive_time = np.asarray(
            data.get("adaptive_time", data.get("adaptive_time_coordinate", np.zeros(n))),
            dtype=np.float64,
        )
        at_change = np.abs(np.diff(adaptive_time, prepend=adaptive_time[0]))
        vol = np.asarray(
            data.get("volatility", _numba_rolling_std(returns, 20)),
            dtype=np.float64,
        )
        market_tension = at_change * vol
        mmax = float(np.max(market_tension))
        if mmax > 1e-12:
            market_tension = market_tension / mmax
        drivers.append({
            "name": "Market Tension",
            "formula": "abs(change(adaptive_time)) * volatility",
            "series": market_tension,
        })

        memory_density = np.asarray(
            data.get("memory_density", np.zeros(n)), dtype=np.float64
        )
        memory_gradient = np.asarray(
            data.get("memory_gradient", np.gradient(memory_density)), dtype=np.float64
        )
        behavioral_conflict = np.abs(memory_density - memory_gradient)
        bmax = float(np.max(behavioral_conflict))
        if bmax > 1e-12:
            behavioral_conflict = behavioral_conflict / bmax
        drivers.append({
            "name": "Behavioral Conflict",
            "formula": "abs(memory_density - memory_gradient)",
            "series": behavioral_conflict,
        })

        energy_storage = np.asarray(
            data.get("energy_storage", np.zeros(n)), dtype=np.float64
        )
        vol_safe = np.maximum(vol, 1e-12)
        energy_comp = _numba_rolling_mean(energy_storage * vol_safe, 20)
        cmax = float(np.max(energy_comp))
        if cmax > 1e-12:
            energy_comp = energy_comp / cmax
        drivers.append({
            "name": "Energy Compression",
            "formula": "rolling_mean(energy_storage * volatility, 20)",
            "series": energy_comp,
        })

        return drivers

    def _empty_result(self) -> dict[str, Any]:
        return {
            "latent_factors": np.zeros((1, self.n_latent), dtype=np.float64),
            "factor_correlations": [],
            "identified_drivers": [],
            "residual_variance": 1.0,
        }
