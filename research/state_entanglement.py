from __future__ import annotations

from typing import ClassVar

import numpy as np
import numba
from numpy.typing import NDArray

from config.settings import settings


@numba.jit(nopython=True)
def _rolling_corr(fast: NDArray[np.float64], slow: NDArray[np.float64], window: int, out: NDArray[np.float32]) -> None:
    n = len(fast)
    for i in range(window, n):
        f = fast[i - window : i]
        s = slow[i - window : i]
        f_mean = np.mean(f)
        s_mean = np.mean(s)
        f_std = np.std(f)
        s_std = np.std(s)
        if f_std > 1e-10 and s_std > 1e-10:
            cov = np.mean((f - f_mean) * (s - s_mean))
            out[i] = np.float32(cov / (f_std * s_std))
        else:
            out[i] = np.float32(0.0)


@numba.jit(nopython=True)
def _rolling_sync(fast: NDArray[np.float64], slow: NDArray[np.float64], window: int, out: NDArray[np.float32]) -> None:
    n = len(fast)
    for i in range(window, n):
        sync_count = 0.0
        for k in range(i - window, i):
            if (fast[k] >= 0 and slow[k] >= 0) or (fast[k] < 0 and slow[k] < 0):
                sync_count += 1.0
        out[i] = np.float32(sync_count / window)


@numba.jit(nopython=True)
def _rolling_desync(fast: NDArray[np.float64], slow: NDArray[np.float64], window: int, out: NDArray[np.float32]) -> None:
    n = len(fast)
    for i in range(window, n):
        desync_count = 0.0
        for k in range(i - window, i):
            if (fast[k] >= 0 and slow[k] < 0) or (fast[k] < 0 and slow[k] >= 0):
                desync_count += 1.0
        out[i] = np.float32(desync_count / window)


@numba.jit(nopython=True)
def _max_cross_corr(fast: NDArray[np.float64], slow: NDArray[np.float64], max_lag: int) -> float:
    n = len(fast)
    best_lag = 0.0
    best_corr = -1.0
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            f = fast[:lag]
            s = slow[-lag:]
        elif lag > 0:
            f = fast[lag:]
            s = slow[:-lag]
        else:
            f = fast
            s = slow
        if len(f) < 3:
            continue
        f_mean = np.mean(f)
        s_mean = np.mean(s)
        f_std = np.std(f)
        s_std = np.std(s)
        if f_std > 1e-10 and s_std > 1e-10:
            corr = np.mean((f - f_mean) * (s - s_mean)) / (f_std * s_std)
            if corr > best_corr:
                best_corr = corr
                best_lag = float(lag)
    return np.float32(best_lag)


@numba.jit(nopython=True)
def _propagation_velocity(fast: NDArray[np.float64], slow: NDArray[np.float64], max_lag: int, window: int, out: NDArray[np.float32]) -> None:
    n = len(fast)
    for i in range(window + max_lag, n):
        f_seg = fast[i - window : i]
        s_seg = slow[i - window : i]
        out[i] = _max_cross_corr(f_seg, s_seg, max_lag)


@numba.jit(nopython=True)
def _inversion_detector(coupling: NDArray[np.float32], threshold: float, out: NDArray[np.float32]) -> None:
    n = len(coupling)
    for i in range(n):
        out[i] = np.float32(1.0 if coupling[i] < threshold else 0.0)


class StateEntanglementResearch:
    TIMEFRAMES: ClassVar[tuple[str, ...]] = settings.research.entanglement_timeframes

    def compute_coupling_strength(
        self, fast_returns: NDArray[np.float64],
        slow_returns: NDArray[np.float64], window: int = 50
    ) -> NDArray[np.float32]:
        n = min(len(fast_returns), len(slow_returns))
        fast = fast_returns[:n]
        slow = slow_returns[:n]
        out = np.zeros(n, dtype=np.float32)
        _rolling_corr(fast, slow, window, out)
        return out

    def compute_synchronization(
        self, fast_returns: NDArray[np.float64],
        slow_returns: NDArray[np.float64], window: int = 50
    ) -> NDArray[np.float32]:
        n = min(len(fast_returns), len(slow_returns))
        out = np.zeros(n, dtype=np.float32)
        _rolling_sync(fast_returns[:n], slow_returns[:n], window, out)
        return out

    def compute_desynchronization(
        self, fast_returns: NDArray[np.float64],
        slow_returns: NDArray[np.float64], window: int = 50
    ) -> NDArray[np.float32]:
        n = min(len(fast_returns), len(slow_returns))
        out = np.zeros(n, dtype=np.float32)
        _rolling_desync(fast_returns[:n], slow_returns[:n], window, out)
        return out

    def compute_state_inversion(
        self, coupling: NDArray[np.float32], threshold: float = -0.5
    ) -> NDArray[np.float32]:
        out = np.zeros(len(coupling), dtype=np.float32)
        _inversion_detector(coupling, threshold, out)
        return out

    def compute_propagation_velocity(
        self, fast_returns: NDArray[np.float64],
        slow_returns: NDArray[np.float64], max_lag: int = 10
    ) -> NDArray[np.float32]:
        n = min(len(fast_returns), len(slow_returns))
        window = 50
        out = np.zeros(n, dtype=np.float32)
        _propagation_velocity(fast_returns[:n], slow_returns[:n], max_lag, window, out)
        return out

    def compute_all(self, timeframe_returns: dict[str, NDArray[np.float64]]) -> dict:
        result: dict[str, NDArray[np.float32]] = {}
        tfs = list(self.TIMEFRAMES)
        n = len(tfs)
        for i in range(n):
            for j in range(i + 1, n):
                tf1 = tfs[i]
                tf2 = tfs[j]
                r1 = timeframe_returns[tf1]
                r2 = timeframe_returns[tf2]
                key = f"{tf1}_{tf2}"
                result[f"{key}_coupling"] = self.compute_coupling_strength(r1, r2)
                result[f"{key}_synchronization"] = self.compute_synchronization(r1, r2)
                result[f"{key}_desynchronization"] = self.compute_desynchronization(r1, r2)
                result[f"{key}_inversion"] = self.compute_state_inversion(
                    result[f"{key}_coupling"]
                )
                result[f"{key}_propagation"] = self.compute_propagation_velocity(r1, r2)
        return result
