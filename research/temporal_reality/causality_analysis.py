"""
Adaptive Time Causality Analysis for Proxima X Reality Phase 4.

Provides lead/lag cross-correlation analysis between adaptive time
and observed signals using numba-accelerated computations.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numba import jit


class AdaptiveTimeCausality:
    """Compute lead/lag cross-correlation between adaptive time and signals."""

    def __init__(self, max_lag: int = 100) -> None:
        if max_lag < 1:
            raise ValueError("max_lag must be >= 1")
        self.max_lag: int = max_lag

    def compute(self, adaptive_time: np.ndarray, signal: np.ndarray) -> dict[str, Any]:
        """Compute lead/lag cross-correlation between adaptive_time and a signal.

        Parameters
        ----------
        adaptive_time : np.ndarray
            1-D array representing the adaptive (warped) time axis.
        signal : np.ndarray
            1-D observed signal array, same length as *adaptive_time*.

        Returns
        -------
        dict
            {
                "lags": list[int],
                "correlations": list[float],
                "peak_lag": int,
                "peak_corr": float,
                "lead_or_follow": str
            }
        """
        adaptive_time = self._as_float64_1d(adaptive_time)
        signal = self._as_float64_1d(signal)

        if adaptive_time.shape != signal.shape:
            raise ValueError(
                f"adaptive_time and signal must have the same shape; "
                f"got {adaptive_time.shape} vs {signal.shape}"
            )
        if adaptive_time.size < self.max_lag * 2 + 1:
            raise ValueError(
                f"Array length ({adaptive_time.size}) is too small for "
                f"max_lag={self.max_lag}; need at least {self.max_lag * 2 + 1}"
            )

        corr = self._cross_correlate(adaptive_time, signal, self.max_lag)
        lags = np.arange(-self.max_lag, self.max_lag + 1, dtype=np.intp)

        peak_idx = int(np.argmax(np.abs(corr)))
        peak_lag = int(lags[peak_idx])
        peak_corr = float(corr[peak_idx])

        if abs(peak_lag) <= 1:
            lead_or_follow = "synchronous"
        elif peak_lag < 0:
            lead_or_follow = "adaptive_time_leads"
        else:
            lead_or_follow = "adaptive_time_follows"

        return {
            "lags": lags.tolist(),
            "correlations": corr.tolist(),
            "peak_lag": peak_lag,
            "peak_corr": peak_corr,
            "lead_or_follow": lead_or_follow,
        }

    def compute_mutation_regime_causality(
        self,
        adaptive_time: np.ndarray,
        state_mutation_rate: np.ndarray,
        regime_change_events: np.ndarray,
    ) -> dict[str, Any]:
        """Combined causality analysis against both mutation rate and regime events.

        Parameters
        ----------
        adaptive_time : np.ndarray
            1-D adaptive time axis.
        state_mutation_rate : np.ndarray
            1-D mutation-rate signal, same length as *adaptive_time*.
        regime_change_events : np.ndarray
            1-D binary or float array marking regime events (same length).

        Returns
        -------
        dict
            Multi-signal causality dictionary with keys
            "mutation_rate_causality" and "regime_events_causality",
            each containing the output of :meth:`compute`.
        """
        mutation_causality = self.compute(adaptive_time, state_mutation_rate)
        regime_causality = self.compute(adaptive_time, regime_change_events)

        return {
            "mutation_rate_causality": mutation_causality,
            "regime_events_causality": regime_causality,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _as_float64_1d(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float64)
        if arr.ndim != 1:
            raise ValueError(f"Expected 1-D array, got {arr.ndim} dimensions")
        return arr

    @staticmethod
    @jit(nopython=True, cache=True)
    def _cross_correlate(x: np.ndarray, y: np.ndarray, max_lag: int) -> np.ndarray:
        """Compute cross-correlation for all lags ``-max_lag .. +max_lag``.

        Both *x* and *y* are z-score normalised internally before computing
        Pearson correlation at each lag on the overlapping portion.

        Parameters
        ----------
        x : np.ndarray
            First 1-D signal (float64).
        y : np.ndarray
            Second 1-D signal (float64).
        max_lag : int
            Maximum lag to evaluate.

        Returns
        -------
        np.ndarray
            1-D array of length ``2 * max_lag + 1`` containing the
            correlation coefficient at each lag (index 0 corresponds to
            lag ``-max_lag``).
        """
        n = len(x)
        result = np.empty(2 * max_lag + 1, dtype=np.float64)

        # Z-score normalise both arrays in-place copies
        x_mean = np.mean(x)
        x_std = np.std(x)
        y_mean = np.mean(y)
        y_std = np.std(y)

        for k in range(-max_lag, max_lag + 1):
            result[k + max_lag] = _pearson_at_lag(
                x, y, k, x_mean, x_std, y_mean, y_std
            )

        return result


@jit(nopython=True, cache=True)
def _pearson_at_lag(
    x: np.ndarray,
    y: np.ndarray,
    lag: int,
    x_mean: float,
    x_std: float,
    y_mean: float,
    y_std: float,
) -> float:
    """Pearson correlation between *x* and *y* at a given *lag*.

    Handles edge-truncation so that only the overlapping portion of the
    two series is included.
    """
    n = len(x)

    if lag >= 0:
        start_x, end_x = lag, n
        start_y, end_y = 0, n - lag
    else:
        start_x, end_x = 0, n + lag
        start_y, end_y = -lag, n

    length = end_x - start_x
    if length < 2:
        return 0.0

    # Overlapping slices
    x_seg = x[start_x:end_x]
    y_seg = y[start_y:end_y]

    # Z-score the overlap
    x_seg_mean = np.mean(x_seg)
    x_seg_std = np.std(x_seg)
    y_seg_mean = np.mean(y_seg)
    y_seg_std = np.std(y_seg)

    # Protect against zero std
    if x_seg_std == 0.0 or y_seg_std == 0.0:
        return 0.0

    # Pearson r
    # r = mean( (x - mu_x)/sigma_x * (y - mu_y)/sigma_y )
    s = 0.0
    for i in range(length):
        s += (x_seg[i] - x_seg_mean) / x_seg_std * (y_seg[i] - y_seg_mean) / y_seg_std

    r = s / length
    # Clamp to [-1, 1] for floating-point safety
    return max(-1.0, min(1.0, r))
