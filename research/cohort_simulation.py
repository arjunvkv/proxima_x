from __future__ import annotations

from typing import ClassVar

import numpy as np
import numba
from numpy.typing import NDArray

from core.event_engine import EventEngine, Event, EventType


COHORT_CENTERS = np.array([0.07, 0.21, 0.36, 0.50, 0.64, 0.79, 0.93], dtype=np.float64)
COHORT_BANDWIDTH = 0.15


@numba.jit(nopython=True, parallel=False)
def _percentile_rank(window: NDArray[np.float64], value: float) -> float:
    n = len(window)
    if n == 0:
        return 0.5
    count = 0.0
    for i in range(n):
        if window[i] < value:
            count += 1.0
        elif window[i] == value:
            count += 0.5
    return count / n


@numba.jit(nopython=True)
def _estimate_activity(volume: NDArray[np.float64], window: int, out: NDArray[np.float32]) -> None:
    n = len(volume)
    n_cohorts = 7
    for i in range(window, n):
        w = volume[i - window : i]
        pct = _percentile_rank(w, volume[i])
        for c in range(n_cohorts):
            z = (pct - COHORT_CENTERS[c]) / COHORT_BANDWIDTH
            out[i, c] = np.float32(np.exp(-0.5 * z * z))


@numba.jit(nopython=True)
def _rolling_stability(data: NDArray[np.float32], window: int, out: NDArray[np.float32]) -> None:
    n = data.shape[0]
    n_cohorts = data.shape[1]
    for i in range(window, n):
        for c in range(n_cohorts):
            segment = data[i - window : i, c]
            std = np.std(segment)
            out[i, c] = np.float32(1.0 / (1.0 + std))


@numba.jit(nopython=True)
def _cohort_stress(returns: NDArray[np.float64], activity: NDArray[np.float32], window: int, out: NDArray[np.float32]) -> None:
    n = len(returns)
    n_cohorts = activity.shape[1]
    for i in range(window, n):
        abs_ret_seg = np.abs(returns[i - window : i])
        for c in range(n_cohorts):
            act_seg = activity[i - window : i, c]
            total = 0.0
            for k in range(window):
                total += act_seg[k] * abs_ret_seg[k]
            out[i, c] = np.float32(total / window)


@numba.jit(nopython=True)
def _rolling_alignment(price: NDArray[np.float64], activity: NDArray[np.float32], window: int, out: NDArray[np.float32]) -> None:
    n = len(price)
    n_cohorts = activity.shape[1]
    price_diff = np.diff(price)
    for i in range(window, n):
        align_sum = 0.0
        count = 0
        for k in range(i - window, i):
            if k < 1:
                continue
            p_sign = 1.0 if price_diff[k - 1] >= 0 else -1.0
            for c in range(n_cohorts):
                a_sign = 1.0 if activity[k, c] >= activity[k - 1, c] else -1.0
                align_sum += p_sign * a_sign
                count += 1
        out[i] = np.float32(align_sum / max(count, 1))


@numba.jit(nopython=True)
def _rolling_conflict(activity: NDArray[np.float32], window: int, out: NDArray[np.float32]) -> None:
    n = activity.shape[0]
    n_cohorts = activity.shape[1]
    for i in range(window, n):
        var_sum = 0.0
        for k in range(i - window, i):
            mean = 0.0
            for c in range(n_cohorts):
                mean += activity[k, c]
            mean /= n_cohorts
            v = 0.0
            for c in range(n_cohorts):
                diff = activity[k, c] - mean
                v += diff * diff
            var_sum += v / n_cohorts
        out[i] = np.float32(var_sum / window)


class CohortSimulationResearch:
    COHORT_NAMES: ClassVar[list[str]] = [
        "Scalpers", "Intraday", "Swing", "Funds",
        "Institutions", "Hedgers", "Systematic"
    ]

    def __init__(self) -> None:
        self._n_cohorts = len(self.COHORT_NAMES)

    def estimate_cohorts_from_volume(
        self, volume: NDArray[np.float64], price: NDArray[np.float64],
        window: int = 50
    ) -> NDArray[np.float32]:
        n = len(volume)
        out = np.zeros((n, self._n_cohorts), dtype=np.float32)
        _estimate_activity(volume, window, out)
        return out

    def compute_cohort_confidence(
        self, cohort_activity: NDArray[np.float32], window: int = 20
    ) -> NDArray[np.float32]:
        n = cohort_activity.shape[0]
        out = np.zeros((n, self._n_cohorts), dtype=np.float32)
        _rolling_stability(cohort_activity, window, out)
        return out

    def compute_cohort_stress(
        self, returns: NDArray[np.float64],
        cohort_activity: NDArray[np.float32], window: int = 20
    ) -> NDArray[np.float32]:
        n = len(returns)
        out = np.zeros((n, self._n_cohorts), dtype=np.float32)
        _cohort_stress(returns, cohort_activity, window, out)
        return out

    def compute_cohort_alignment(
        self, price: NDArray[np.float64],
        cohort_activity: NDArray[np.float32], window: int = 20
    ) -> NDArray[np.float32]:
        n = len(price)
        out = np.zeros(n, dtype=np.float32)
        _rolling_alignment(price, cohort_activity, window, out)
        return out

    def compute_cohort_conflict(
        self, cohort_activity: NDArray[np.float32], window: int = 20
    ) -> NDArray[np.float32]:
        n = cohort_activity.shape[0]
        out = np.zeros(n, dtype=np.float32)
        _rolling_conflict(cohort_activity, window, out)
        return out

    def compute_all(
        self, volume: NDArray[np.float64], price: NDArray[np.float64],
        returns: NDArray[np.float64]
    ) -> dict:
        activity = self.estimate_cohorts_from_volume(volume, price)
        confidence = self.compute_cohort_confidence(activity)
        stress = self.compute_cohort_stress(returns, activity)
        alignment = self.compute_cohort_alignment(price, activity)
        conflict = self.compute_cohort_conflict(activity)
        return {
            "cohort_activity": activity,
            "cohort_confidence": confidence,
            "cohort_stress": stress,
            "cohort_alignment": alignment,
            "cohort_conflict": conflict,
        }

    def emit_events(
        self, timestamps: list[int], alignment: NDArray[np.float32],
        conflict: NDArray[np.float32], event_engine: EventEngine,
        align_threshold: float = 0.8, conflict_threshold: float = 0.8
    ) -> None:
        events: list[Event] = []
        n = min(len(timestamps), len(alignment), len(conflict))
        for i in range(n):
            if alignment[i] >= align_threshold:
                events.append(Event(
                    event_type=EventType.COHORT_ALIGNMENT,
                    timestamp=timestamps[i],
                    data={"alignment": float(alignment[i])},
                    source="cohort_simulation",
                    confidence=float(min(alignment[i], 1.0)),
                ))
            if conflict[i] >= conflict_threshold:
                events.append(Event(
                    event_type=EventType.COHORT_CONFLICT,
                    timestamp=timestamps[i],
                    data={"conflict": float(conflict[i])},
                    source="cohort_simulation",
                    confidence=float(min(conflict[i] / 2.0, 1.0)),
                ))
        event_engine.emit_batch(events)
