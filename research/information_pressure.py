from __future__ import annotations

from typing import Optional

import numpy as np
import numba
from numpy.typing import NDArray

from config.settings import settings
from core.event_engine import EventEngine, Event, EventType


@numba.jit(nopython=True, cache=True)
def _cum_dev_weighted(returns: NDArray[np.float64], volume: NDArray[np.float64], window: int) -> NDArray[np.float32]:
    n = min(len(returns), len(volume))
    result = np.zeros(n, dtype=np.float32)
    for i in range(window - 1, n):
        r_seg = returns[i - window + 1 : i + 1]
        v_seg = volume[i - window + 1 : i + 1]
        v_sum = np.sum(v_seg)
        if v_sum < 1e-10:
            continue
        w = v_seg / v_sum
        dev = 0.0
        for j in range(window):
            dev += r_seg[j] * w[j]
        result[i] = dev
    return result


@numba.jit(nopython=True, cache=True)
def _detect_release(returns: NDArray[np.float64], pressure: NDArray[np.float32], threshold: float) -> NDArray[np.float32]:
    n = min(len(returns), len(pressure))
    result = np.zeros(n, dtype=np.float32)
    for i in range(1, n):
        ret = abs(returns[i])
        if pressure[i - 1] > 1.0 and ret > threshold * np.std(returns[: i + 1]) if i > 10 else 0:
            result[i] = ret
    return result


@numba.jit(nopython=True, cache=True)
def _second_derivative(x: NDArray[np.float32], window: int) -> NDArray[np.float32]:
    n = len(x)
    result = np.zeros(n, dtype=np.float32)
    for i in range(window, n):
        m1 = np.mean(x[i - window : i])
        m2 = np.mean(x[i - 2 * window : i - window]) if i >= 2 * window else m1
        result[i] = m1 - m2
    return result


@numba.jit(nopython=True, cache=True)
def _rolling_variance(x: NDArray[np.float32], window: int) -> NDArray[np.float32]:
    n = len(x)
    result = np.zeros(n, dtype=np.float32)
    for i in range(window - 1, n):
        seg = x[i - window + 1 : i + 1]
        m = np.mean(seg)
        v = 0.0
        for j in range(window):
            d = seg[j] - m
            v += d * d
        result[i] = v / window
    return result


class InformationPressureResearch:
    def __init__(self, window: int = settings.research.pressure_window):
        self.window = window

    def compute_pressure_build(self, returns: NDArray[np.float64], volume: NDArray[np.float64], window: int | None = None) -> NDArray[np.float32]:
        w = window if window is not None else self.window
        return _cum_dev_weighted(returns.astype(np.float64), volume.astype(np.float64), min(w, len(returns), len(volume)))

    def compute_pressure_release(self, returns: NDArray[np.float64], pressure_build: NDArray[np.float32], threshold: float = 2.0) -> NDArray[np.float32]:
        return _detect_release(returns.astype(np.float64), pressure_build, threshold)

    def compute_pressure_acceleration(self, pressure_build: NDArray[np.float32], window: int = 5) -> NDArray[np.float32]:
        return _second_derivative(pressure_build, min(window, len(pressure_build) // 3)) if len(pressure_build) > 0 else np.array([], dtype=np.float32)

    def compute_pressure_instability(self, pressure_build: NDArray[np.float32], window: int = 20) -> NDArray[np.float32]:
        return _rolling_variance(pressure_build, min(window, len(pressure_build)))

    def compute_all(self, returns: NDArray[np.float64], volume: NDArray[np.float64]) -> dict:
        build = self.compute_pressure_build(returns, volume)
        release = self.compute_pressure_release(returns, build)
        accel = self.compute_pressure_acceleration(build)
        inst = self.compute_pressure_instability(build)
        return {
            "pressure_build": build,
            "pressure_release": release,
            "pressure_acceleration": accel,
            "pressure_instability": inst,
        }

    def emit_events(self, timestamps: list[int], pressure_build: NDArray[np.float32], release: NDArray[np.float32], event_engine: EventEngine) -> None:
        for i in range(min(len(timestamps), len(pressure_build), len(release))):
            if pressure_build[i] > 1.5:
                event_engine.emit(Event(
                    event_type=EventType.PRESSURE_BUILD,
                    timestamp=timestamps[i],
                    data={"index": i, "pressure": float(pressure_build[i])},
                    source="information_pressure",
                    confidence=min(1.0, float(pressure_build[i]) / 3.0),
                ))
            if release[i] > 0.0:
                event_engine.emit(Event(
                    event_type=EventType.PRESSURE_RELEASE,
                    timestamp=timestamps[i],
                    data={"index": i, "release_magnitude": float(release[i])},
                    source="information_pressure",
                    confidence=min(1.0, float(release[i]) * 2.0),
                ))
