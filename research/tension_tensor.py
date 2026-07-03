from __future__ import annotations

import numpy as np
import numba
from numpy.typing import NDArray

from core.event_engine import EventEngine, Event, EventType


@numba.jit(nopython=True)
def _normalize(arr: NDArray[np.float32]) -> NDArray[np.float32]:
    mn = np.min(arr)
    mx = np.max(arr)
    if mx - mn > 1e-10:
        return (arr - mn) / (mx - mn)
    return np.zeros_like(arr, dtype=np.float32)


@numba.jit(nopython=True)
def _tension_score(
    memory: NDArray[np.float32], pressure: NDArray[np.float32],
    liquidity: NDArray[np.float32], cohort: NDArray[np.float32],
    volatility: NDArray[np.float32], state_alignment: NDArray[np.float32],
    out: NDArray[np.float32]
) -> None:
    n = out.shape[0]
    weights = np.array([1.0, 1.5, 1.5, 1.2, 2.0, 1.8], dtype=np.float32)
    for i in range(n):
        val = (
            weights[0] * memory[i] +
            weights[1] * pressure[i] +
            weights[2] * liquidity[i] +
            weights[3] * cohort[i] +
            weights[4] * volatility[i] +
            weights[5] * state_alignment[i]
        )
        out[i] = val / np.sum(weights)


@numba.jit(nopython=True)
def _gradient(tension: NDArray[np.float32], out: NDArray[np.float32]) -> None:
    n = len(tension)
    for i in range(1, n):
        out[i] = tension[i] - tension[i - 1]
    out[0] = tension[0]


@numba.jit(nopython=True)
def _instability_prob(
    tension: NDArray[np.float32], gradient: NDArray[np.float32],
    window: int, out: NDArray[np.float32]
) -> None:
    n = len(tension)
    for i in range(window, n):
        t_mean = 0.0
        g_mean = 0.0
        for k in range(i - window, i):
            t_mean += tension[k]
            g_mean += gradient[k]
        t_mean /= window
        g_mean /= window
        z = t_mean + g_mean - 1.0
        prob = 1.0 / (1.0 + np.exp(-z))
        out[i] = np.float32(prob)


class TensionTensorResearch:

    def compute_tension_score(
        self, memory: NDArray[np.float32], pressure: NDArray[np.float32],
        liquidity: NDArray[np.float32], cohort: NDArray[np.float32],
        volatility: NDArray[np.float32], state_alignment: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        n = min(len(memory), len(pressure), len(liquidity), len(cohort), len(volatility), len(state_alignment))
        mem_n = _normalize(memory[:n].astype(np.float32))
        pre_n = _normalize(pressure[:n].astype(np.float32))
        liq_n = _normalize(liquidity[:n].astype(np.float32))
        coh_n = _normalize(cohort[:n].astype(np.float32))
        vol_n = _normalize(volatility[:n].astype(np.float32))
        align_n = _normalize(state_alignment[:n].astype(np.float32))
        out = np.zeros(n, dtype=np.float32)
        _tension_score(mem_n, pre_n, liq_n, coh_n, vol_n, align_n, out)
        return out

    def compute_tension_gradient(self, tension: NDArray[np.float32]) -> NDArray[np.float32]:
        out = np.zeros(len(tension), dtype=np.float32)
        _gradient(tension, out)
        return out

    def compute_instability_probability(
        self, tension: NDArray[np.float32],
        gradient: NDArray[np.float32], window: int = 50
    ) -> NDArray[np.float32]:
        n = len(tension)
        out = np.zeros(n, dtype=np.float32)
        _instability_prob(tension, gradient, window, out)
        return out

    def compute_all(self, inputs: dict) -> dict:
        score = self.compute_tension_score(
            inputs["memory"], inputs["pressure"], inputs["liquidity"],
            inputs["cohort"], inputs["volatility"], inputs["state_alignment"],
        )
        grad = self.compute_tension_gradient(score)
        prob = self.compute_instability_probability(score, grad)
        return {
            "tension_score": score,
            "tension_gradient": grad,
            "instability_probability": prob,
        }

    def emit_events(
        self, timestamps: list[int], tension: NDArray[np.float32],
        event_engine: EventEngine, threshold: float = 2.0
    ) -> None:
        events: list[Event] = []
        n = min(len(timestamps), len(tension))
        for i in range(n):
            if tension[i] >= threshold:
                events.append(Event(
                    event_type=EventType.TENSION_SPIKE,
                    timestamp=timestamps[i],
                    data={"tension": float(tension[i])},
                    source="tension_tensor",
                    confidence=float(min(tension[i] / 5.0, 1.0)),
                ))
        event_engine.emit_batch(events)
