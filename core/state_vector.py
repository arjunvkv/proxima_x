from __future__ import annotations

from typing import NamedTuple

import numpy as np
import numba
from numpy.typing import NDArray


class StateVector(NamedTuple):
    memory_score: float
    dna_cluster: float
    pressure_level: float
    liquidity_mass: float
    cohort_alignment: float
    tension_score: float
    entropy_score: float

    @property
    def as_array(self) -> NDArray[np.float32]:
        return np.array(self, dtype=np.float32)

    @classmethod
    def from_array(cls, arr: NDArray[np.float32]) -> StateVector:
        return cls(*arr.tolist())

    @classmethod
    def zero(cls) -> StateVector:
        return cls(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@numba.jit(nopython=True, cache=True)
def normalize_state(state: NDArray[np.float32]) -> NDArray[np.float32]:
    norm = np.sqrt(np.sum(state ** 2))
    if norm > 0:
        return state / norm
    return state


@numba.jit(nopython=True, cache=True)
def state_distance(s1: NDArray[np.float32], s2: NDArray[np.float32]) -> float:
    return float(np.sqrt(np.sum((s1 - s2) ** 2)))


@numba.jit(nopython=True, cache=True)
def state_similarity(s1: NDArray[np.float32], s2: NDArray[np.float32]) -> float:
    dot = float(np.sum(s1 * s2))
    n1 = float(np.sqrt(np.sum(s1 ** 2)))
    n2 = float(np.sqrt(np.sum(s2 ** 2)))
    if n1 * n2 == 0:
        return 0.0
    return dot / (n1 * n2)
