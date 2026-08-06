from __future__ import annotations

import numpy as np
import polars as pl

from mvs.utils.memory_pool import RingMemoryPool


PERCEPTION_DTYPE = np.dtype([
    ("tick_id", np.int64),
    ("symbol", object),
    ("ts_ns", np.int64),
    ("tpi", np.float64),
    ("tpi_sign", np.int8),
    ("tpi_decay", np.float64),
    ("tpi_curvature", np.float64),
    ("tpi_persistence", np.float64),
    ("tpi_propagation", np.float64),
    ("tpi_pressure", np.float64),
    ("entropy", np.float64),
    ("entropy_state", object),
    ("regime", object),
    ("regime_transition_prob", np.float64),
    ("vpl_state", object),
    ("vpl_stability", np.float64),
    ("observer_state", object),
    ("observer_confidence", np.float64),
    ("drift_score", np.float64),
    ("drift_flag", np.bool_),
    ("calibration_threshold", np.float64),
    ("calibration_bucket", object),
    ("age_ticks", np.int64),
    ("age_seconds", np.float64),
])


class PerceptionStatePlane:
    __slots__ = ("pool",)

    def __init__(self, size: int = 32768) -> None:
        self.pool = RingMemoryPool(size=size, dtype=PERCEPTION_DTYPE)

    def append_state(self, state: dict) -> int:
        row = np.zeros(1, dtype=PERCEPTION_DTYPE)
        for k in state:
            if k in PERCEPTION_DTYPE.names:
                row[k] = state[k]
        return self.pool.append(row[0])

    def latest(self) -> dict:
        return dict(zip(PERCEPTION_DTYPE.names, self.pool.latest()))

    def window(self, n: int) -> np.ndarray:
        return self.pool.window(n)

    def to_polars(self) -> pl.DataFrame:
        return pl.DataFrame(self.pool.view())

    def flush(self) -> np.ndarray:
        data = self.pool.view().copy()
        self.pool.clear()
        return data

    @property
    def count(self) -> int:
        return self.pool.count
