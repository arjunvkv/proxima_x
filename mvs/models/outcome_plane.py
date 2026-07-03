from __future__ import annotations

import numpy as np
import polars as pl

from mvs.utils.memory_pool import RingMemoryPool


OUTCOME_DTYPE = np.dtype([
    ("trade_id", np.int64),
    ("symbol", object),
    ("entry_ts_ns", np.int64),
    ("entry_price", np.float64),
    ("exit_ts_ns", np.int64),
    ("actual_exit_price", np.float64),
    ("model_exit_price", np.float64),
    ("shadow_exit_price", np.float64),
    ("mfe", np.float64),
    ("mae", np.float64),
    ("h20", np.float64),
    ("h50", np.float64),
    ("h100", np.float64),
    ("h250", np.float64),
    ("h500", np.float64),
    ("path_signature", object),
    ("continuation_alpha", np.float64),
    ("optimal_exit_price", np.float64),
    ("optimal_hold_ticks", np.int64),
])


class OutcomeStatePlane:
    __slots__ = ("pool",)

    def __init__(self, size: int = 16384) -> None:
        self.pool = RingMemoryPool(size=size, dtype=OUTCOME_DTYPE)

    def append_outcome(self, outcome: dict) -> int:
        row = np.zeros(1, dtype=OUTCOME_DTYPE)
        for k in outcome:
            row[k] = outcome[k]
        return self.pool.append(row[0])

    def latest(self) -> dict:
        return dict(zip(OUTCOME_DTYPE.names, self.pool.latest()))

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
