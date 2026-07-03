from __future__ import annotations

import numpy as np
import polars as pl

from mvs.utils.memory_pool import RingMemoryPool


ACTION_DTYPE = np.dtype([
    ("action_id", np.int64),
    ("trade_id", np.int64),
    ("ticket", np.int64),
    ("symbol", object),
    ("ts_ns", np.int64),
    ("action_type", object),
    ("direction", np.int8),
    ("entry_price", np.float64),
    ("sl", np.float64),
    ("tp", np.float64),
    ("size", np.float64),
    ("regime", object),
    ("signal_strength", np.float64),
    ("rf_prob", np.float64),
    ("reason_code", object),
    ("forced_close", np.bool_),
    ("manual_intervention", np.bool_),
])


class ActionStatePlane:
    __slots__ = ("pool",)

    def __init__(self, size: int = 16384) -> None:
        self.pool = RingMemoryPool(size=size, dtype=ACTION_DTYPE)

    def append_action(self, action: dict) -> int:
        row = np.zeros(1, dtype=ACTION_DTYPE)
        for k in action:
            row[k] = action[k]
        return self.pool.append(row[0])

    def latest(self) -> dict:
        return dict(zip(ACTION_DTYPE.names, self.pool.latest()))

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
