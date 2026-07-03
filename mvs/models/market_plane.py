from __future__ import annotations

import numpy as np
import polars as pl

from mvs.utils.memory_pool import RingMemoryPool


MARKET_DTYPE = np.dtype([
    ("tick_id", np.int64),
    ("symbol", "U10"),
    ("ts_ns", np.int64),
    ("bid", np.float64),
    ("ask", np.float64),
    ("mid", np.float64),
    ("spread", np.float64),
    ("delta", np.float64),
    ("velocity", np.float64),
    ("acceleration", np.float64),
    ("jerk", np.float64),
    ("entropy", np.float64),
    ("d_entropy", np.float64),
    ("compression_ratio", np.float64),
    ("burst_density", np.float64),
    ("regime_hint", object),
    ("liquidity_proxy", np.float64),
    ("pressure_proxy", np.float64),
    ("vol_cluster", np.float64),
])


class MarketRealityPlane:
    __slots__ = ("pool",)

    def __init__(self, size: int = 32768) -> None:
        self.pool = RingMemoryPool(size=size, dtype=MARKET_DTYPE)

    def append_tick(self, tick_data: dict) -> int:
        row = np.zeros(1, dtype=MARKET_DTYPE)
        for k in tick_data:
            row[k] = tick_data[k]
        return self.pool.append(row[0])

    def latest(self) -> dict:
        return dict(zip(MARKET_DTYPE.names, self.pool.latest()))

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
