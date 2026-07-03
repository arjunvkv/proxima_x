from __future__ import annotations

from collections import deque
from typing import Dict

import numpy as np

from layer7.tick_thermodynamics import TickThermodynamicsEngine
from layer7.meta_state import MetaStateFusionEngine
from layer7.tpi_outcomes import TPIPersistenceTracker, TPICurvatureTracker


class TpiRebuilder:
    __slots__ = (
        "persistence_tracker", "curvature_tracker", "thermo_engine", "fusion_engine",
        "window", "ticks", "_ema", "_alpha",
        "_last_tpi", "_prev_tpi",
    )

    def __init__(self, window: int = 64) -> None:
        self.persistence_tracker = TPIPersistenceTracker()
        self.curvature_tracker = TPICurvatureTracker()
        self.thermo_engine = TickThermodynamicsEngine()
        self.fusion_engine = MetaStateFusionEngine()
        self.window = window
        self.ticks = deque(maxlen=window)
        self._alpha = 2.0 / (window + 1.0)
        self._ema = 0.0
        self._last_tpi = 0.0
        self._prev_tpi = 0.0

    def _persistence(self, tpi_series: np.ndarray) -> float:
        if len(tpi_series) == 0:
            return 0.0
        signs = np.sign(tpi_series)
        current = signs[-1]
        if current == 0:
            return 0.0
        count = np.sum(signs == current)
        return float(count / len(signs))

    def on_tick(self, tick_id: int, symbol: str, ts_ns: int, mid: float, bid: float, ask: float) -> Dict[str, float]:
        self.ticks.append(mid)
        tick_arr = np.array(self.ticks, dtype=np.float64)
        if len(tick_arr) < 3:
            tpi = 0.0
        else:
            delta = tick_arr[-1] - tick_arr[-2]
            tpi = float(delta)
        tpi_sign = int(np.sign(tpi))
        self._ema = self._alpha * tpi + (1.0 - self._alpha) * self._ema
        curvature_data = self.curvature_tracker.update(symbol, tpi)
        curvature = curvature_data.get("d2TPI", 0.0) or 0.0
        persistence_data = self.persistence_tracker.update(symbol, tpi, tpi_sign)
        persistence = persistence_data.get("normalized_persistence", 0.0) or 0.0
        propagation = 0.0
        velocity = (tick_arr[-1] - tick_arr[-2]) if len(tick_arr) >= 2 else 0.0
        pressure = velocity * persistence
        self._prev_tpi = self._last_tpi
        self._last_tpi = tpi
        return {
            "tpi": float(tpi),
            "tpi_sign": np.int8(tpi_sign),
            "tpi_decay": float(self._ema),
            "tpi_curvature": float(curvature),
            "tpi_persistence": float(persistence),
            "tpi_propagation": float(propagation),
            "tpi_pressure": float(pressure),
        }
