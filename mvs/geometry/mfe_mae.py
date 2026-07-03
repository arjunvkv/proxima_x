from __future__ import annotations

from typing import Dict
import numpy as np


class MfeMaeCalculator:
    __slots__ = ()

    def compute(self, entry_price: float, exit_price: float, tick_prices: np.ndarray, direction: int) -> Dict[str, float]:
        if len(tick_prices) == 0:
            return {"mfe": 0.0, "mae": 0.0, "h20": 0.0, "h50": 0.0, "h100": 0.0, "h250": 0.0, "h500": 0.0}
        if direction == 1:
            mfe = float(np.max(tick_prices) - entry_price)
            mae = float(entry_price - np.min(tick_prices))
        else:
            mfe = float(entry_price - np.min(tick_prices))
            mae = float(np.max(tick_prices) - entry_price)

        def horizon(n: int) -> float:
            idx = min(n, len(tick_prices) - 1)
            px = tick_prices[idx]
            return abs(px - entry_price)

        return {"mfe": abs(mfe), "mae": abs(mae), "h20": horizon(20), "h50": horizon(50), "h100": horizon(100), "h250": horizon(250), "h500": horizon(500)}
