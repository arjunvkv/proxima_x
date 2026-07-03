import numpy as np
from typing import Optional


class FutureReturnEngine:
    def __init__(self, rates_provider):
        self._rates = rates_provider

    def compute(self, symbol: str, current_price: float,
                horizons: list[int] = None) -> dict:
        if horizons is None:
            horizons = [20, 50, 100]
        rates = self._rates(symbol)
        if rates is None or len(rates) < max(horizons) + 1:
            return {f"return_h{h}": None for h in horizons}

        closes = np.array([r["close"] for r in rates], dtype=np.float64)
        result = {}
        for h in horizons:
            if len(closes) > h:
                future_close = closes[-(h + 1)]
                ret = (future_close - current_price) / current_price if current_price > 0 else 0.0
            else:
                ret = None
            result[f"return_h{h}"] = float(ret) if ret is not None else None
            result[f"pp_h{h}"] = 1.0 if ret is not None and ret > 0 else 0.0 if ret is not None else None
        return result
