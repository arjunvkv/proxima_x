from collections import deque
from typing import Optional
import numpy as np


class CurrencyRangeBucket:
    def __init__(self, window: int = 20):
        self.window = window
        self.history: dict[str, deque] = {}

    def update(self, currency_strengths: dict[str, float]) -> None:
        for currency, strength in currency_strengths.items():
            if currency not in self.history:
                self.history[currency] = deque(maxlen=self.window)
            self.history[currency].append(strength)

    def get_state(self, currency: str) -> dict:
        values = list(self.history.get(currency, []))
        if len(values) < 5:
            return {
                "percentile": None,
                "median": None,
                "drift": None,
                "state": "INSUFFICIENT_DATA",
                "sample_size": len(values),
            }

        current = values[-1]
        median = float(np.median(values))

        # mid-rank percentile
        count_below = sum(1 for v in values if v < current)
        count_equal = sum(1 for v in values if v == current)
        percentile = (count_below + 0.5 * count_equal) / len(values) * 100

        # median drift: recent half vs older half, require >= 10 samples
        drift: Optional[float] = None
        if len(values) >= 10:
            mid = len(values) // 2
            recent_median = float(np.median(values[mid:]))
            old_median = float(np.median(values[:mid]))
            drift = recent_median - old_median

        # range state by percentile
        if percentile <= 10:
            state = "OUTSIDE_RANGE_LOW"
        elif percentile >= 90:
            state = "OUTSIDE_RANGE_HIGH"
        elif percentile < 25:
            state = "STRETCHED_LOW"
        elif percentile > 75:
            state = "STRETCHED_HIGH"
        else:
            state = "WITHIN_RANGE"

        return {
            "percentile": round(percentile, 1),
            "median": round(median, 8),
            "drift": round(drift, 8) if drift is not None else None,
            "state": state,
            "sample_size": len(values),
        }

    def get_all_states(self) -> dict[str, dict]:
        return {c: self.get_state(c) for c in self.history}

    def reset(self) -> None:
        self.history.clear()
