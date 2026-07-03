from __future__ import annotations

from typing import Dict
import numpy as np


class ExcursionTopologyEngine:
    __slots__ = ()

    def compute(self, tick_prices: np.ndarray, entry_price: float, direction: int) -> Dict:
        if len(tick_prices) < 3:
            return {"max_excursion": 0.0, "time_to_max_ticks": 0, "oscillation_count": 0, "convexity": 0.0, "conviction_half_life": 0}
        excursion = (tick_prices - entry_price) * direction
        max_excursion = float(np.max(excursion))
        time_to_max_ticks = int(np.argmax(excursion))
        oscillation_count = int(np.sum(np.sign(excursion[1:]) != np.sign(excursion[:-1])))
        convexity = float(np.gradient(np.gradient(excursion)).mean())
        half_life = len(excursion)
        peak_idx = time_to_max_ticks
        threshold = max_excursion * 0.5
        for i in range(peak_idx, len(excursion)):
            if excursion[i] <= threshold:
                half_life = i - peak_idx
                break
        return {"max_excursion": max_excursion, "time_to_max_ticks": time_to_max_ticks, "oscillation_count": oscillation_count, "convexity": convexity, "conviction_half_life": int(half_life)}
