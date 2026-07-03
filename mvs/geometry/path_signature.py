from __future__ import annotations

import numpy as np


class PathSignatureEngine:
    __slots__ = ()

    def classify(self, tick_prices: np.ndarray, entry_price: float, direction: int) -> str:
        if len(tick_prices) < 3:
            return "NEUTRAL"
        excursion = (tick_prices - entry_price) * direction
        max_excursion = float(np.max(excursion))
        time_to_max = int(np.argmax(excursion))
        signs = np.sign(excursion)
        oscillation_count = int(np.sum(signs[1:] != signs[:-1]))
        excursion_range = np.max(excursion) - np.min(excursion)
        if time_to_max < len(excursion) * 0.1 and oscillation_count < 3:
            return "IMPULSE"
        if oscillation_count > 5 and max_excursion < excursion_range * 0.5:
            return "GRIND"
        if time_to_max < len(excursion) * 0.2 and excursion[-1] < max_excursion * 0.4:
            return "FAKEOUT"
        midpoint = len(excursion) // 2
        if np.any(np.sign(excursion[midpoint:]) != np.sign(excursion[:midpoint].mean())):
            return "REVERSAL"
        if max_excursion < 0:
            return "COLLAPSE"
        return "NEUTRAL"
