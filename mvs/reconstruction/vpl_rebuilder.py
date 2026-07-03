from __future__ import annotations

from typing import Dict
from collections import deque
import numpy as np


class VplRebuilder:
    __slots__ = ("symbol", "_last_state", "_stability", "_prices", "_volatilities")

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self._last_state = "NORMAL"
        self._stability = 1.0
        self._prices = deque(maxlen=100)
        self._volatilities = deque(maxlen=20)

    def on_tick(self, tick_id: int, symbol: str, ts_ns: int, mid: float, bid: float, ask: float) -> Dict:
        self._prices.append(mid)
        arr = np.array(self._prices, dtype=np.float64)

        if len(arr) >= 20:
            rets = np.diff(arr) / arr[:-1]
            vol = float(np.std(rets))
            self._volatilities.append(vol)
            recent_vol = np.mean(list(self._volatilities))
            vol_regime = recent_vol / max(np.median(list(self._volatilities)), 1e-9)

            if vol_regime > 1.5:
                state = "HIGH_VOL"
            elif vol_regime < 0.5:
                state = "LOW_VOL"
            else:
                state = "NORMAL"
        else:
            state = "BOOTSTRAP"

        if state == self._last_state:
            self._stability = min(1.0, self._stability + 0.02)
        else:
            self._stability = max(0.0, self._stability - 0.25)

        self._last_state = state
        return {"vpl_state": str(state), "vpl_stability": float(self._stability)}
