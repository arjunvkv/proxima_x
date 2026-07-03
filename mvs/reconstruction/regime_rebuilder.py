from __future__ import annotations

from typing import Dict
from collections import deque
import numpy as np


class RegimeRebuilder:
    __slots__ = ("window", "prices", "_last_regime", "_last_vol", "_last_entropy")

    def __init__(self, window: int = 64) -> None:
        self.window = window
        self.prices = deque(maxlen=window)
        self._last_regime = "CALM"
        self._last_vol = 0.0
        self._last_entropy = 1.0

    def _classify_regime(self, volatility: float, entropy: float, momentum: float) -> str:
        if volatility > np.percentile([self._last_vol, volatility], 80) and abs(momentum) > 0.001:
            return "TREND"
        elif entropy < 0.4 and volatility < self._last_vol * 0.5:
            return "CALM"
        elif volatility > self._last_vol * 2.0:
            return "SHOCK"
        else:
            return "RANGE"

    def on_tick(self, tick_id: int, symbol: str, ts_ns: int, mid: float, entropy_data: Dict, tpi_data: Dict) -> Dict:
        self.prices.append(mid)
        arr = np.array(self.prices, dtype=np.float64)

        if len(arr) > 2:
            volatility = float(np.std(arr))
            momentum = float(np.mean(np.diff(arr)))
            entropy = entropy_data.get("entropy", 1.0)
        else:
            volatility = 0.0
            momentum = 0.0
            entropy = 1.0

        regime = self._classify_regime(volatility, entropy, momentum)

        transition_prob = 0.0
        if regime != self._last_regime:
            transition_prob = min(1.0, abs(momentum) * 100)

        self._last_regime = regime
        self._last_vol = volatility
        self._last_entropy = entropy

        return {"regime": str(regime), "regime_transition_prob": float(transition_prob)}
