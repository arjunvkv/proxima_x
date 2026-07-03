"""Lognormal latency model for order execution simulation."""
import numpy as np
from typing import Optional

_SESSION_MULTIPLIERS = {
    "ASIA": 1.0,
    "LONDON": 1.3,
    "OVERLAP": 0.8,
    "NY": 1.2,
    "DEAD": 2.0,
}


class LatencyModel:
    def __init__(self, seed: Optional[int] = None):
        self._rng = np.random.default_rng(seed)

    def sample_ms(self, spread: float = 0.0, volatility: float = 0.0,
                  session: str = "ASIA") -> float:
        raw = float(self._rng.lognormal(mean=-3.2, sigma=0.45)) * 1000.0
        multiplier = _SESSION_MULTIPLIERS.get(session.upper(), 1.0)
        if volatility > 0.01:
            multiplier *= min(2.0, 1.0 + volatility * 10.0)
        return max(8.0, min(400.0, raw * multiplier))
