"""Queue position proxy model — estimates fill probability from market conditions."""
import numpy as np
from typing import Optional


class QueueModel:
    def __init__(self, seed: Optional[int] = None):
        self._rng = np.random.default_rng(seed)

    def queue_probability(self, spread: float, volatility: float = 0.0,
                          tick_velocity: float = 1.0,
                          aggressiveness: float = 0.5) -> float:
        wide_spread = 1.0 - min(1.0, spread * 200.0)
        fast_tape = min(1.0, tick_velocity / 10.0)
        vol_penalty = max(0.5, 1.0 - volatility * 5.0)
        agg_bonus = 0.5 + aggressiveness * 0.5
        prob = wide_spread * 0.3 + fast_tape * 0.3 + vol_penalty * 0.2 + agg_bonus * 0.2
        return max(0.05, min(0.95, float(prob)))
