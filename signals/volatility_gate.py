"""Volatility gate.
Prevents trading during dead/noisy states.
Requires: vol_min <= realized_vol <= vol_max
"""
from collections import deque
import numpy as np


class VolatilityGate:
    def __init__(self, window=20, vol_min_z=0.15, vol_max_z=2.0, sym_stats=None):
        self.window = window
        self.vol_min_z = vol_min_z
        self.vol_max_z = vol_max_z
        self.sym_stats = sym_stats or {}
        self._buffers = {}  # sym -> deque of prices

    def update(self, sym, price):
        if sym not in self._buffers:
            self._buffers[sym] = deque(maxlen=self.window + 1)
        self._buffers[sym].append(price)

    def realized_vol(self, sym):
        buf = self._buffers.get(sym)
        if buf is None or len(buf) < 2:
            return None
        prices = list(buf)
        returns = [abs((prices[i+1] - prices[i]) / prices[i]) for i in range(len(prices)-1)]
        return np.mean(returns) if returns else None

    def tradable(self, sym):
        vol = self.realized_vol(sym)
        if vol is None:
            return True  # unknown state = tradable
        sym_vol = self.sym_stats.get(sym, {}).get("sigma", 0.001)
        vol_z = vol / sym_vol if sym_vol > 0 else 1.0
        return self.vol_min_z <= vol_z <= self.vol_max_z

    def reset(self):
        self._buffers.clear()
