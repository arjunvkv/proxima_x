from __future__ import annotations

import math
from typing import Any


class MarketRealityAnchor:
    def __init__(self) -> None:
        self._price_history: dict[str, list[float]] = {}
        self._spread_history: dict[str, list[float]] = {}
        self._atr_buffer: dict[str, list[float]] = {}

    def update(self, symbol: str, price: float, spread: float) -> None:
        if symbol not in self._price_history:
            self._price_history[symbol] = []
            self._spread_history[symbol] = []
            self._atr_buffer[symbol] = []
        self._price_history[symbol].append(price)
        self._spread_history[symbol].append(spread)
        window = 200
        if len(self._price_history[symbol]) > window:
            self._price_history[symbol] = self._price_history[symbol][-window:]
        if len(self._spread_history[symbol]) > window:
            self._spread_history[symbol] = self._spread_history[symbol][-window:]

    def compute_atr(self, symbol: str, period: int = 14) -> float:
        prices = self._price_history.get(symbol, [])
        if len(prices) < period + 1:
            return 0.0
        diffs = [abs(prices[i] - prices[i - 1]) for i in range(-period, 0)]
        return sum(diffs) / len(diffs) if diffs else 0.0

    def compute_percentile(self, values: list[float], value: float) -> float:
        if not values:
            return 0.5
        count_below = sum(1 for v in values if v <= value)
        return count_below / len(values)

    def compute_atr_normalized(self, symbol: str) -> float:
        short_atr = self.compute_atr(symbol, 14)
        long_atr = self.compute_atr(symbol, 200)
        if long_atr == 0.0 or short_atr == 0.0:
            return 0.5
        ratio = short_atr / long_atr
        buf = self._atr_buffer[symbol]
        buf.append(ratio)
        if len(buf) > 200:
            buf.pop(0)
        percentile = self.compute_percentile(buf, ratio)
        return max(0.05, min(0.95, percentile))

    def compute_spread_stability(self, symbol: str) -> float:
        spreads = self._spread_history.get(symbol, [])
        if len(spreads) < 10:
            return 1.0
        mean_s = sum(spreads) / len(spreads)
        var_s = sum((s - mean_s) ** 2 for s in spreads) / len(spreads)
        std_s = math.sqrt(var_s)
        if mean_s == 0.0:
            return 1.0
        cv = std_s / mean_s
        return max(0.05, min(0.95, 1.0 - cv * 2.0))

    def get_regime_volatility(self, symbol: str) -> float:
        atr_norm = self.compute_atr_normalized(symbol)
        reg_vol = 0.5 + (atr_norm - 0.5) * 0.8
        return max(0.1, min(1.0, reg_vol))

    def get_mra(self, symbol: str, dampen: bool = False) -> dict[str, float]:
        atr_norm = self.compute_atr_normalized(symbol)
        sp_stab = self.compute_spread_stability(symbol)
        mra_score = 0.6 * atr_norm + 0.4 * sp_stab
        if dampen:
            regime_vol = self.get_regime_volatility(symbol)
            damp_factor = 0.5 if regime_vol > 0.7 else 1.0
            mra_score = 0.5 * mra_score + 0.5 * (0.5 + damp_factor * (mra_score - 0.5))
        return {
            "mra_score": round(mra_score, 4),
            "atr_normalized": round(atr_norm, 4),
            "spread_stability": round(sp_stab, 4),
        }
