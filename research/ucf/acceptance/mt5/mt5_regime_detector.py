import math
from typing import Any


class MT5RegimeDetector:
    def __init__(self, atr_period: int = 20) -> None:
        self._price_buffer: dict[str, list[float]] = {}
        self._spread_buffer: dict[str, list[float]] = {}
        self._atr_period = atr_period

    def feed_tick(self, tick: dict[str, Any]) -> None:
        sym = tick["symbol"]
        bid = tick.get("bid", 0)
        ask = tick.get("ask", 0)
        mid = (bid + ask) / 2 if bid and ask else 0
        spread = abs(ask - bid) if bid and ask else 0
        if sym not in self._price_buffer:
            self._price_buffer[sym] = []
            self._spread_buffer[sym] = []
        self._price_buffer[sym].append(mid)
        self._spread_buffer[sym].append(spread)
        if len(self._price_buffer[sym]) > 200:
            self._price_buffer[sym] = self._price_buffer[sym][-200:]
            self._spread_buffer[sym] = self._spread_buffer[sym][-200:]

    def compute_atr(self, symbol: str) -> float:
        prices = self._price_buffer.get(symbol, [])
        if len(prices) < self._atr_period + 1:
            return 0.0001
        diffs = [abs(prices[i] - prices[i - 1]) for i in range(-self._atr_period, 0)]
        return sum(diffs) / len(diffs) if diffs else 0.0001

    def compute_spread_volatility(self, symbol: str) -> float:
        spreads = self._spread_buffer.get(symbol, [])
        if len(spreads) < 10:
            return 0.0
        mean_s = sum(spreads) / len(spreads)
        if mean_s == 0:
            return 0.0
        var_s = sum((s - mean_s) ** 2 for s in spreads) / len(spreads)
        return math.sqrt(var_s) / mean_s

    def detect_regime(self, symbol: str) -> dict[str, Any]:
        atr = self.compute_atr(symbol)
        spread_vol = self.compute_spread_volatility(symbol)
        prices = self._price_buffer.get(symbol, [])
        mean_price = sum(prices) / max(1, len(prices))
        vol_score = atr / max(mean_price, 0.0001)

        if vol_score > 0.02 or spread_vol > 0.5:
            regime = "high_vol"
        elif vol_score < 0.005 and spread_vol < 0.2:
            regime = "low_vol"
        else:
            regime = "mixed"

        volatility_score = min(1.0, vol_score * 100)
        stability_index = max(0.0, 1.0 - spread_vol * 2)

        return {
            "regime": regime,
            "volatility_score": round(volatility_score, 4),
            "stability_index": round(stability_index, 4),
            "atr": round(atr, 6),
            "spread_volatility": round(spread_vol, 4),
        }
