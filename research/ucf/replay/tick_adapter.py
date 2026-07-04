from __future__ import annotations

import statistics
from typing import Any


class TickAdapter:
    def __init__(self) -> None:
        self._prev_prices: dict[str, float] = {}
        self._max_volume: float = 1.0

    def adapt_tick(self, tick: dict) -> dict:
        symbol: str = tick["symbol"]
        bid: float = tick["bid"]
        ask: float = tick["ask"]
        mid_price: float = (bid + ask) / 2.0
        spread: float = float(tick["spread"])
        timestamp: float = tick["time"]
        volume: float = float(tick["volume"])

        prev_price: float | None = self._prev_prices.get(symbol)
        if prev_price is None:
            direction: int = 0
            price_change: float = 0.0
        else:
            price_change = mid_price - prev_price
            if price_change > 0:
                direction = 1
            elif price_change < 0:
                direction = -1
            else:
                direction = 0

        self._prev_prices[symbol] = mid_price

        if volume > self._max_volume:
            self._max_volume = volume

        price_level: float = mid_price if mid_price != 0.0 else 1.0
        volatility: float = abs(price_change) / price_level

        liquidity_signal: float = min(1.0, volume / self._max_volume)

        return {
            "symbol": symbol,
            "mid_price": mid_price,
            "spread": spread,
            "timestamp": timestamp,
            "direction": direction,
            "volatility": volatility,
            "liquidity_signal": liquidity_signal,
        }

    def adapt_tick_batch(self, ticks: list[dict]) -> list[dict]:
        return [self.adapt_tick(t) for t in ticks]

    def ticks_to_technical_state(self, tick_signals: list[dict]) -> dict[str, dict[str, Any]]:
        symbol_groups: dict[str, list[dict]] = {}
        for signal in tick_signals:
            sym: str = signal["symbol"]
            if sym not in symbol_groups:
                symbol_groups[sym] = []
            symbol_groups[sym].append(signal)

        result: dict[str, dict[str, Any]] = {}
        for symbol, signals in symbol_groups.items():
            recent: list[dict] = signals[-10:]

            directions: list[int] = [s["direction"] for s in recent]
            volatilities: list[float] = [s["volatility"] for s in recent]

            avg_direction: float = sum(directions) / len(directions) if directions else 0.0
            majority_direction: int = 1 if avg_direction > 0 else (-1 if avg_direction < 0 else 0)

            direction_consistency: float = abs(sum(d for d in directions)) / len(directions) if directions else 0.0
            avg_volatility: float = sum(volatilities) / len(volatilities) if volatilities else 0.0
            conviction: float = direction_consistency * min(1.0, avg_volatility * 10000.0) if directions else 0.0

            direction_changes: list[int] = []
            for i in range(1, len(directions)):
                direction_changes.append(abs(directions[i] - directions[i - 1]))

            if len(direction_changes) > 1:
                std_dir: float = statistics.stdev(direction_changes)
            else:
                std_dir = 0.0

            stability: float = 1.0 - min(1.0, std_dir)

            result[symbol] = {
                "conviction": conviction,
                "direction": majority_direction,
                "stability": stability,
            }

        return result
