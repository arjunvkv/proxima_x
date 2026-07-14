"""Tick/bar store adapted for 1-minute bar data from MT5."""
import numpy as np
from collections import deque, defaultdict
from typing import Optional
from config.settings import SYMBOLS
from data.models import Tick


class TickStore:
    """Stores the latest M1 bar close for each symbol. Returns are simple log-returns."""
    def __init__(self):
        self._bars: dict[str, deque[Tick]] = defaultdict(lambda: deque(maxlen=100))
        self._last_ts: dict[str, float] = {}

    def add_ticks(self, ticks: list[Tick]) -> None:
        for tick in ticks:
            if tick.symbol in SYMBOLS:
                self._bars[tick.symbol].append(tick)
                self._last_ts[tick.symbol] = tick.timestamp

    def add_tick(self, tick: Tick) -> None:
        if tick.symbol in SYMBOLS:
            self._bars[tick.symbol].append(tick)
            self._last_ts[tick.symbol] = tick.timestamp

    def latest(self, symbol: str) -> Optional[Tick]:
        bars = self._bars.get(symbol)
        return bars[-1] if bars else None

    def calculate_returns(self, symbols: Optional[list[str]] = None, window: int = 15) -> dict[str, float]:
        results = {}
        target = symbols or list(self._bars.keys())
        for symbol in target:
            bars = self._bars.get(symbol, [])
            n = len(bars)
            if n < 2:
                results[symbol] = 0.0
                continue
            open_ = bars[0].mid if n < window else bars[-window].mid
            close_ = bars[-1].mid
            if open_ > 0 and close_ > 0:
                results[symbol] = float(np.log(close_ / open_))
            else:
                results[symbol] = 0.0
        return results

    def freshness(self, symbol: str) -> float:
        ts = self._last_ts.get(symbol, 0.0)
        if ts == 0.0:
            return 0.0
        import time
        age = time.time() - ts
        return max(0.0, 1.0 - age / 120.0)

    def all_freshness(self) -> dict[str, float]:
        return {sym: self.freshness(sym) for sym in SYMBOLS}

    def stale_symbols(self, max_age: float = 120.0) -> list[str]:
        import time
        stale = []
        for sym in SYMBOLS:
            ts = self._last_ts.get(sym, 0.0)
            if ts == 0.0 or (time.time() - ts) > max_age:
                stale.append(sym)
        return stale

    def clear(self) -> None:
        self._bars.clear()
        self._last_ts.clear()
