import logging
from abc import ABC, abstractmethod
from typing import Iterator, Optional

logger = logging.getLogger("proxima.adapters.tick_source")


class TickSource(ABC):

    @abstractmethod
    def get_tick(self, symbol: str) -> Optional[dict]:
        pass

    @abstractmethod
    def get_ticks(self, symbol: str, start_ts: int, end_ts: int) -> list[dict]:
        pass

    @abstractmethod
    def stream(self, symbols: list[str] = None) -> Iterator[dict]:
        pass

    @abstractmethod
    def current_spread(self, symbol: str) -> float:
        pass


class MT5TickSource(TickSource):
    def __init__(self, mt5_connector):
        self._mt5 = mt5_connector

    def get_tick(self, symbol: str) -> Optional[dict]:
        return self._mt5.get_tick(symbol)

    def get_ticks(self, symbol: str, start_ts: int, end_ts: int) -> list[dict]:
        tick = self._mt5.get_tick(symbol)
        return [tick] if tick else []

    def stream(self, symbols: list[str] = None) -> Iterator[dict]:
        while True:
            for sym in (symbols or []):
                tick = self.get_tick(sym)
                if tick:
                    yield tick

    def current_spread(self, symbol: str) -> float:
        tick = self.get_tick(symbol)
        return tick["spread"] if tick else 999.0


class ReplayTickSource(TickSource):
    def __init__(self, replay_feed):
        self._feed = replay_feed
        self._symbol_cursors: dict[str, int] = {}
        self._current_ticks: dict[str, Optional[dict]] = {}
        self._ledger = None

    def set_ledger(self, ledger):
        self._ledger = ledger

    def get_tick(self, symbol: str) -> Optional[dict]:
        symbol = symbol.upper()
        if self._feed.total == 0:
            return None
        if self._feed.done:
            return self._current_ticks.get(symbol)

        cursor = self._symbol_cursors.get(symbol, -1)
        start = max(cursor + 1, 0)

        for i in range(start, self._feed.total):
            tick = self._feed.get_by_index(i)
            if tick is None:
                continue
            ts = tick.get("time_sec", tick.get("timestamp", 0))
            self._feed._clock.advance_to(ts)
            self._feed._cursor = i + 1
            self._current_ticks[tick.get("symbol", "").upper()] = tick
            if self._ledger is not None:
                self._ledger.add_tick(tick)
            if tick.get("symbol", "").upper() == symbol:
                self._symbol_cursors[symbol] = i
                return tick

        return self._current_ticks.get(symbol)

    def get_ticks(self, symbol: str, start_ts: int, end_ts: int) -> list[dict]:
        return self._feed.get_range(symbol, start_ts, end_ts)

    def stream(self, symbols: list[str] = None) -> Iterator[dict]:
        return self._feed.stream(symbols)

    def current_spread(self, symbol: str) -> float:
        tick = self.get_tick(symbol)
        return (tick["ask"] - tick["bid"]) / 1e-5 if tick else 999.0
