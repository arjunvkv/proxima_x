"""LiveTickSource — wraps MT5Connector as a TickSource.

Emits CANONICAL ticks (see data.canonical_tick) so the consumer receives
exactly the same shape as the replay path — apples-to-apples by construction.
"""
from typing import Optional, Dict, List

from .tick_source import TickSource
from .canonical_tick import normalize_tick


class LiveTickSource(TickSource):
    def __init__(self, mt5):
        self.mt5 = mt5

    def next_tick(self, symbol: str = None) -> Optional[Dict]:
        if symbol is None:
            return None
        raw = self.mt5.get_tick(symbol)
        if raw is None:
            return None
        return normalize_tick(raw, symbol=symbol)

    def stream(self, symbols: List[str] = None):
        while True:
            for sym in (symbols or []):
                tick = self.next_tick(sym)
                if tick:
                    yield tick

    def reset(self):
        pass
