"""LiveTickSource — wraps MT5Connector as a TickSource."""
from typing import Optional, Dict, List
from .tick_source import TickSource


class LiveTickSource(TickSource):
    def __init__(self, mt5):
        self.mt5 = mt5

    def next_tick(self, symbol: str = None) -> Optional[Dict]:
        if symbol is None:
            return None
        return self.mt5.get_tick(symbol)

    def stream(self, symbols: List[str] = None):
        while True:
            for sym in (symbols or []):
                tick = self.next_tick(sym)
                if tick:
                    yield tick

    def reset(self):
        pass
