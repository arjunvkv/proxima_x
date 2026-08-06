"""ReplayTickSource — wraps ReplayFeed as a TickSource.

Consumes the next merged tick event from the global stream and emits it in the
CANONICAL shape (see data.canonical_tick) — identical to what LiveTickSource
emits, so backtest and live consume byte-identical ticks.
"""
from typing import Optional, Dict, List

from .tick_source import TickSource
from .canonical_tick import normalize_tick


class ReplayTickSource(TickSource):
    def __init__(self, feed):
        self.feed = feed

    def next_tick(self, symbol: str = None) -> Optional[Dict]:
        raw = self.feed.next()
        if raw is None:
            return None
        return normalize_tick(raw)

    def stream(self, symbols: List[str] = None):
        while True:
            tick = self.next_tick()
            if tick is None:
                break
            if symbols is None or tick.get("symbol") in symbols:
                yield tick

    def reset(self):
        self.feed.seek(0)
