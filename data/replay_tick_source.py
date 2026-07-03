"""ReplayTickSource — wraps ReplayFeed as a TickSource.
Consumes the next merged tick event from the global stream. Symbol param is ignored."""
from typing import Optional, Dict, List
from .tick_source import TickSource


class ReplayTickSource(TickSource):
    def __init__(self, feed):
        self.feed = feed

    def next_tick(self, symbol: str = None) -> Optional[Dict]:
        return self.feed.next()

    def stream(self, symbols: List[str] = None):
        while True:
            tick = self.next_tick()
            if tick is None:
                break
            if symbols is None or tick.get("symbol") in symbols:
                yield tick

    def reset(self):
        self.feed.seek(0)
