"""TickSource abstraction — unified tick interface for live/replay modes."""
from typing import Optional, Dict, List


class TickSource:
    def next_tick(self, symbol: str = None) -> Optional[Dict]:
        """Live: poll specific symbol. Replay: consume next merged event."""
        raise NotImplementedError

    def stream(self, symbols: List[str] = None):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError
