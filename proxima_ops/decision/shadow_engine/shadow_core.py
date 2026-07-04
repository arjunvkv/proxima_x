import time
import threading
import queue
import json


class ShadowCore:
    """
    Lightweight capture-only interface.
    MUST NOT compute anything in Phase 1.
    """

    def __init__(self):
        self.queue = queue.Queue(maxsize=10000)

    def capture(self, layer: str, symbol: str, state: dict):
        """
        O(1) non-blocking capture
        """
        try:
            self.queue.put_nowait({
                "ts": time.time(),
                "layer": layer,
                "symbol": symbol,
                "state": state
            })
        except queue.Full:
            # drop silently (Phase 1 safety rule)
            pass
