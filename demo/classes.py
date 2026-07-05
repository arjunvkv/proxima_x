"""Standalone classes extracted from run_proxima_demo.py.

These classes have no dependency on the ProximaDemo class or any demo/ module.
They are fully self-contained utility classes.
"""

import logging
from collections import defaultdict
import threading as _threading


logger = logging.getLogger("proxima_demo")


_EMITTING = False


class DashboardLogHandler(logging.Handler):
    """Log handler that feeds log messages to the demo dashboard activity feed."""

    def __init__(self, demo_instance):
        super().__init__()
        self.demo = demo_instance

    def emit(self, record):
        global _EMITTING
        if _EMITTING:
            return
        _EMITTING = True
        try:
            msg = record.getMessage()
            self.demo.add_activity(msg)
        except Exception:
            self.handleError(record)
        finally:
            _EMITTING = False


class SymbolTrustModel:
    """Online adaptive trust per symbol. Replaces static priors with outcome-driven Bayesian-style EMA."""

    def __init__(self, alpha: float = 0.08, prior: float = 1.0):
        self.alpha = alpha
        self.prior = prior
        self.trust = defaultdict(lambda: prior)
        self.observations = defaultdict(int)

    def get(self, symbol: str) -> float:
        return self.trust[symbol]

    def update(self, symbol: str, pnl: float):
        reward = 1.0 if pnl > 0 else 0.0
        current = self.trust[symbol]
        updated = (1 - self.alpha) * current + self.alpha * reward
        self.trust[symbol] = min(1.2, max(0.05, updated))
        self.observations[symbol] += 1


class TickCache:
    """Background-threaded tick cache that polls MT5 periodically."""

    def __init__(self, mt5, symbols, poll_interval=0.2):
        self._mt5 = mt5
        self._symbols = list(symbols)
        self._poll_interval = poll_interval
        self._cache = {}
        self._lock = _threading.Lock()
        self._running = True
        self._thread = _threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def _poll(self):
        while self._running:
            for sym in self._symbols:
                try:
                    tick = self._mt5.get_tick(sym)
                    if tick:
                        with self._lock:
                            self._cache[sym] = tick
                except Exception:
                    pass
            _threading.Event().wait(self._poll_interval)

    def get_tick(self, sym):
        with self._lock:
            return self._cache.get(sym)

    def get_all(self):
        with self._lock:
            return dict(self._cache)

    def stop(self):
        self._running = False
