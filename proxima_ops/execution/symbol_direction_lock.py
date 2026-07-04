import logging
from typing import Optional

logger = logging.getLogger("proxima_ops.execution.symbol_direction_lock")


class SymbolDirectionLock:
    """Ensures directional coherence per symbol across cycles.

    Once a symbol is locked in a direction (BUY/SELL), opposite-direction
    orders are blocked until the lock is explicitly released or reset.
    """

    def __init__(self) -> None:
        self._state: dict[str, str] = {}
        self._cycle_log: list[dict] = []

    def is_allowed(self, symbol: str, direction: str) -> bool:
        """Check whether *direction* is permitted for *symbol*.

        Returns True when no lock exists or the lock matches *direction*.
        Returns False when an opposite-direction lock is active.
        """
        current = self._state.get(symbol)
        if current is None:
            return True
        if current != direction:
            logger.warning(
                "[SDL_BLOCK] %s: current=%s requested=%s",
                symbol, current, direction,
            )
            return False
        return True

    def lock(self, symbol: str, direction: str) -> None:
        """Record that *symbol* is now committed to *direction*."""
        old = self._state.get(symbol)
        self._state[symbol] = direction
        self._cycle_log.append({
            "event": "lock",
            "symbol": symbol,
            "direction": direction,
            "previous_direction": old,
        })
        logger.info("[SDL_LOCK] %s -> %s (was %s)", symbol, direction, old)

    def release(self, symbol: str) -> None:
        """Remove any direction lock for *symbol*."""
        old = self._state.pop(symbol, None)
        if old is not None:
            self._cycle_log.append({
                "event": "release",
                "symbol": symbol,
                "direction": old,
            })
            logger.info("[SDL_RELEASE] %s (was %s)", symbol, old)

    def get_current(self, symbol: str) -> Optional[str]:
        """Return the currently locked direction for *symbol*, or None."""
        return self._state.get(symbol)

    def reset(self) -> None:
        """Clear all locks and cycle log."""
        self._state.clear()
        self._cycle_log.clear()
        logger.info("[SDL_RESET] all locks cleared")
