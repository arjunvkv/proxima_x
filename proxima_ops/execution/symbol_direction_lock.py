import logging
from typing import Optional

logger = logging.getLogger("proxima_ops.execution.symbol_direction_lock")


class SymbolDirectionLock:
    """Regime-conditioned directional persistence with strength decay.

    Direction locks are not binary — they have a *strength* that decays
    based on regime continuity. Strong locks persist; weak locks allow
    reversals under high-confidence regime shifts.
    """

    _REGIME_DECAY = {
        "TREND": 0.97,
        "CHOP": 0.80,
        "TRANSITION": 0.85,
        "CHAOTIC": 0.70,
        "STRUCTURED": 0.95,
        "HIGH_VOL_TREND": 0.93,
        "LOW_VOL_DRIFT": 0.90,
    }

    def __init__(self, min_strength: float = 0.35) -> None:
        self._state: dict[str, str] = {}
        self._strength: dict[str, float] = {}
        self._cycle_log: list[dict] = []
        self._min_strength = min_strength
        self._default_regime = "TRANSITION"

    def decay_all(self, regime: str = "TRANSITION") -> None:
        """Apply regime-conditioned decay to all active locks."""
        decay = self._REGIME_DECAY.get(regime, 0.85)
        expired = []
        for sym in list(self._state.keys()):
            old = self._strength.get(sym, 1.0)
            self._strength[sym] = old * decay
            if self._strength[sym] < self._min_strength:
                expired.append(sym)
        for sym in expired:
            old_dir = self._state.pop(sym, None)
            self._strength.pop(sym, None)
            if old_dir is not None:
                self._cycle_log.append({
                    "event": "decay_expired",
                    "symbol": sym,
                    "direction": old_dir,
                })
                logger.info("[SDL_DECAY_EXPIRED] %s (was %s)", sym, old_dir)

    def is_allowed(self, symbol: str, direction: str) -> bool:
        """Check whether *direction* is permitted for *symbol*.

        Returns True when no lock exists, the lock matches *direction*,
        or the lock has decayed below minimum strength.
        Returns False when an opposite-direction lock is still strong.
        """
        current = self._state.get(symbol)
        if current is None:
            return True
        if current == direction:
            return True
        strength = self._strength.get(symbol, 0.0)
        if strength < self._min_strength:
            return True
        logger.warning(
            "[SDL_BLOCK] %s: current=%s requested=%s strength=%.2f",
            symbol, current, direction, strength,
        )
        return False

    def lock(self, symbol: str, direction: str) -> None:
        """Record that *symbol* is now committed to *direction*, resetting strength to 1.0."""
        old = self._state.get(symbol)
        self._state[symbol] = direction
        self._strength[symbol] = 1.0
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
        self._strength.pop(symbol, None)
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

    def get_strength(self, symbol: str) -> float:
        """Return the current lock strength for *symbol* (0.0 if no lock)."""
        return self._strength.get(symbol, 0.0)

    def reset(self) -> None:
        """Clear all locks, strengths, and cycle log."""
        self._state.clear()
        self._strength.clear()
        self._cycle_log.clear()
        logger.info("[SDL_RESET] all locks cleared")
