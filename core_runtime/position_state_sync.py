"""
Position State Synchronizer — Fix false "Position Exists" rejections.

The problem:
  Position-exists false rejections occur because the local position cache
  is not properly synchronized with MT5. Signals are rejected because the
  system thinks a position exists when it has already been closed.

The fix:
  1. Create a position state cache that syncs every cycle from MT5
  2. Invalidate stale entries > N seconds old
  3. Track position lifecycle (open → closing → closed → removed)
  4. Provide authoritative position-state queries

This module is designed to replace the ad-hoc position checks in
run_proxima_demo.py with a reliable, synchronized cache.
"""
import os
import json
import time
import logging
from collections import defaultdict
from datetime import datetime
from typing import Optional, Dict, Any, List, Callable, Tuple

logger = logging.getLogger("position_state_sync")

# Position lifecycle states
POSITION_OPEN = "OPEN"
POSITION_CLOSING = "CLOSING"  # Close requested, awaiting MT5 confirmation
POSITION_CLOSED = "CLOSED"    # Confirmed closed by MT5
POSITION_STALE = "STALE"      # Cache entry expired, needs re-verification


class PositionState:
    """Represents the known state of a position for a symbol."""

    def __init__(self, symbol: str, ticket: int, volume: float,
                 direction: str, open_price: float, open_time: int):
        self.symbol = symbol
        self.ticket = ticket
        self.volume = volume
        self.direction = direction  # "buy" or "sell"
        self.open_price = open_price
        self.open_time = open_time
        self.state = POSITION_OPEN
        self.last_verified = time.time()
        self.close_request_time: Optional[float] = None
        self.close_price: Optional[float] = None
        self.close_time: Optional[int] = None

    @property
    def age_seconds(self) -> float:
        return time.time() - self.last_verified

    @property
    def is_active(self) -> bool:
        return self.state in (POSITION_OPEN, POSITION_CLOSING)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "ticket": self.ticket,
            "volume": self.volume,
            "direction": self.direction,
            "open_price": self.open_price,
            "open_time": self.open_time,
            "state": self.state,
            "last_verified": self.last_verified,
            "age_seconds": round(self.age_seconds, 1),
            "close_price": self.close_price,
            "close_time": self.close_time,
        }


class PositionStateSynchronizer:
    """
    Synchronized position state cache.

    Usage:
        sync = PositionStateSynchronizer(mt5_connector)
        sync.sync()  # Call every cycle
        if sync.has_active_position("EURUSD"):
            # position exists
        positions = sync.get_active_positions()
    """

    def __init__(self, mt5_connector=None,
                 max_stale_seconds: float = 10.0,
                 verify_on_query: bool = True):
        """
        Args:
            mt5_connector: MT5 connector with get_positions() method
            max_stale_seconds: Max age before cache entry is stale
            verify_on_query: Whether to re-verify on query if stale
        """
        self._mt5 = mt5_connector
        self._max_stale = max_stale_seconds
        self._verify_on_query = verify_on_query

        # Internal state: symbol → PositionState
        self._positions: Dict[str, PositionState] = {}

        # Broker ticket → symbol mapping for reverse lookup
        self._ticket_to_symbol: Dict[int, str] = {}

        # Sync tracking
        self._last_sync_time = 0.0
        self._sync_count = 0
        self._sync_errors = 0

        # Staleness tracking
        self._stale_hits = 0
        self._stale_reverified = 0

        # Position exists false-positive tracking
        self._false_positive_exists = 0
        self._false_negative_exists = 0

        logger.info(
            f"[POS_SYNC] Initialized max_stale={max_stale_seconds}s "
            f"verify_on_query={verify_on_query}"
        )

    def sync(self) -> int:
        """
        Synchronize position state with MT5.

        Should be called every cycle (~60s).

        Returns:
            Number of active positions found
        """
        if self._mt5 is None:
            if self._positions:
                # No MT5 connector — just age all entries
                now = time.time()
                for pos in self._positions.values():
                    pos.last_verified = now
            return len(self._positions)

        try:
            mt5_positions = self._mt5.get_positions()
        except Exception as e:
            self._sync_errors += 1
            logger.warning(f"[POS_SYNC] MT5 get_positions() error: {e}")
            # On error, keep existing state but mark all as potentially stale
            for pos in self._positions.values():
                if pos.state == POSITION_OPEN:
                    pos.state = POSITION_STALE
            return len(self._positions)

        self._sync_count += 1
        self._last_sync_time = time.time()

        # Build set of symbols currently in MT5
        mt5_symbols: Dict[str, dict] = {}
        for p in mt5_positions:
            sym = p.get("symbol", "")
            if sym:
                mt5_symbols[sym] = p

        # Process MT5 positions
        for sym, p in mt5_symbols.items():
            ticket = p.get("ticket", 0)
            volume = p.get("volume", 0.0)
            direction = "buy" if p.get("type", 0) in (0,) else "sell"
            open_price = p.get("price_open", 0.0)
            open_time = p.get("time", 0)

            if sym in self._positions:
                existing = self._positions[sym]
                if existing.ticket == ticket:
                    # Same ticket — just update timestamp
                    existing.state = POSITION_OPEN
                    existing.last_verified = time.time()
                    existing.volume = volume
                    if existing.state == POSITION_CLOSING:
                        # Close completed, but position still open? 
                        # Possibly a close-request that failed
                        existing.state = POSITION_OPEN
                        existing.close_request_time = None
                        logger.info(f"[POS_SYNC] Position {sym} ticket={ticket} "
                                    f"still open after close request")
                else:
                    # Different ticket — old closed, new opened
                    old = existing
                    if old.state != POSITION_CLOSED:
                        self._detect_missed_close(old)
                    # Replace with new
                    self._positions[sym] = PositionState(
                        sym, ticket, volume, direction, open_price, open_time
                    )
                    self._ticket_to_symbol[ticket] = sym
                    logger.info(f"[POS_SYNC] New position {sym} ticket={ticket}")
            else:
                # New position
                self._positions[sym] = PositionState(
                    sym, ticket, volume, direction, open_price, open_time
                )
                self._ticket_to_symbol[ticket] = sym

        # Mark removed positions
        for sym, pos in list(self._positions.items()):
            if sym not in mt5_symbols and pos.state == POSITION_OPEN:
                # Position disappeared from MT5 — it was closed externally
                pos.state = POSITION_CLOSED
                pos.close_time = int(time.time())
                logger.info(f"[POS_SYNC] Position {sym} ticket={pos.ticket} "
                            f"disappeared from MT5 -> CLOSED")
            elif sym not in mt5_symbols and pos.state == POSITION_CLOSING:
                # Our close request succeeded
                pos.state = POSITION_CLOSED
                pos.close_time = int(time.time())
                logger.info(f"[POS_SYNC] Position {sym} close confirmed")
                # Remove from active tracking after brief grace
                # (keep for 2 more sync cycles for lifecycle tracking)
            elif sym not in mt5_symbols and pos.state == POSITION_STALE:
                # Stale and gone — clean up
                del self._positions[sym]
                logger.debug(f"[POS_SYNC] Cleaned stale entry for {sym}")

        return len([p for p in self._positions.values() if p.is_active])

    def _detect_missed_close(self, old_pos: PositionState):
        """Log when a position was expected to close but didn't."""
        self._false_negative_exists += 1
        logger.warning(
            f"[POS_SYNC] Missed close detection: {old_pos.symbol} "
            f"ticket={old_pos.ticket} state={old_pos.state}"
        )

    def has_active_position(self, symbol: str) -> bool:
        """
        Check if a symbol has an active position.

        Returns False immediately for CLOSED or CLOSING->confirmed positions.
        """
        pos = self._positions.get(symbol)
        if pos is None:
            return False

        # If stale and verify_on_query, re-check
        if pos.age_seconds > self._max_stale and self._verify_on_query and self._mt5:
            self._stale_hits += 1
            try:
                mt5_positions = self._mt5.get_positions()
                mt5_symbols = {p.get("symbol", "") for p in mt5_positions}
                if symbol not in mt5_symbols:
                    pos.state = POSITION_CLOSED
                    pos.close_time = int(time.time())
                    self._stale_reverified += 1
                    logger.info(f"[POS_SYNC] Stale re-verify: {symbol} -> CLOSED")
                    return False
                else:
                    pos.last_verified = time.time()
                    self._stale_reverified += 1
                    return True
            except Exception:
                # On error, trust the cache
                pass

        # If position is closing, check how long ago
        if pos.state == POSITION_CLOSING:
            if pos.close_request_time and (time.time() - pos.close_request_time) > 30.0:
                # Close request >30s ago, position should be gone
                # But it's still in our cache — might be stuck
                logger.warning(f"[POS_SYNC] Stuck close request for {symbol}, "
                               f"ticket={pos.ticket}")
                # Return True to prevent duplicate entries
                return True

        return pos.is_active

    def request_close(self, symbol: str):
        """Mark a position as closing (close requested to MT5)."""
        pos = self._positions.get(symbol)
        if pos and pos.state == POSITION_OPEN:
            pos.state = POSITION_CLOSING
            pos.close_request_time = time.time()
            logger.info(f"[POS_SYNC] Close requested for {symbol} ticket={pos.ticket}")

    def get_active_positions(self) -> List[PositionState]:
        """Get all currently active positions."""
        return [p for p in self._positions.values() if p.is_active]

    def get_position(self, symbol: str) -> Optional[PositionState]:
        """Get position state for a symbol."""
        return self._positions.get(symbol)

    def position_count(self) -> int:
        """Count of active positions."""
        return len(self.get_active_positions())

    def summary(self) -> dict:
        """Return synchronization summary."""
        active = self.get_active_positions()
        closed = [p for p in self._positions.values() if p.state == POSITION_CLOSED]
        return {
            "last_sync_time": self._last_sync_time,
            "sync_count": self._sync_count,
            "sync_errors": self._sync_errors,
            "active_positions": len(active),
            "closed_positions": len(closed),
            "total_tracked": len(self._positions),
            "max_stale_seconds": self._max_stale,
            "stale_hits": self._stale_hits,
            "stale_reverified": self._stale_reverified,
            "false_positive_exists": self._false_positive_exists,
            "false_negative_exists": self._false_negative_exists,
            "positions": {
                sym: pos.to_dict() for sym, pos in self._positions.items()
            },
            "sync_health": "OK" if self._sync_errors < self._sync_count * 0.1 else "DEGRADED",
        }

    def to_dict(self) -> dict:
        """Serialize all position states."""
        return {
            "positions": {
                sym: pos.to_dict() for sym, pos in sorted(self._positions.items())
            },
            "sync_count": self._sync_count,
            "sync_errors": self._sync_errors,
        }


# Singleton for global use
_INSTANCE: Optional[PositionStateSynchronizer] = None


def get_position_sync(mt5_connector=None) -> PositionStateSynchronizer:
    """Get or create the global PositionStateSynchronizer instance."""
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = PositionStateSynchronizer(mt5_connector)
    return _INSTANCE
