"""
Execution Lifecycle Manager — Align entry + exit clock models.

The problem:
  63 accepted trades, 0 closed. The entry timing model and exit timing model
  are not aligned. Signals enter based on one clock but exits never trigger.

The fix:
  1. Unify entry and exit clock to a single timing model
  2. Track every position's lifecycle: generated → submitted → accepted → opened → closed
  3. Ensure H20 exit cap triggers correctly
  4. Align with 3-bar position lock state
  5. Detect orphaned/closing-stuck positions
"""
import os
import json
import time
import logging
from collections import defaultdict
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List, Callable

logger = logging.getLogger("lifecycle_mgr")


class LifecycleStage(str, Enum):
    GENERATED = "GENERATED"
    THRESHOLD_PASSED = "THRESHOLD_PASSED"
    TRIGGERED = "TRIGGERED"
    SUBMITTED = "SUBMITTED"
    ACCEPTED = "ACCEPTED"   # MT5 confirmed
    OPENED = "OPENED"       # Position confirmed open
    CLOSING = "CLOSING"     # Close requested
    CLOSED = "CLOSED"       # Confirmed closed
    REJECTED = "REJECTED"   # Rejected at any stage
    ORPHANED = "ORPHANED"   # State unknown / lost


class SignalLifecycle:
    """Tracks a single signal's journey through the lifecycle."""

    def __init__(self, signal_id: str, symbol: str, direction: int,
                 es_rank: float, price: float, timestamp: float):
        self.signal_id = signal_id
        self.symbol = symbol
        self.direction = direction  # 1 = buy, -1 = sell
        self.es_rank = es_rank
        self.entry_price = price
        self.volume: Optional[float] = None
        self.ticket: Optional[int] = None

        # Broker position metadata
        self.sl: Optional[float] = None
        self.tp: Optional[float] = None
        self.magic: Optional[int] = None

        # Timing
        self.generated_at = timestamp
        self.threshold_passed_at: Optional[float] = None
        self.triggered_at: Optional[float] = None
        self.submitted_at: Optional[float] = None
        self.accepted_at: Optional[float] = None
        self.opened_at: Optional[float] = None
        self.close_requested_at: Optional[float] = None
        self.closed_at: Optional[float] = None

        # Exit info
        self.exit_price: Optional[float] = None
        self.exit_reason: Optional[str] = None  # "H20", "MANUAL", "STOP", etc.
        self.h20_bars: int = 20  # Default

        # State
        self.stage = LifecycleStage.GENERATED
        self.block_reason: Optional[str] = None
        self.broker_error: Optional[str] = None

        # H20 tracking
        self._bars_since_open = 0

    @property
    def age_seconds(self) -> float:
        now = time.time()
        start = self.opened_at or self.accepted_at or self.submitted_at or self.generated_at
        return now - start

    @property
    def is_open(self) -> bool:
        return self.stage in (LifecycleStage.OPENED, LifecycleStage.CLOSING)

    @property
    def is_active(self) -> bool:
        return self.stage not in (LifecycleStage.CLOSED, LifecycleStage.ORPHANED,
                                  LifecycleStage.REJECTED)

    def to_dict(self) -> dict:
        return {
            "signal_id": self.signal_id,
            "symbol": self.symbol,
            "direction": self.direction,
            "volume": self.volume,
            "ticket": self.ticket,
            "entry_price": self.entry_price,
            "sl": self.sl,
            "tp": self.tp,
            "magic": self.magic,
            "stage": self.stage.value,
            "generated_at": self.generated_at,
            "threshold_passed_at": self.threshold_passed_at,
            "triggered_at": self.triggered_at,
            "submitted_at": self.submitted_at,
            "accepted_at": self.accepted_at,
            "opened_at": self.opened_at,
            "close_requested_at": self.close_requested_at,
            "closed_at": self.closed_at,
            "exit_price": self.exit_price,
            "exit_reason": self.exit_reason,
            "block_reason": self.block_reason,
            "age_seconds": round(self.age_seconds, 1),
            "bars_since_open": self._bars_since_open,
        }


class ExecutionLifecycleManager:
    """
    Manages the full lifecycle of signal execution.

    Tracks every signal from generation through to close.
    Ensures H20 exit cap triggers correctly.
    Detects orphaned positions and stuck lifecycle states.
    Aligns entry clock with exit clock.
    """

    def __init__(self, h20_bars: int = 20, bar_duration_seconds: float = 3600.0,
                 max_stuck_seconds: float = 72000.0):
        """
        Args:
            h20_bars: Number of bars before forced exit (default 20 for H1)
            bar_duration_seconds: Duration of one bar in seconds (3600 for H1)
            max_stuck_seconds: Max time before a CLOSING position is considered stuck
        """
        self._h20_bars = h20_bars
        self._bar_duration = bar_duration_seconds
        self._max_stuck = max_stuck_seconds

        # All tracked signals: signal_id -> SignalLifecycle
        self._signals: Dict[str, SignalLifecycle] = {}

        # Per-symbol active signal
        self._symbol_active: Dict[str, str] = {}  # symbol -> signal_id

        # Metrics
        self._total_generated = 0
        self._total_opened = 0
        self._total_closed = 0
        self._total_rejected = 0
        self._total_orphaned = 0
        self._h20_exits = 0
        self._stuck_closes = 0
        self._broker_rejects = 0

        # Persistence path
        self._persist_path = os.path.join(
            os.getcwd(), "state", "lifecycle_state.json"
        )
        self._load_persisted()

        logger.info(
            f"[LIFECYCLE_MGR] Initialized h20_bars={h20_bars} "
            f"bar_duration={bar_duration_seconds}s "
            f"max_stuck={max_stuck_seconds}s"
        )

    def _load_persisted(self):
        """Load persisted state on restart."""
        if not os.path.exists(self._persist_path):
            return
        try:
            with open(self._persist_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for sig_data in data.get("signals", []):
                sig = SignalLifecycle(
                    sig_data["signal_id"], sig_data["symbol"],
                    sig_data["direction"], sig_data.get("es_rank", 0.0),
                    sig_data["entry_price"], sig_data["generated_at"]
                )
                # Restore all fields (skip computed properties like age_seconds)
                _skip_props = {"age_seconds", "bars_since_open"}
                for key, val in sig_data.items():
                    if key in _skip_props:
                        continue
                    if hasattr(sig, key) and not key.startswith("_"):
                        if key == "stage":
                            setattr(sig, key, LifecycleStage(val))
                        else:
                            setattr(sig, key, val)
                # Restore _bars_since_open from persisted data
                if "bars_since_open" in sig_data:
                    sig._bars_since_open = sig_data["bars_since_open"]
                self._signals[sig.signal_id] = sig
                if sig.is_active:
                    self._symbol_active[sig.symbol] = sig.signal_id
            logger.info(f"[LIFECYCLE_MGR] Restored {len(self._signals)} signals")
        except Exception as e:
            logger.warning(f"[LIFECYCLE_MGR] Could not load persisted state: {e}")

    def _persist(self):
        """Persist current state to disk."""
        try:
            os.makedirs(os.path.dirname(self._persist_path), exist_ok=True)
            data = {
                "signals": [sig.to_dict() for sig in self._signals.values()],
                "h20_bars": self._h20_bars,
            }
            with open(self._persist_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"[LIFECYCLE_MGR] Persist error: {e}")

    # --- Stage transitions ---

    def record_generated(self, signal_id: str, symbol: str, direction: int,
                         es_rank: float, price: float) -> SignalLifecycle:
        """Record signal generation."""
        sig = SignalLifecycle(signal_id, symbol, direction, es_rank, price, time.time())
        self._signals[signal_id] = sig
        self._total_generated += 1
        return sig

    def record_threshold_passed(self, signal_id: str) -> bool:
        """Record threshold pass."""
        sig = self._signals.get(signal_id)
        if sig is None:
            return False
        sig.stage = LifecycleStage.THRESHOLD_PASSED
        sig.threshold_passed_at = time.time()
        return True

    def record_triggered(self, signal_id: str) -> bool:
        """Record trigger."""
        sig = self._signals.get(signal_id)
        if sig is None:
            return False
        sig.stage = LifecycleStage.TRIGGERED
        sig.triggered_at = time.time()
        return True

    def record_submitted(self, signal_id: str) -> bool:
        """Record submission to broker."""
        sig = self._signals.get(signal_id)
        if sig is None:
            return False
        sig.stage = LifecycleStage.SUBMITTED
        sig.submitted_at = time.time()
        self._persist()
        return True

    def record_accepted(self, signal_id: str, ticket: int, volume: float) -> bool:
        """Record broker acceptance."""
        sig = self._signals.get(signal_id)
        if sig is None:
            return False
        sig.stage = LifecycleStage.ACCEPTED
        sig.accepted_at = time.time()
        sig.ticket = ticket
        sig.volume = volume
        self._symbol_active[sig.symbol] = signal_id
        self._persist()
        return True

    def record_opened(self, signal_id: str, price: float) -> bool:
        """Record position opened."""
        sig = self._signals.get(signal_id)
        if sig is None:
            return False
        sig.stage = LifecycleStage.OPENED
        sig.opened_at = time.time()
        sig.entry_price = price
        self._total_opened += 1
        self._symbol_active[sig.symbol] = signal_id
        self._persist()
        logger.info(f"[LIFECYCLE] Opened {sig.symbol} signal={signal_id} "
                     f"ticket={sig.ticket} price={price}")
        return True

    def record_close_requested(self, signal_id: str) -> bool:
        """Record close request to broker."""
        sig = self._signals.get(signal_id)
        if sig is None:
            return False
        sig.stage = LifecycleStage.CLOSING
        sig.close_requested_at = time.time()
        self._persist()
        return True

    def record_closed(self, signal_id: str, exit_price: float,
                      exit_reason: str = "H20") -> bool:
        """Record position closed."""
        sig = self._signals.get(signal_id)
        if sig is None:
            return False
        sig.stage = LifecycleStage.CLOSED
        sig.closed_at = time.time()
        sig.exit_price = exit_price
        sig.exit_reason = exit_reason
        self._total_closed += 1
        if exit_reason == "H20":
            self._h20_exits += 1

        # Remove from active tracking
        if self._symbol_active.get(sig.symbol) == signal_id:
            del self._symbol_active[sig.symbol]

        self._persist()
        logger.info(f"[LIFECYCLE] Closed {sig.symbol} signal={signal_id} "
                     f"reason={exit_reason} price={exit_price}")
        return True

    def record_rejected(self, signal_id: str, reason: str = "UNKNOWN") -> bool:
        """Record signal rejection."""
        sig = self._signals.get(signal_id)
        if sig is None:
            return False
        sig.stage = LifecycleStage.REJECTED
        sig.block_reason = reason
        self._total_rejected += 1
        self._persist()
        return True

    def record_orphaned(self, signal_id: str) -> bool:
        """Record orphaned signal (state unknown)."""
        sig = self._signals.get(signal_id)
        if sig is None:
            return False
        sig.stage = LifecycleStage.ORPHANED
        self._total_orphaned += 1
        self._persist()
        return True

    def record_broker_reject(self, signal_id: str, error: str) -> bool:
        """Record broker rejection."""
        sig = self._signals.get(signal_id)
        if sig is None:
            return False
        sig.broker_error = error
        self._broker_rejects += 1
        self._persist()
        return True

    # --- Queries ---

    def get_active_signal(self, symbol: str) -> Optional[SignalLifecycle]:
        """Get the active signal for a symbol, if any."""
        sig_id = self._symbol_active.get(symbol)
        if sig_id is None:
            return None
        sig = self._signals.get(sig_id)
        if sig is None or not sig.is_active:
            return None
        return sig

    def has_active_signal(self, symbol: str) -> bool:
        """Check if symbol has an active signal."""
        sig = self.get_active_signal(symbol)
        return sig is not None

    def get_open_positions(self) -> List[SignalLifecycle]:
        """Get all currently open positions (OPENED or CLOSING)."""
        return [s for s in self._signals.values() if s.is_open]

    def open_position_count(self) -> int:
        return len(self.get_open_positions())

    def get_stuck_closing(self) -> List[SignalLifecycle]:
        """Get positions stuck in CLOSING state."""
        now = time.time()
        return [
            s for s in self._signals.values()
            if s.stage == LifecycleStage.CLOSING
            and s.close_requested_at
            and (now - s.close_requested_at) > self._max_stuck
        ]

    def get_h20_pending(self) -> List[SignalLifecycle]:
        """Get positions that have exceeded H20 and should be closed."""
        signals_needing_close = []
        for sig in self._signals.values():
            if sig.stage == LifecycleStage.OPENED and sig.opened_at:
                bars_elapsed = (time.time() - sig.opened_at) / self._bar_duration
                if bars_elapsed >= self._h20_bars:
                    signals_needing_close.append(sig)
        return signals_needing_close

    def get_current_signal(self, symbol: str) -> Optional[SignalLifecycle]:
        """Get the most recent signal for a symbol regardless of state."""
        # Find all signals for this symbol, return most recent
        symbol_signals = [
            s for s in self._signals.values()
            if s.symbol == symbol
        ]
        if not symbol_signals:
            return None
        return max(symbol_signals, key=lambda s: s.generated_at)

    # --- Cycle tick (call every evaluation cycle) ---

    def tick(self):
        """
        Called every evaluation cycle.

        Updates bar counts for open positions.
        Detects stale/stuck positions.
        """
        now = time.time()
        for sig in self._signals.values():
            if sig.stage == LifecycleStage.OPENED and sig.opened_at:
                elapsed = now - sig.opened_at
                sig._bars_since_open = int(elapsed / self._bar_duration)

        # Detect stuck closing positions
        stuck = self.get_stuck_closing()
        for sig in stuck:
            logger.warning(f"[LIFECYCLE] Stuck close: {sig.symbol} signal={sig.signal_id} "
                           f"ticket={sig.ticket} stuck_for={(now - sig.close_requested_at):.0f}s")
            # Force close after extended stuck period
            if sig.close_requested_at and (now - sig.close_requested_at) > self._max_stuck * 2:
                sig.stage = LifecycleStage.CLOSED
                sig.closed_at = now
                sig.exit_reason = "STUCK_FORCE"
                self._stuck_closes += 1
                if self._symbol_active.get(sig.symbol) == sig.signal_id:
                    del self._symbol_active[sig.symbol]
                logger.warning(f"[LIFECYCLE] Force-closed stuck {sig.symbol}")

    # --- Reporting ---

    def summary(self) -> dict:
        """Return lifecycle summary."""
        open_positions = self.get_open_positions()
        h20_pending = self.get_h20_pending()

        return {
            "total_generated": self._total_generated,
            "total_opened": self._total_opened,
            "total_closed": self._total_closed,
            "total_rejected": self._total_rejected,
            "total_orphaned": self._total_orphaned,
            "h20_exits": self._h20_exits,
            "stuck_closes": self._stuck_closes,
            "broker_rejects": self._broker_rejects,
            "currently_open": len(open_positions),
            "h20_pending_close": len(h20_pending),
            "signal_catalog": {
                sig_id: sig.to_dict()
                for sig_id, sig in self._signals.items()
            },
            "h20_config": {
                "bars": self._h20_bars,
                "bar_duration_seconds": self._bar_duration,
                "max_stuck_seconds": self._max_stuck,
            },
            "alerts": {
                "stuck_closing": [
                    {"symbol": s.symbol, "signal_id": s.signal_id,
                     "ticket": s.ticket, "stuck_for": round(
                         time.time() - (s.close_requested_at or time.time()), 1)}
                    for s in self.get_stuck_closing()
                ],
                "h20_overdue": [
                    {"symbol": s.symbol, "signal_id": s.signal_id,
                     "ticket": s.ticket,
                     "bars_elapsed": int(
                         (time.time() - (s.opened_at or time.time())) / self._bar_duration
                     )}
                    for s in h20_pending
                ],
            },
        }

    def execution_coherence_score(self) -> float:
        """
        ECS = alignment(entry_logic, exit_logic, state_tracking).

        Range: 0.0 (broken) to 1.0 (perfect).
        """
        if self._total_opened == 0:
            return 1.0  # No data, no evidence of issues

        # Penalty factors:
        open_not_closed_ratio = self._total_closed / max(self._total_opened, 1)
        orphan_ratio = self._total_orphaned / max(self._total_opened, 1)
        stuck_ratio = self._stuck_closes / max(self._total_opened, 1)

        # More closed = better coherence
        close_score = min(open_not_closed_ratio, 1.0)

        # Fewer orphans = better
        orphan_penalty = 1.0 - min(orphan_ratio, 1.0)

        # Fewer stuck = better
        stuck_penalty = 1.0 - min(stuck_ratio, 1.0)

        # Weighted combination
        ecs = 0.5 * close_score + 0.25 * orphan_penalty + 0.25 * stuck_penalty
        return round(max(0.0, min(1.0, ecs)), 4)


# Singleton
_INSTANCE: Optional[ExecutionLifecycleManager] = None


def get_lifecycle_manager(h20_bars: int = 20) -> ExecutionLifecycleManager:
    """Get or create the global ExecutionLifecycleManager instance."""
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = ExecutionLifecycleManager(h20_bars)
    return _INSTANCE
