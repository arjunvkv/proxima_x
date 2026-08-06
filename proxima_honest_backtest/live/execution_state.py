"""Per-symbol one-in-flight execution state machine.

FLAT -> ENTRY_PENDING -> OPEN -> EXIT_PENDING -> FLAT
                 |                     |
                 v                     v
             UNKNOWN               UNKNOWN          (timeout; resolved via positions_get)

Rules (frozen IT3/IT5/IT6):
  - max 1 pending order per symbol (never blind retry).
  - UNKNOWN is reconciled from broker positions_get, never re-sent.
  - partial fill is a failure unless explicitly supported.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class ExecState(str, Enum):
    FLAT = "FLAT"
    ENTRY_PENDING = "ENTRY_PENDING"
    OPEN = "OPEN"
    EXIT_PENDING = "EXIT_PENDING"
    UNKNOWN = "UNKNOWN"


@dataclass
class SymbolEntry:
    symbol: str
    state: ExecState = ExecState.FLAT
    decision_id: Optional[str] = None
    order_ticket: Optional[str] = None
    requested_price: Optional[float] = None
    requested_time: Optional[float] = None  # monotonic, seconds
    quantity: float = 0.0
    side: str = "L"
    timeout_sec: float = 5.0
    last_broker_check: Optional[float] = None
    lifecycle: list = field(default_factory=list)  # ordered state history for audit

    def transition(self, to: ExecState, note: str = "") -> None:
        self.state = to
        self.lifecycle.append({"ts": time.time(), "state": to.value, "note": note})

    def is_pending(self) -> bool:
        return self.state in (ExecState.ENTRY_PENDING, ExecState.EXIT_PENDING, ExecState.UNKNOWN)


class ExecutionStateMachine:
    """Owns one SymbolEntry per symbol; enforces the one-in-flight contract."""

    def __init__(self, timeout_sec: float = 5.0, now_fn=time.time) -> None:
        self.timeout_sec = timeout_sec
        self._now = now_fn
        self._entries: Dict[str, SymbolEntry] = {}
        self._seq = 0

    def _entry(self, symbol: str) -> SymbolEntry:
        if symbol not in self._entries:
            self._entries[symbol] = SymbolEntry(symbol=symbol, timeout_sec=self.timeout_sec)
        return self._entries[symbol]

    # ---- queries ------------------------------------------------------
    def state_of(self, symbol: str) -> ExecState:
        return self._entry(symbol).state

    def can_enter(self, symbol: str) -> bool:
        return self._entry(symbol).state == ExecState.FLAT

    def is_open(self, symbol: str) -> bool:
        return self._entry(symbol).state == ExecState.OPEN

    def can_exit(self, symbol: str) -> bool:
        return self._entry(symbol).state == ExecState.OPEN

    def next_seq(self) -> int:
        return self._seq

    # ---- transitions ---------------------------------------------------
    def mark_sent_enter(self, symbol: str, decision_id: str, qty: float,
                        side: str, ticket: Optional[str]) -> SymbolEntry:
        e = self._entry(symbol)
        e.decision_id = decision_id
        e.quantity = qty
        e.side = side
        e.order_ticket = ticket
        e.requested_time = self._now()
        e.transition(ExecState.ENTRY_PENDING, "order_sent")
        self._seq += 1
        return e

    def mark_fill(self, symbol: str, ticket: Optional[str], filled_qty: float) -> None:
        e = self._entry(symbol)
        e.order_ticket = ticket
        e.transition(ExecState.OPEN if e.state in (ExecState.ENTRY_PENDING, ExecState.UNKNOWN) else e.state,
                     f"fill qty={filled_qty}")

    def mark_sent_exit(self, symbol: str, qty: float, side: str) -> None:
        e = self._entry(symbol)
        e.quantity = qty
        e.side = side
        e.requested_time = self._now()
        e.transition(ExecState.EXIT_PENDING, "exit_sent")
        self._seq += 1

    def mark_closed(self, symbol: str) -> None:
        e = self._entry(symbol)
        e.transition(ExecState.FLAT, "closed")
        e.decision_id = None
        e.order_ticket = None
        e.quantity = 0.0

    def mark_reject(self, symbol: str, reason: str) -> None:
        e = self._entry(symbol)
        e.transition(ExecState.FLAT if e.state == ExecState.ENTRY_PENDING else ExecState.FLAT,
                     f"reject:{reason}")
        e.decision_id = None

    def mark_exit_reject(self, symbol: str, reason: str) -> None:
        """A failed close leaves the position OPEN (never blind-retry, reconcile later)."""
        e = self._entry(symbol)
        e.transition(ExecState.OPEN, f"exit_reject:{reason}")

    def mark_unknown(self, symbol: str) -> None:
        e = self._entry(symbol)
        e.transition(ExecState.UNKNOWN, "timeout")

    # ---- timeout / reconcile ------------------------------------------
    def check_timeout(self) -> Dict[str, SymbolEntry]:
        """Return symbols whose pending order has exceeded timeout_sec -> UNKNOWN."""
        timed = {}
        for sym, e in self._entries.items():
            if e.state in (ExecState.ENTRY_PENDING, ExecState.EXIT_PENDING) and e.requested_time is not None:
                if self._now() - e.requested_time >= e.timeout_sec and e.last_broker_check is None:
                    timed[sym] = e
        return timed

    def reconcile_from_broker(self, symbol: str, broker_has_position: bool) -> None:
        """Resolve an UNKNOWN by broker truth (positions_get)."""
        e = self._entry(symbol)
        e.last_broker_check = self._now()
        if broker_has_position:
            e.transition(ExecState.OPEN, "reconciled:found")
        else:
            e.transition(ExecState.FLAT, "reconciled:absent")

    # ---- introspection -------------------------------------------------
    def snapshot(self) -> Dict[str, Dict[str, Any]]:
        return {sym: {"state": e.state.value, "symbol": sym, "id_decision": e.decision_id}
                for sym, e in self._entries.items()}

    def reset(self) -> None:
        self._entries.clear()
        self._seq = 0