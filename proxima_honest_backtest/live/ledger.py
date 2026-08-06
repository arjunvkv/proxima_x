"""Restart ledger — durable decision/position state so a crash never double-enters.

A crash after ORDER_SENT (before BROKER_FILL flush) must not, on restart, re-enter
Tokyo H0. Recovery derives state from TWO sources and takes the union:

  1. the JSONL event stream (replayed by event_seq) — what the engine intended,
  2. MT5 positions_get() by magic range — what the broker actually holds.

Invariant (apples-to-apples):
    broker OPEN position (same magic + symbol)  <==>  engine OPEN state.
    If the ledger shows an ENTER decision at bar T for symbol S, and the broker
    holds an open position for S, the engine MUST be OPEN for S at startup
    (final lifecycle completed) — never re-enter.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

from proxima_honest_backtest.live.events.schema import EventType, normalize_side


class Ledger:
    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path
        self.events: List[Dict[str, Any]] = []
        self._seq_entered: Dict[str, Dict[str, str]] = {}  # symbol -> {bar_ts: decision_id}
        self._open_decisions: Dict[str, str] = {}  # symbol -> decision_id (enter pending or open)
        self._filled_entries: Dict[str, Dict[str, Any]] = {}  # symbol -> {decision_id, entry_price, qty, side}

    def load(self, path: Optional[str] = None) -> int:
        path = path or self.path
        self.path = path
        if not path or not os.path.exists(path):
            return 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self.events.append(ev)
                self._ingest(ev)
        return len(self.events)

    def _ingest(self, ev: Dict[str, Any]) -> None:
        etype = ev.get("event_type") or ev.get("type")
        symbol = ev.get("symbol")
        did = ev.get("decision_id")
        if not symbol:
            return
        if etype == "DECISION" and ev.get("action") in ("ENTER", "ENTER_LONG", "ENTER_SHORT"):
            bar = ev.get("bar_ts_utc")
            if bar:
                self._seq_entered.setdefault(symbol, {}).setdefault(bar, did or "")
        elif etype == "ORDER_SENT":
            if did:
                self._open_decisions[symbol] = did
        elif etype == "BROKER_FILL":
            if did:
                # only track enter-side fills as open; exit-side fills close it
                kind = did.split("|")[-1]
                self._filled_entries[symbol] = {
                    "decision_id": did,
                    "entry_price": ev.get("fill_price"),
                    "side": normalize_side(ev.get("side", "")),
                    "quantity": ev.get("filled_quantity"),
                }
                self._open_decisions.pop(symbol, None)
        elif etype == "BROKER_REJECT":
            if did and self._open_decisions.get(symbol) == did:
                self._open_decisions.pop(symbol, None)

    @property
    def last_seq(self) -> int:
        """Highest event_seq — emit continuation must start above this."""
        best = 0
        for ev in self.events:
            s = ev.get("event_seq")
            if isinstance(s, (int, float)):
                best = max(best, int(s))
        return best

    def entered_decisions(self) -> Dict[str, Dict[str, str]]:
        return dict(self._seq_entered)

    def has_entered(self, symbol: str, bar: str) -> bool:
        return symbol in self._seq_entered and bar in self._seq_entered[symbol]

    def engine_open_symbols(self) -> List[str]:
        open_syms = []
        for sym, ents in self._seq_entered.items():
            if not self._is_closed(sym, ents):
                open_syms.append(sym)
        return open_syms

    def _is_closed(self, symbol: str, ents: Dict[str, str]) -> bool:
        # naive: if any EXIT lifecycle present for these decisions -> closed
        for ev in self.events:
            if (ev.get("symbol") == symbol
                    and (ev.get("event_type") or ev.get("type")) in ("BROKER_FILL",)
                    and ev.get("decision_id")
                    and str(ev.get("decision_id")).rstrip().endswith(("|EXIT", "X"))):
                return True
        return False

    def pending_enter_decision_ids(self) -> Dict[str, str]:
        return dict(self._open_decisions)


class RecoveryChecker:
    """Merges ledger + broker positions into a safe startup snapshot."""

    def __init__(self, mt5: Any, magic_base: int, pairs: List[str]) -> None:
        self.mt5 = mt5
        self.magic_base = magic_base
        self.pairs = pairs

    def broker_open(self) -> Dict[str, Dict[str, Any]]:
        out = {}
        for pos in (self.mt5.positions_get() or []):
            if self.magic_base <= pos.magic < self.magic_base + len(self.pairs):
                out[pos.symbol] = {
                    "ticket": int(pos.ticket),
                    "side": "L" if pos.type == 0 else "S",
                    "entry_price": float(pos.price_open),
                    "volume": float(pos.volume),
                }
        return out

    def reconcile(self, ledger: Ledger) -> Dict[str, Any]:
        broker = self.broker_open()
        engine_open = set(ledger.engine_open_symbols())
        broker_syms = set(broker.keys())
        missing = sorted(engine_open - broker_syms)   # engine says open, broker none
        extra = sorted(broker_syms - engine_open)     # broker open, engine didn't
        matched = sorted(broker_syms & engine_open)
        return {
            "broker_open": {s: broker[s] for s in sorted(broker)},
            "engine_open": sorted(engine_open),
            "matched": matched,
            "missing_in_broker": missing,
            "orphan_broker_positions": extra,
            "ok": not missing and not extra,
        }