"""Single JSONL event writer (append-only, monotonic event_seq).

Used by the LIVE path (Process A), the offline island (FakeBroker), and the
backtest/paper replay — one schema everywhere. Emitter is thread-local safe for
a single writer; ReconMonitor only READS the stream and is stateless.
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from proxima_honest_backtest.live.events.schema import (
    ENVELOPE_FIELDS,
    EventType,
    TYPE_FIELDS,
)

_EVENT_TYPES = [e.value for e in EventType]

# Event types that must be flushed to disk immediately (broker-critical).
# Losing one of these on crash = a reconciliation hole.
_CRITICAL_EVENTS = {"ORDER_SENT", "BROKER_FILL", "BROKER_REJECT", "POSITION_SYNC"}


class EmitterMode(Enum):
    """Durability/throughput trade-off.

    ISLAND — offline validation: bounded process, close() flushes; batch writes
            are safe and give ~15x throughput.
    LIVE   — Process A: critical broker events flush immediately; only
            diagnostics (HEARTBEAT/TICK/BAR/DECISION) may buffer.
    """

    ISLAND = "island"
    LIVE = "live"


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


class EventEmitter:
    """Append-only JSONL stream. Not safe for multi-process write (single owner).

    Buffering: writes batch into an in-memory buffer and flush on `flush_every`
    events, on a 1s wall-clock interval (LIVE only), or on close(). ISLAND mode
    relies on close() for the final flush; LIVE mode force-flushes critical
    broker events immediately so a crash never loses an ORDER/FILL/REJECT.
    """

    def __init__(
        self,
        path: Optional[str] = None,
        strategy: str = "unknown",
        run_id: str = "local",
        validate: bool = True,
        mode: EmitterMode = EmitterMode.ISLAND,
        flush_every: Optional[int] = None,
        flush_interval_sec: float = 1.0,
    ) -> None:
        self.path = path
        self.strategy = strategy
        self.run_id = run_id
        self.validate = validate
        self.mode = EmitterMode(mode) if not isinstance(mode, EmitterMode) else mode
        if flush_every is None:
            flush_every = 256 if self.mode == EmitterMode.ISLAND else 512
        self.flush_every = max(1, int(flush_every))
        self.flush_interval_sec = flush_interval_sec
        self._seq = 0
        self._lock = threading.Lock()
        self._fh = None
        self._buffer: List[str] = []
        self._pending = 0
        self._last_flush_ts = 0.0
        if path:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            # reopen-append preserves monotonic seq across process restarts.
            self._seq = self._last_seq(path)
            self._fh = open(path, "a", encoding="utf-8")

    @staticmethod
    def _last_seq(path: str) -> int:
        last = 0
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        seq = int(json.loads(line).get("event_seq", 0))
                        last = max(last, seq)
                    except Exception:
                        continue
        except FileNotFoundError:
            pass
        return last

    def _should_flush(self, event_type: str, now: float) -> bool:
        if self.mode == EmitterMode.LIVE and event_type in _CRITICAL_EVENTS:
            return True
        if self._pending >= self.flush_every:
            return True
        if now - self._last_flush_ts >= self.flush_interval_sec:
            return True
        return False

    def emit(self, event_type: str, symbol: Optional[str] = None,
             decision_id: Optional[str] = None, **payload) -> Dict[str, Any]:
        with self._lock:
            self._seq += 1
            event: Dict[str, Any] = {
                "event_seq": self._seq,
                "event_type": event_type,
                "event_time_utc": _utcnow(),
                "strategy": self.strategy,
                "run_id": self.run_id,
                "symbol": symbol,
                "decision_id": decision_id,
            }
            event.update(payload)
            if self.validate:
                self._validate(event)
            if self._fh is not None:
                self._buffer.append(json.dumps(event, default=str) + "\n")
                self._pending += 1
                import time as _time
                if self._should_flush(event_type, _time.monotonic()):
                    self._flush()
            return event

    def flush(self) -> None:
        with self._lock:
            self._flush()

    def _flush(self) -> None:
        if self._fh is None or not self._buffer:
            return
        try:
            self._fh.write("".join(self._buffer))
            self._fh.flush()
            self._buffer.clear()
            self._pending = 0
            import time as _time
            self._last_flush_ts = _time.monotonic()
        except Exception:
            pass

    def _validate(self, event: Dict[str, Any]) -> None:
        et = event.get("event_type")
        if et not in _EVENT_TYPES:
            raise ValueError(f"unknown event_type {et!r}")
        for f in TYPE_FIELDS.get(EventType(et), []):
            if f not in event:
                raise ValueError(f"event {et} missing required field {f!r}")

    def close(self) -> None:
        if self._fh is not None:
            self._flush()
            self._fh.close()
            self._fh = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def replay_stream(path: str) -> List[Dict[str, Any]]:
    """Read a stream.jsonl back, sorted by event_seq (stable)."""
    events = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except Exception:
                continue
    events.sort(key=lambda e: e.get("event_seq", 0))
    return events


def stream_tail(path: str, n: int = 50) -> List[Dict[str, Any]]:
    events = replay_stream(path)
    return events[-n:]