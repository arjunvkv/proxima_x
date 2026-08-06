"""Immutable event schema for the tick-validation / reconciliation stream.

Frozen agreement (GPT IT5/IT7): one stateless stream.jsonl, every event carries
the shared envelope (event_seq, event_type, event_time_utc, strategy, run_id,
decision_id, symbol). Consumers must never mutate events; ReconMonitor re-derives
state purely from this stream keyed by event_seq.
"""
from __future__ import annotations

from enum import Enum


class EventType(str, Enum):
    BAR = "BAR"
    DECISION = "DECISION"
    ORDER_SENT = "ORDER_SENT"
    BROKER_FILL = "BROKER_FILL"
    BROKER_REJECT = "BROKER_REJECT"
    POSITION_SYNC = "POSITION_SYNC"
    TICK = "TICK"
    HEARTBEAT = "HEARTBEAT"

    @classmethod
    def all(cls):
        return [e.value for e in cls]


# Envelope fields present on EVERY event.
ENVELOPE_FIELDS = [
    "event_seq",
    "event_type",
    "event_time_utc",
    "strategy",
    "run_id",
    "symbol",      # nullable (BAR/HEARTBEAT)
    "decision_id",  # nullable (BAR/TICK/HEARTBEAT/POSITION_SYNC)
]

# Per-type required fields (union with envelope).
TYPE_FIELDS = {
    EventType.BAR: ["bar_ts_utc", "open", "timeframe", "source"],
    EventType.DECISION: ["action", "side", "quantity", "requested_price", "bar_ts_utc"],
    EventType.ORDER_SENT: ["send_time_utc", "requested_price", "quantity"],
    EventType.BROKER_FILL: ["broker_ticket", "fill_price", "fill_time_utc", "filled_quantity", "slippage_pips"],
    EventType.BROKER_REJECT: ["reject_reason", "broker_code"],
    EventType.POSITION_SYNC: ["engine_positions", "broker_positions", "reconciliation_status"],
    EventType.TICK: ["tick_ts_utc", "bid", "ask", "spread_pips", "flags"],
    EventType.HEARTBEAT: ["engine_status", "last_bar_ts", "mt5_connected"],
}

# Broker actions — the normalized enum subset actually emitted.
ACTION_ENTER = "ENTER"
ACTION_EXIT = "EXIT"

# Reconciliation statuses.
RECON_OK = "PASS"
RECON_FAIL = "FAIL"

# Executor side normalization.
SIDE_LONG = "L"
SIDE_SHORT = "S"


def normalize_side(side: str) -> str:
    up = str(side).upper()
    if up in ("LONG", "BUY", "L"):
        return SIDE_LONG
    if up in ("SHORT", "SELL", "S"):
        return SIDE_SHORT
    return up


import hashlib  # noqa: E402

_DFLT_SOURCE = "proxima"


def decision_id(symbol: str, bar_ts_utc, run_id: str, kind: str) -> str:
    """Deterministic correlation key across backtest/paper/live for ONE event.

    Uses ONLY intent (symbol, bar time, kind) — never price. The same decision
    in replay, paper and live must yield the SAME id so streams are diffable.
    """
    payload = f"{_DFLT_SOURCE}|{run_id}|{symbol}|{bar_ts_utc}|{kind}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]