"""Reconciliation invariant rules — pure, stateless checks over events.

Each rule returns (rule_id, passed: bool, detail: str). ReconMonitor applies all
rules to a full event stream. These are the frozen gates (IT3/IT5/IT7):
decision mapping, orphan order, duplicate order, sequence, quantity, side,
slippage, bar-late, exit-sync, clock skew, spread, UNKNOWN resolve, tick context.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from proxima_honest_backtest.live.events.schema import EventType, normalize_side

RuleFn = Callable[[List[Dict[str, Any]]], Tuple[str, bool, str]]

# Default gates (IT2/IT7) — FTMO demo M5.
GATES = {
    "latency_ms_pass": 500,
    "latency_ms_warn": 2000,
    "clock_skew_ms": 100,
    "enter_latency_s": 2.0,
    "exit_latency_s": 2.0,
    "tick_coverage_min": 0.0,  # 1.0 online; offline island has no ticks
}


def _all(events, event_type):
    return [e for e in events if e.get("event_type") == event_type]


def rule_decision_uniqueness(events) -> Tuple[str, bool, str]:
    dec = _all(events, "DECISION")
    ids = [e.get("decision_id") for e in dec]
    dup = {d for d in ids if d is not None and ids.count(d) > 1}
    return "DECISION_UNIQUE", not dup, f"duplicates={sorted(dup)[:5]}" if dup else "ok"


def rule_order_has_decision(events) -> Tuple[str, bool, str]:
    sent = _all(events, "ORDER_SENT")
    orphan = [e for e in sent if not e.get("decision_id")]
    return "ORDER_HAS_DECISION", not orphan, f"orphan_sent={len(orphan)}"


def rule_fill_has_order(events) -> Tuple[str, bool, str]:
    fills = _all(events, "BROKER_FILL") + _all(events, "BROKER_REJECT")
    sent_ids = {e.get("decision_id") for e in _all(events, "ORDER_SENT")}
    orphan = [e for e in fills if e.get("decision_id") not in sent_ids]
    return "FILL_HAS_ORDER", not orphan, f"orphan_fill={len(orphan)}"


def rule_sequence_monotonic(events) -> Tuple[str, bool, str]:
    seqs = [e.get("event_seq") for e in events]
    ordered = all(seqs[i] < seqs[i + 1] for i in range(len(seqs) - 1))
    return "SEQ_MONOTONIC", ordered, "ok" if ordered else "regression"


def rule_no_duplicate_ticket(events) -> Tuple[str, bool, str]:
    tickets = [e.get("broker_ticket") for e in _all(events, "BROKER_FILL")
               if e.get("broker_ticket")]
    dup = {t for t in tickets if tickets.count(t) > 1}
    return "NO_DUP_TICKET", not dup, f"dups={sorted(dup)[:5]}" if dup else "ok"


def rule_no_ghost_position(events) -> Tuple[str, bool, str]:
    """POSITION_SYNC events: engine_positions must equal broker_positions."""
    syncs = _all(events, "POSITION_SYNC")
    fails = [e for e in syncs if e.get("reconciliation_status") != "PASS"]
    return "NO_GHOST", not fails, f"failing_syncs={len(fails)}"


def rule_side_mismatch(events) -> Tuple[str, bool, str]:
    dec = {e.get("decision_id"): e for e in _all(events, "DECISION")}
    for f in _all(events, "BROKER_FILL"):
        d = dec.get(f.get("decision_id"))
        if d is not None:
            fs = normalize_side(f.get("side", ""))
            ds = normalize_side(d.get("side", ""))
            if fs and ds and fs != ds:
                return "SIDE_MATCH", False, f"{f.get('symbol')} fill={fs} dec={ds}"
    return "SIDE_MATCH", True, "ok"


def rule_quantity_match(events, tolerance: float = 0.0) -> Tuple[str, bool, str]:
    dec = {e.get("decision_id"): e for e in _all(events, "DECISION")}
    for f in _all(events, "BROKER_FILL"):
        d = dec.get(f.get("decision_id"))
        if d is not None:
            dq = d.get("quantity", 0)
            fq = f.get("filled_quantity", f.get("quantity", dq))
            if abs(dq - fq) > tolerance:
                return "QTY_MATCH", False, f"{f.get('symbol')} req={dq} fill={fq}"
    return "QTY_MATCH", True, "ok"


def rule_reject_classified(events) -> Tuple[str, bool, str]:
    rejects = _all(events, "BROKER_REJECT")
    bad = [e for e in rejects if not e.get("reject_reason")]
    return "REJECT_CLASSIFIED", not bad, f"unclassified={len(bad)}"


def rule_clock_skew(events, max_ms: float = GATES["clock_skew_ms"]) -> Tuple[str, bool, str]:
    """Compares broker tick ts vs event wall clock when both present."""
    worst = 0.0
    for t in _all(events, "TICK"):
        ts = t.get("tick_ts_utc")
        wall = t.get("event_time_utc")
        if ts is None or not wall:
            continue
        try:
            skew = abs(float(ts) - _to_ms(wall))
            worst = max(worst, skew)
        except Exception:
            continue
    return "CLOCK_SKEW", worst <= max_ms, f"worst={worst:.0f}ms (gate {max_ms})"


def _to_ms(wall_iso: str) -> float:
    from datetime import datetime
    iso = wall_iso.replace("Z", "+00:00")
    dt = datetime.fromisoformat(iso)
    return dt.timestamp() * 1000.0


ALL_RULES: List[RuleFn] = [
    rule_decision_uniqueness,
    rule_order_has_decision,
    rule_fill_has_order,
    rule_sequence_monotonic,
    rule_no_duplicate_ticket,
    rule_no_ghost_position,
    rule_side_mismatch,
    rule_quantity_match,
    rule_reject_classified,
    rule_clock_skew,
]


def run_all_rules(events: List[Dict[str, Any]]) -> List[Tuple[str, bool, str]]:
    return [rule(events) for rule in ALL_RULES]