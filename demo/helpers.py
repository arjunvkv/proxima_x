"""Small helper methods extracted from ProximaDemo.

All functions are stateless — they take `demo` as first parameter and
read from or write to `demo.*` attributes explicitly. No cross-module imports.
"""

import json
import os
import time
import logging
from datetime import datetime

import numpy as np

from proxima_ops.config.settings import SETTINGS

logger = logging.getLogger("proxima_demo")


def compute_observer_state(demo, tpi_confidence: float, persistence_streak: int,
                           curvature_state: str, normalized_entropy: float) -> dict:
    from mvs.observer.observer_features import (
        normalize_tpi, persistence_ratio_from_streak,
        curvature_strength_from_state, compute_entropy_alignment,
        compute_confidence, state_from_confidence,
    )
    _ntpi = normalize_tpi(tpi_confidence)
    _pers = persistence_ratio_from_streak(persistence_streak)
    _curv = curvature_strength_from_state(curvature_state)
    _ent = compute_entropy_alignment(normalized_entropy, max_entropy=1.0)
    confidence = compute_confidence(_ntpi, _pers, _curv, _ent)
    state = state_from_confidence(confidence)
    return {"observer_state": state, "observer_confidence": float(confidence),
            "reality_score": min(1.0, max(0.0, confidence + 0.1))}


def freq_rates_provider(demo, symbol: str) -> list:
    return demo.mt5.get_rates(symbol, count=150, timeframe="H1")


def bars_elapsed(demo, entry_bar_time, symbol) -> int:
    if not entry_bar_time:
        return -1
    elapsed_secs = demo._now_ts() - entry_bar_time
    return max(0, int(elapsed_secs // 300))


def atomic_write_json(demo, path: str, data) -> None:
    try:
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception as e:
        logger.error(f"Atomic write failed for {path}: {e}")


def save_active_positions_metadata(demo):
    meta_file = os.path.join(os.path.dirname(SETTINGS.db_path), "active_positions_metadata.json")
    atomic_write_json(demo, meta_file, demo._active_positions_metadata)
    guard_file = os.path.join(os.path.dirname(SETTINGS.db_path), "applied_feedback_tickets.json")
    atomic_write_json(demo, guard_file, list(demo._applied_feedback_tickets))


def load_active_positions_metadata(demo):
    meta_file = os.path.join(os.path.dirname(SETTINGS.db_path), "active_positions_metadata.json")
    if os.path.exists(meta_file):
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            demo._active_positions_metadata = {int(k): v for k, v in data.items()}
        except Exception as e:
            logger.error(f"Error loading active positions metadata: {e}")
    guard_file = os.path.join(os.path.dirname(SETTINGS.db_path), "applied_feedback_tickets.json")
    if os.path.exists(guard_file):
        try:
            with open(guard_file, "r", encoding="utf-8") as f:
                demo._applied_feedback_tickets = set(json.load(f))
        except Exception as e:
            logger.error(f"Error loading idempotency guard: {e}")


def update_symbol_trust_from_bd(demo, ticket: int, symbol: str) -> None:
    from proxima_ops.reality.outcome_ledger import OutcomeLedger
    if ticket in demo._applied_feedback_tickets:
        return
    entry_data = demo._active_positions_metadata.get(ticket, {})
    entry_price = entry_data.get("entry_price", 0.0)
    min_price = demo._active_positions_metadata.get(ticket, {}).get("min_price", entry_price)
    max_price = demo._active_positions_metadata.get(ticket, {}).get("max_price", entry_price)
    direction = entry_data.get("direction", 0)
    if direction == 0 or entry_price == 0.0:
        return
    mfe_peak = (max_price - entry_price) if direction == 1 else (entry_price - min_price)
    outcome_pts = mfe_peak
    ticket_outcome = demo._outcome_ledger.get_ticket_outcome(ticket)
    if ticket_outcome:
        outcome_pts = ticket_outcome.get("h20", {}).get("return", mfe_peak)
    pnl = outcome_pts if outcome_pts != 0 else mfe_peak
    demo._symbol_trust.update(symbol, pnl)
    demo._applied_feedback_tickets.add(ticket)


def now_ts(demo) -> float:
    if demo._clock:
        return demo._clock.time()
    return time.time()


def now_dt(demo):
    if demo._clock:
        return demo._clock.now()
    return datetime.now()


def emit_reconciliation_event(demo, event_type: str, ticket: int, symbol: str,
                               details: dict = None) -> None:
    event = {
        "type": event_type,
        "ticket": ticket,
        "symbol": symbol or "?",
        "cycle_id": demo._cycle_id,
        "timestamp": time.time(),
        "details": details or {},
    }
    demo._reconciliation_events.append(event)
    if hasattr(demo, 'exec_stats') and demo.exec_stats:
        demo.exec_stats.record_reconciliation(event_type, symbol or "?", ticket, details)
    if len(demo._reconciliation_events) > 1000:
        demo._reconciliation_events = demo._reconciliation_events[-500:]


def select_balanced_top3(demo, candidates, execution_plan, n=3):
    selected = []
    for sym in candidates:
        if len(selected) >= n:
            break
        selected.append(sym)
    return set(selected[:n])


def shadow_regime(demo, sym: str) -> str:
    ed = getattr(demo, '_shadow_state', {})
    state = ed.get(sym, {})
    spread_p95 = state.get("spread_p95", 0)
    spread_p50 = state.get("spread_p50", 0)
    entropy = state.get("entropy", 0.5)

    if spread_p95 > 0 and spread_p50 > 0 and spread_p95 > spread_p50 * 2:
        return "WIDE"
    elif entropy > 0.85:
        return "CHAOTIC"
    elif entropy < 0.45:
        return "COMPRESSED"
    else:
        return "NORMAL"
