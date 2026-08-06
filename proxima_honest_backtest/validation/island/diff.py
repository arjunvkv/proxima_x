"""Level-2 execution diff — sim events vs broker events, normalized to pips.

Decisions MUST be identical (decision stream == execution stream). Prices are
compared as slippage in pips from the requested anchor (never raw equality),
side normalized to L/S, timestamps to UTC epoch ms.
"""
from __future__ import annotations

from typing import Any, Dict, List

from proxima_honest_backtest.live.events.schema import normalize_side


def _pip_size(symbol: str) -> float:
    return 0.01 if "JPY" in symbol else 0.0001


def _epoch_ms(iso) -> float:
    if iso is None:
        return 0.0
    if isinstance(iso, (int, float)):
        return float(iso)
    from datetime import datetime
    try:
        dt = datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
        return dt.timestamp() * 1000.0
    except Exception:
        return 0.0


def extract_enters(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [e for e in events if e.get("event_type") == "DECISION" and e.get("action") == "ENTER"]


def extract_fills(events: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {e.get("decision_id"): e for e in events if e.get("event_type") == "BROKER_FILL"}


def decision_key(e: Dict[str, Any]) -> tuple:
    return (
        _epoch_ms(e.get("bar_ts_utc")),
        e.get("symbol"),
        normalize_side(e.get("side", "")),
    )


def compare_level2(sim_events: List[Dict[str, Any]],
                   broker_events: List[Dict[str, Any]],
                   slip_threshold_pips: float = 1.5) -> Dict[str, Any]:
    sim_fills = extract_fills(sim_events)
    bro_fills = extract_fills(broker_events)
    sim_ids = set(sim_fills)
    bro_ids = set(bro_fills)

    missing_broker = sorted(sim_ids - bro_ids)
    extra_broker = sorted(bro_ids - sim_ids)

    slippages = []
    mismatches = []
    common = sim_ids & bro_ids
    for did in common:
        s = sim_fills[did]
        b = bro_fills[did]
        sym = b.get("symbol") or s.get("symbol")
        pip = _pip_size(sym or "EURUSD")
        s_fill = s.get("fill_price")
        b_fill = b.get("fill_price")
        if s_fill is None or b_fill is None:
            continue
        slip_pips = abs(b_fill - s_fill) / pip
        slippages.append(slip_pips)
        if slip_pips > slip_threshold_pips:
            mismatches.append({"decision_id": did, "symbol": sym, "slip_pips": round(slip_pips, 3)})

    # side/decision parity
    sim_dec = {decision_key(e): e for e in extract_enters(sim_events)}
    bro_dec = {decision_key(e): e for e in extract_enters(broker_events)}
    decision_mismatch = [k for k in sim_dec if k not in bro_dec] or \
        [k for k in bro_dec if k not in sim_dec]

    return {
        "n_sim_fills": len(sim_ids),
        "n_broker_fills": len(bro_ids),
        "missing_broker": missing_broker[:10],
        "extra_broker": extra_broker[:10],
        "slippage_beyond_threshold": mismatches[:10],
        "max_slippage_pips": round(max(slippages), 3) if slippages else 0.0,
        "avg_slippage_pips": round(sum(slippages) / len(slippages), 3) if slippages else 0.0,
        "decision_mismatches": decision_mismatch[:10],
        "passed": (not missing_broker and not extra_broker
                   and not mismatches and not decision_mismatch),
    }