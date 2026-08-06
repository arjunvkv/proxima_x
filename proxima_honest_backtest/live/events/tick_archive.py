"""Owner-process tick capture helper.

Process A (the ONE MT5 owner) reads copy_ticks_from for the ship pairs and emits
TICK events through the shared EventEmitter so downstream ReconMonitor / tick
analyzer have a stateless market-evidence stream. This file has NO MT5 import at
module scope so it stays import-safe offline.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def extract_tick_bars(tick: Any) -> Dict[str, Any]:
    """Flatten an MT5 tick tuple into a TICK payload.

    Only fields below are consumed; everything else is dropped. Handles either a
    named-tuple-like object or a plain dict (offline FakeTick).
    """
    if hasattr(tick, "time_msc"):
        return {
            "tick_ts_utc": int(tick.time_msc),
            "bid": float(getattr(tick, "bid", 0.0) or 0.0),
            "ask": float(getattr(tick, "ask", 0.0) or 0.0),
            "last": float(getattr(tick, "last", 0.0) or 0.0),
            "flags": int(getattr(tick, "flags", 0) or 0),
        }
    if isinstance(tick, dict):
        return {
            "time_msc": int(tick.get("time_msc", 0)),
            "bid": float(tick.get("bid", 0.0)),
            "ask": float(tick.get("ask", 0.0)),
            "last": float(tick.get("last", 0.0)),
            "flags": int(tick.get("flags", 0)),
        }
    return {"time_msc": 0, "bid": 0.0, "ask": 0.0, "last": 0.0, "flags": 0}


def build_tick_context(ticks: List[Dict[str, Any]], symbol: str,
                       bar_open_ts_ms: int) -> Optional[Dict[str, Any]]:
    """First bid/ask after bar_open_ts_ms — the online executable-reality anchor."""
    for t in ticks:
        if t.get("time_ms") >= bar_open_ts_ms:
            return {"symbol": symbol, "first_bid": t["bid"], "first_ask": t["ask"]}
    return None