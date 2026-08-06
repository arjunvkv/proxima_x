"""Canonical tick contract — the single normalized shape every tick source emits.

This is the heart of Phase 0 of the apples-to-apples alignment. Every producer
(live MT5 connector, replay archive, replay bars, MT5 history) has a slightly
different raw field naming. ``normalize_tick`` converts *any* of them into ONE
canonical dict so that the MVS engine consumes byte-identical ticks whether it
is running against live FTMO ticks or a recorded replay — making a tested
strategy reproduce live by construction (no code drift between modes).
"""
from __future__ import annotations

from typing import Any, Dict, Optional


# Canonical field contract. Every tick produced downstream carries these.
CANONICAL_FIELDS = (
    "symbol",
    "time_sec",        # seconds since epoch
    "time_msc",        # milliseconds since epoch
    "ts_ns",           # nanoseconds since epoch (canonical high-resolution clock)
    "bid",
    "ask",
    "mid",
    "spread",          # price units (ask - bid)
    "spread_pts",      # spread in points (for consistency with MT5 display)
    "last",
    "volume",
    "volume_real",
    "flags",
    "point",           # tick size in price units
    "digits",          # symbol decimals
    "_source",         # which raw source produced this tick (diagnostic)
)


def _to_float(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _to_int(v: Any) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def normalize_tick(raw: Dict[str, Any], *, symbol: Optional[str] = None) -> Dict[str, Any]:
    """Normalize one raw tick dict from any producer into the canonical shape.

    Accepts the known variants produced across this repo:
      * live MT5 connector:    {symbol, bid, ask, spread(pts), time, time_msc, flags, volume, _real_volume}
      * replay archive rows:   {symbol, time_sec, time_msc, timestamp_ns, bid, ask, last, volume, volume_real, flags}
      * MT5 history ticks:     {symbol, time_msc, bid, ask, last, volume, flags}
      * MVS TickLoader input:  {symbol, time_msc, time, bid, ask}
    """
    sym = raw.get("symbol") or symbol or ""

    # Time resolution recovery: prefer ns, then msc, then seconds.
    timestamp_ns = raw.get("timestamp_ns")
    ts_ns = raw.get("ts_ns")
    if timestamp_ns:
        ts_ns = _to_int(timestamp_ns)
    if not ts_ns and raw.get("time_msc"):
        ts_ns = _to_int(raw["time_msc"]) * 1_000_000
    if not ts_ns and raw.get("time"):
        base = _to_int(raw["time"])
        # If time looks like milliseconds (13 digits) vs seconds (10 digits)
        ts_ns = base * 1_000_000 if base < 100_000_000_00 else base * 1_000
    if not ts_ns:
        ts_ns = 0

    time_msc = raw.get("time_msc") or _to_float((ts_ns or 0)) / 1_000_000.0
    if time_msc and _to_float(time_msc) < 100_000_000_00:
        # time_msc is actually seconds (10-digit) — upgrade to ms
        time_msc = _to_float(time_msc) * 1000.0
    time_sec = raw.get("time_sec") or raw.get("time")
    if not time_sec:
        time_sec = _to_float((ts_ns or 0)) / 1_000_000_000.0

    bid = _to_float(raw.get("bid", 0.0))
    ask = _to_float(raw.get("ask", 0.0))
    last = _to_float(raw.get("last", 0.0))
    if not last:
        last = bid
    point = _to_float(raw.get("point", 1e-5))
    # Canonical `spread` is ALWAYS in price units (ask - bid). No ambiguity
    # between the live connector (which stores points) and the archive (price
    # units): downstream always reads a deterministic price-unit spread.
    spread = max(ask - bid, point) if (ask and bid) else 0.0
    # Deterministic: quantize to point granularity so ask-bid float noise
    # (e.g. 162.008-162.0 = 0.008000000000009777) cannot leak downstream.
    if spread:
        spread = round(spread / point) * point
    # Spread in points: from explicit field, else derived from price spread.
    if raw.get("spread_points") or raw.get("spread_pts"):
        spread_pts = _to_float(raw["spread_points"] if raw.get("spread_points") else raw.get("spread_pts"))
    else:
        raw_spread = raw.get("spread")
        if raw_spread is not None and raw_spread > 1:
            spread_pts = _to_float(raw_spread)  # live connector stored points here
        else:
            spread_pts = max(1, int(spread / max(point, 1e-12))) if spread else 0

    mid = (bid + ask) * 0.5 if bid or ask else last

    tick = {
        "symbol": sym,
        "time_sec": _to_float(time_sec) if time_sec else int((ts_ns or 0) / 1_000_000_000),
        "time_msc": _to_float(time_msc) if time_msc else int((ts_ns or 0) / 1_000_000),
        "ts_ns": ts_ns,
        "bid": bid,
        "ask": ask,
        "mid": mid,
        "spread": spread,
        "spread_pts": spread_pts,
        "last": last,
        "volume": _to_float(raw.get("volume", 0.0)),
        "volume_real": _to_float(raw.get("volume_real", raw.get("real_volume", 0.0))),
        "flags": _to_int(raw.get("flags", 0)),
        "seq_num": _to_int(raw.get("seq_num", raw.get("_seq", 0))),
        "point": point,
        "digits": _to_int(raw.get("digits", 0)),
        "_source": raw.get("_source", "unknown"),
    }
    return tick