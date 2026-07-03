"""
TPI Flow Overlay — Live Signal Interface (Layer 7)

Usage:
    from layer7.get_tpi_signal import get_tpi_signal

    signal = get_tpi_signal("EURUSD", existing_signal={"direction": 1})
    # Returns: { "symbol": "EURUSD", "tpi": 0.082, "direction": 1,
    #            "confidence": 0.082, "percentile": 92.0, "eligible": True,
    #            "alignment": "MATCH", "session_ok": True, ... }
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np

from data.tick_buffer import get_offline_tpi_signal, get_offline_tpi

# Symbols eligible for TPI overlay
TPI_ELIGIBLE = ["EURJPY", "EURUSD", "GBPJPY", "USDJPY", "XAUUSD"]
# p90 confidence threshold
P90 = 90.0
# Session definitions (UTC)
TPI_SESSION = {
    "EURJPY": ("London", 9, 17),
    "EURUSD": ("London", 9, 17),
    "GBPJPY": ("London", 9, 17),
    "USDJPY": ("NY", 13, 22),
    "XAUUSD": ("Overlap", 8, 20),
}


def _get_session(symbol):
    if symbol in TPI_SESSION:
        name, lo, hi = TPI_SESSION[symbol]
        return name, lo, hi
    return None, None, None


def get_tpi_signal(symbol, existing_signal=None, tick_idx=-1, clock=None):
    """Get the latest TPI direction signal for a symbol.

    Args:
        symbol: Instrument symbol (e.g. "EURUSD")
        existing_signal: Optional dict with "direction" key for alignment check
        tick_idx: Tick index for offline data (-1 = latest)
        clock: Optional CycleClock for temporal coherence (P3.3)

    Returns:
        dict with TPI signal data, or None if no data available
    """
    raw = get_offline_tpi_signal(symbol, tick_idx, percentile=P90)
    if raw is None:
        return None

    session_name, session_lo, session_hi = _get_session(symbol)

    alignment = None
    if existing_signal and "direction" in existing_signal:
        ex_dir = existing_signal["direction"]
        if ex_dir == 0:
            alignment = "NEUTRAL"
        elif raw["direction"] == ex_dir:
            alignment = "MATCH"
        else:
            alignment = "CONFLICT"

    return {
        "symbol": symbol,
        "tpi": round(raw["tpi"], 5),
        "direction": raw["direction"],
        "direction_label": "LONG" if raw["direction"] == 1 else ("SHORT" if raw["direction"] == -1 else "FLAT"),
        "confidence": round(raw["confidence"], 5),
        "percentile": round(raw["percentile"], 1) if raw["percentile"] is not None else None,
        "eligible": raw["eligible"],
        "session_name": session_name,
        "session_hour": raw.get("session_hour"),
        "session_ok": raw["session_ok"],
        "n_ticks": raw["n_ticks"],
        "alignment": alignment,
    }


# ============================================================
# Live Mode: Persistent TickBuffer (stream-based, not poll-based)
# ============================================================

_LIVE_BUFFERS = {}  # symbol -> TickBuffer singleton

def get_or_init_buffer(symbol):
    """Get or create persistent TickBuffer for a symbol.
    Initializes by loading recent MT5 ticks on first call.
    """
    if symbol not in _LIVE_BUFFERS:
        from data.tick_buffer import TickBuffer
        buf = TickBuffer()
        loaded = buf.load_from_mt5(symbol)
        if loaded == 0:
            # Fallback: try loading with fewer ticks
            buf.load_from_mt5(symbol, num_ticks=100)
        _LIVE_BUFFERS[symbol] = buf
    return _LIVE_BUFFERS[symbol]


def feed_live_ticks(symbol):
    """Feed one live MT5 tick into the persistent buffer.
    Call this periodically (every 1-10s) per symbol.
    """
    buf = get_or_init_buffer(symbol)
    return buf.feed_from_mt5_tick(symbol)


def get_tpi_signal_live(symbol, existing_signal=None):
    """Get TPI signal from persistent TickBuffer (stream-driven).

    Uses the incremental TickBuffer — NOT mt5.copy_ticks_from().
    No timing skew, no duplicate windows, no hidden latency.

    Args:
        symbol: Instrument symbol (EURUSD or USDJPY)
        existing_signal: Optional dict with 'direction' for alignment check

    Returns:
        dict with tpi, direction, direction_label, confidence,
        n_ticks, alignment (if existing_signal provided), or None
    """
    buf = get_or_init_buffer(symbol)
    raw = buf.get_tpi(symbol)
    if raw is None:
        return None

    alignment = None
    if existing_signal and "direction" in existing_signal:
        ex_dir = existing_signal["direction"]
        if ex_dir == 0:
            alignment = "NEUTRAL"
        elif raw["direction"] == ex_dir:
            alignment = "MATCH"
        else:
            alignment = "CONFLICT"

    return {
        "symbol": symbol,
        "tpi": raw["tpi"],
        "direction": raw["direction"],
        "direction_label": "LONG" if raw["direction"] == 1 else ("SHORT" if raw["direction"] == -1 else "FLAT"),
        "confidence": raw["confidence"],
        "n_ticks": raw["n_ticks"],
        "alignment": alignment,
    }


def print_tpi_signal(symbol):
    sig = get_tpi_signal(symbol)
    if sig is None:
        print(f"{symbol}: No TPI signal (insufficient data)")
        return
    print(f"\n{symbol} TPI Signal:")
    for k, v in sig.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    for sym in TPI_ELIGIBLE:
        print_tpi_signal(sym)
