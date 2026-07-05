"""Module-level constants, flags, and top-level helpers for Proxima demo deployment.

All modules in this package are stateless — they receive the ProximaDemo instance
as their first parameter and never import from run_proxima_demo.py or from each other.
"""

import os
import sys
import logging

logger = logging.getLogger("proxima_demo")

# ── CLI flags ──────────────────────────────────────────────────────────────────
import tempfile as _tf
ACCEPTANCE_MODE = "--acceptance" in sys.argv
ACCEPTANCE_LOG_PATH = os.path.join(_tf.gettempdir(), "proxima_acceptance_only.log")

# ── Shutdown coordination ──────────────────────────────────────────────────────
_SHUTDOWN = False
RUNNING = True

# ── Environment config ─────────────────────────────────────────────────────────
PROXIMA_MAX_CYCLES = int(os.environ.get("PROXIMA_MAX_CYCLES", "0"))
_VALIDATION_ENABLED = os.environ.get("VALIDATION_ENABLED", "1") == "1"

# ── Tick dispatch thresholds ───────────────────────────────────────────────────
MIN_HOLD_TICKS_FLIP = 12
MIN_HOLD_TICKS_MIGRATION = 20
MAX_HOLD_TICKS = 200
EXPLORATION_TTL = 24
MICRO_VOL_LOOKBACK = 20


def compute_micro_volatility(symbol: str, rates: list, lookback: int = None) -> float:
    """Median-based micro-volatility for a symbol."""
    if lookback is None:
        lookback = 20
    if not rates or len(rates) < lookback:
        return 0.0001
    _slice = rates[-lookback:]
    _deltas = []
    for i in range(1, len(_slice)):
        _prev = _slice[i-1].get("close", _slice[i-1].get("open", 0))
        _cur = _slice[i].get("close", _slice[i].get("open", 0))
        _deltas.append(abs(_cur - _prev))
    if not _deltas:
        return 0.0001
    _deltas.sort()
    _med = _deltas[len(_deltas) // 2]
    point_val = 0.01 if "JPY" in symbol else (0.1 if "XAU" in symbol or "XAG" in symbol else 0.0001)
    return max(_med / max(point_val, 1e-9), 0.0001)


def signal_handler(sig, frame):
    global RUNNING, _SHUTDOWN
    logger.info("[SHUTDOWN] Signal %s received", sig)
    RUNNING = False
    _SHUTDOWN = True
