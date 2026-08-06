"""proxima_ops/reality/_gate.py — explicit reality-layer gate (C6).

The proxima_ops.reality.* modules (OutcomeLedger, LayerPruner, ImpulseGraph,
etc.) were proven never to exist in git history (74 commits, 0 objects) and
are being rebuilt deliberately (Track B). Until they exist:

  * PROXIMA_REALITY_ENABLED unset or "0"  -> explicit no-op stubs load, demo
    runs in degraded (paper) mode. Stubs are LOUD at import: they log a
    warning, they do not pretend to be real.
  * PROXIMA_REALITY_ENABLED = "1"         -> the stub submodules raise
    ImportError. Real modules are expected here; a silent stub would swallow
    a genuine production failure. This is the GPT-7 rule: explicit flag,
    NEVER a broad try/except.
"""
import os
import logging

logger = logging.getLogger("proxima_ops.reality")

REALITY_ENABLED = os.environ.get("PROXIMA_REALITY_ENABLED", "0").strip().lower() in (
    "1", "true", "yes", "on")

if not REALITY_ENABLED:
    logger.warning(
        "proxima_ops.reality layer DISABLED (PROXIMA_REALITY_ENABLED unset/0): "
        "loading explicit no-op stubs. Set PROXIMA_REALITY_ENABLED=1 once the "
        "reality modules are rebuilt.")
