"""Stub-gated re-export for the missing real module (live_ig_audit).

The original live_ig_audit was proven to have never been in git history (74 commits,
0 objects) and is tracked for rebuild (Track B). The gate in
proxima_ops.reality decides:
  * PROXIMA_REALITY_ENABLED=1 -> raises ImportError (real module expected).
  * else -> loads the explicit no-op stub from _stubs.py.
"""
import proxima_ops.reality._gate as _gate

if _gate.REALITY_ENABLED:
    raise ImportError(
        "proxima_ops.reality.live_ig_audit is not rebuilt yet but "
        "PROXIMA_REALITY_ENABLED=1. Either rebuild it or unset the flag. "
        "Refusing to silently run a no-op stub in production.")

from proxima_ops.reality._stubs import LiveInformationGainAudit  # noqa: F401, E402
