"""proxima_ops.reality — reality observability layer (C6 gate).

Imports the real submodules when PROXIMA_REALITY_ENABLED=1; loads explicit
no-op stubs otherwise. See _gate.py for the flag contract.
"""
from proxima_ops.reality import _gate  # noqa: F401  (side-effect: gate check + warning)
from proxima_ops.reality._gate import REALITY_ENABLED

__all__ = ["REALITY_ENABLED"]