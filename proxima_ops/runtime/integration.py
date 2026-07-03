"""SOVEREIGN_EXECUTION_RUNTIME — Batch 10 Integration.

Combines all 5 runtime modules into single sovereign execution process.
"""

import json
import logging
import time

logger = logging.getLogger("proxima_ops.runtime.integration")

try:
    from .edek import EventDrivenExecutionKernel
    from .sfpf import StateFreezingFix
    from .etd import ExecutionTriggerDemux
    from .mrbl import MT5RuntimeBinding
    from .sel import SovereignExecutionLoop
    _HAS_MODULES = True
except ImportError as e:
    _HAS_MODULES = False
    logger.warning("Some runtime modules unavailable: %s", e)


class SovereignRuntimeManager:
    """Manages the full sovereign execution runtime lifecycle."""

    def __init__(self, mt5_connector=None, symbol_universe: list = None):
        self._edek = EventDrivenExecutionKernel(mt5_connector) if _HAS_MODULES else None
        self._sfpf = StateFreezingFix() if _HAS_MODULES else None
        self._etd = ExecutionTriggerDemux() if _HAS_MODULES else None
        self._mrbl = MT5RuntimeBinding(mt5_connector) if _HAS_MODULES else None
        self._ses = None
        self._ecl = None
        self._efk = None
        self._sel = None
        self._symbol_universe = symbol_universe or []
        self._started = False

    def initialize(self, ses, ecl, efk):
        """Wire sovereignty modules into the runtime."""
        self._ses = ses
        self._ecl = ecl
        self._efk = efk
        if _HAS_MODULES and self._edek and self._sfpf and self._etd and self._mrbl:
            self._sel = SovereignExecutionLoop(
                ses, ecl, efk, self._etd, self._edek,
                self._sfpf, self._mrbl, self._symbol_universe
            )
            restored = self._sfpf.load()
            if restored.get("restored"):
                logger.info("Sovereignty state restored from %s", restored.get("restored_at"))
            return True
        return False

    def process_tick(self, symbol: str, tick: dict) -> dict:
        if not self._sel:
            return {"tick_processed": False}
        return self._sel.process_tick(symbol, tick)

    def process_cycle(self, **kwargs) -> dict:
        if not self._sel:
            return {"cycle_processed": False}
        return self._sel.process_cycle(**kwargs)

    def start(self) -> bool:
        if self._sel and self._sel.start():
            self._started = True
            return True
        return False

    def stop(self) -> bool:
        if self._sel:
            self._sel.stop()
        self._started = False
        return True

    def get_status(self) -> dict:
        sel_state = self._sel.get_state() if self._sel else {}
        mrbl_status = self._mrbl.get_status() if self._mrbl else {}
        return {
            "runtime_active": self._started,
            "total_ticks": sel_state.get("total_ticks_processed", 0),
            "total_cycles": sel_state.get("total_cycles_processed", 0),
            "total_orders": sel_state.get("total_orders_emitted", 0),
            "mt5_connected": mrbl_status.get("connected", False),
            "last_tick": sel_state.get("last_tick_time"),
            "last_order": sel_state.get("last_order_time"),
            "uptime": sel_state.get("uptime_seconds", 0),
        }
