"""Volume Expansion Layer — post-governor execution admission gate.

Controls trade frequency without modifying CORE, ERL, confirm gate, or governor.
Positioned as final gate before MT5 order submission."""
import logging
from collections import deque

logger = logging.getLogger("proxima_ops.execution.vel")

SYMBOLS = ["EURUSD", "GBPUSD", "EURJPY", "USDJPY"]


class VolumeExpansionLayer:
    """Post-decision execution admission control."""

    def __init__(self):
        self._last_exec_cycle: dict[str, int] = {s: 0 for s in SYMBOLS}
        self._exec_history: dict[str, deque] = {
            s: deque(maxlen=20) for s in SYMBOLS
        }
        self._cycle_count = 0

    def record_cycle(self):
        self._cycle_count += 1

    def record_execution(self, symbol: str):
        self._last_exec_cycle[symbol] = self._cycle_count
        self._exec_history[symbol].append(self._cycle_count)

    def should_allow_execution(self, symbol: str, direction: str,
                                staircase_phase: int = 1) -> tuple[bool, str]:
        """Check if execution is allowed based on VEL rules.
        Returns (allow: bool, reason: str)."""

        # 1. Temporal spacing: min cycles since last execution per symbol
        min_gap = 1
        if self._cycle_count - self._last_exec_cycle.get(symbol, 0) < min_gap:
            return False, f"temporal_spacing: {self._cycle_count} - {self._last_exec_cycle[symbol]} < {min_gap}"

        # 2. Exposure smoothing: phase-dependent lookback
        phase_lookback = {1: 5, 2: 3, 3: 2, 4: 1}
        n_lookback = phase_lookback.get(staircase_phase, 5)
        recent = [c for c in self._exec_history[symbol]
                  if self._cycle_count - c <= n_lookback]
        if recent:
            return False, (f"exposure_smoothing: {len(recent)} execs in last "
                           f"{n_lookback} cycles (phase={staircase_phase})")

        # 3. Burst prevention: max 1 exec per 10-cycle window
        recent_burst = [c for c in self._exec_history[symbol]
                        if self._cycle_count - c <= 10]
        if len(recent_burst) >= 1:
            return False, f"burst_prevention: {len(recent_burst)} execs in last 10 cycles"

        return True, "allowed"

    def describe(self) -> dict:
        return {
            "cycle": self._cycle_count,
            "last_exec_cycles": dict(self._last_exec_cycle),
            "exec_counts": {s: len(h) for s, h in self._exec_history.items()},
        }
