import time
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ExecutionRecord:
    timestamp: float = field(default_factory=time.time)
    signal_id: str = ""
    symbol: str = ""
    action: str = ""
    mof_state: str = ""
    mof_score: float = 0.0
    rf_drift: float = 0.0
    lifecycle_orphans: int = 0


class ExecutionFrequencyController:
    MIN_STABILIZATION_CYCLES = 3
    MAX_ACTUATIONS_PER_WINDOW = 2
    WINDOW_CYCLES = 10
    RF_DRIFT_THRESHOLD = 0.05
    MOF_RECOVERY_THRESHOLD = 0.35

    def __init__(self):
        self._executions: deque[ExecutionRecord] = deque(maxlen=100)
        self._cycle_count: int = 0
        self._stabilization_count: int = 0
        self._pre_execution_baseline: Optional[dict] = None

    def record_cycle(self):
        self._cycle_count += 1
        self._stabilization_count += 1

    def record_execution(self, record: ExecutionRecord):
        self._executions.append(record)
        self._stabilization_count = 0

    def set_pre_execution_baseline(self, baseline: dict):
        self._pre_execution_baseline = baseline

    @property
    def executions_in_window(self) -> int:
        cutoff = time.time() - (self.WINDOW_CYCLES * 60)
        return sum(1 for e in self._executions if hasattr(e, 'timestamp') and e.timestamp > cutoff)

    @property
    def stabilization_complete(self) -> bool:
        return self._stabilization_count >= self.MIN_STABILIZATION_CYCLES

    @property
    def within_frequency_budget(self) -> bool:
        return self.executions_in_window < self.MAX_ACTUATIONS_PER_WINDOW

    def check_rf_recovery(self, current_drift: float) -> bool:
        return current_drift <= self.RF_DRIFT_THRESHOLD

    def check_mof_recovery(self, current_mof_score: float) -> bool:
        if self._pre_execution_baseline is None:
            return True
        pre_mof = self._pre_execution_baseline.get("mof_score", 0.0)
        min_recovery = max(self.MOF_RECOVERY_THRESHOLD, pre_mof * 0.85)
        return current_mof_score >= min_recovery

    def can_arm(self, current_drift: float, current_mof_score: float) -> tuple[bool, str]:
        if not self.stabilization_complete:
            return False, f"Stabilization not complete ({self._stabilization_count}/{self.MIN_STABILIZATION_CYCLES} cycles)"
        if not self.within_frequency_budget:
            return False, f"Frequency budget exceeded ({self.executions_in_window}/{self.MAX_ACTUATIONS_PER_WINDOW} in {self.WINDOW_CYCLES} cycles)"
        if not self.check_rf_recovery(current_drift):
            return False, f"RF drift {current_drift:.4f} exceeds threshold {self.RF_DRIFT_THRESHOLD}"
        if not self.check_mof_recovery(current_mof_score):
            return False, f"MOF score {current_mof_score:.4f} below recovery threshold"
        return True, "All constraints satisfied"

    def describe(self) -> dict:
        return {
            "cycle_count": self._cycle_count,
            "stabilization_count": self._stabilization_count,
            "stabilization_complete": self.stabilization_complete,
            "executions_in_window": self.executions_in_window,
            "window_capacity": self.MAX_ACTUATIONS_PER_WINDOW,
            "window_size_cycles": self.WINDOW_CYCLES,
            "within_frequency_budget": self.within_frequency_budget,
            "rf_drift_threshold": self.RF_DRIFT_THRESHOLD,
            "mof_recovery_threshold": self.MOF_RECOVERY_THRESHOLD,
            "total_executions": len(self._executions),
            "latest_execution": self._executions[-1] if self._executions else None,
        }
