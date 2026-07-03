import time
import json
import os
import logging
from collections import deque, Counter
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class BehaviorRecord:
    timestamp: float = field(default_factory=time.time)
    cycle: int = 0
    state: str = ""
    elapsed_in_state: float = 0.0
    transition_type: str = ""
    execution_authorized: bool = False
    edge_eligible_count: int = 0


class ExecutionBehaviorAnalyzer:
    MAX_HISTORY = 500
    ARMING_BIAS_WARN = 0.60
    ARMING_BIAS_CRIT = 0.80
    LOCKING_BIAS_WARN = 0.15
    LOCKING_BIAS_CRIT = 0.30
    EXECUTION_FREQ_WARN = 0.20
    EXECUTION_FREQ_CRIT = 0.35

    def __init__(self, state_dir: str = None):
        self._history: deque[BehaviorRecord] = deque(maxlen=self.MAX_HISTORY)
        self._state_dir = state_dir or os.path.join("state", "behavior_audit_logs")
        os.makedirs(self._state_dir, exist_ok=True)

    def record(self, record: BehaviorRecord):
        self._history.append(record)
        self._save_record(record)

    def _save_record(self, rec: BehaviorRecord):
        path = os.path.join(self._state_dir, f"behavior_{rec.cycle:04d}_{int(rec.timestamp)}.json")
        with open(path, "w") as f:
            json.dump(rec.__dict__, f, indent=2, default=str)

    def state_distribution(self) -> dict:
        if not self._history:
            return {}
        states = Counter(r.state for r in self._history)
        total = len(self._history)
        return {s: {"count": c, "ratio": round(c / total, 4)} for s, c in states.most_common()}

    def arming_bias(self) -> dict:
        if len(self._history) < 5:
            return {"status": "INSUFFICIENT_DATA", "samples": len(self._history)}
        armed_count = sum(1 for r in self._history if r.state == "ARMED")
        total = len(self._history)
        arming_ratio = armed_count / total
        return {
            "arming_ratio": round(arming_ratio, 4),
            "armed_count": armed_count,
            "total_cycles": total,
            "status": "NORMAL" if arming_ratio < self.ARMING_BIAS_WARN
            else "WARNING" if arming_ratio < self.ARMING_BIAS_CRIT
            else "CRITICAL",
            "warn_threshold": self.ARMING_BIAS_WARN,
            "crit_threshold": self.ARMING_BIAS_CRIT,
        }

    def locking_bias(self) -> dict:
        if len(self._history) < 5:
            return {"status": "INSUFFICIENT_DATA", "samples": len(self._history)}
        locked_count = sum(1 for r in self._history if r.state == "LOCKED")
        cooldown_locked = sum(1 for r in self._history if r.transition_type == "COOLDOWN_TO_LOCKED")
        total = len(self._history)
        locking_ratio = locked_count / total
        cooldown_fail_rate = cooldown_locked / total if total > 0 else 0
        return {
            "locking_ratio": round(locking_ratio, 4),
            "cooldown_fail_rate": round(cooldown_fail_rate, 4),
            "locked_count": locked_count,
            "cooldown_locked": cooldown_locked,
            "status": "NORMAL" if locking_ratio < self.LOCKING_BIAS_WARN
            else "WARNING" if locking_ratio < self.LOCKING_BIAS_CRIT
            else "CRITICAL",
        }

    def execution_frequency_analysis(self) -> dict:
        if len(self._history) < 10:
            return {"status": "INSUFFICIENT_DATA", "samples": len(self._history)}
        exec_cycles = sum(1 for r in self._history if r.execution_authorized)
        total = len(self._history)
        exec_ratio = exec_cycles / total
        return {
            "execution_ratio": round(exec_ratio, 4),
            "execution_cycles": exec_cycles,
            "total_cycles": total,
            "status": "NORMAL" if exec_ratio < self.EXECUTION_FREQ_CRIT
            else "WARNING" if exec_ratio < self.EXECUTION_FREQ_WARN
            else "CRITICAL",
        }

    def summary(self) -> dict:
        return {
            "samples": len(self._history),
            "state_distribution": self.state_distribution(),
            "arming_bias": self.arming_bias(),
            "locking_bias": self.locking_bias(),
            "execution_frequency": self.execution_frequency_analysis(),
        }
