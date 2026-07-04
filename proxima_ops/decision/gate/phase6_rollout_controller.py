from __future__ import annotations

from typing import Any


class Phase6RolloutController:
    SHADOW = "SHADOW"
    MICRO_CAPITAL = "MICRO_CAPITAL"
    FULL_LIVE = "FULL_LIVE"

    def __init__(self) -> None:
        self._state = self.SHADOW
        self._transition_log: list[dict] = []
        self._stability_window: list[bool] = []
        self._min_cycles_before_advance = 20

    @property
    def state(self) -> str:
        return self._state

    def _evaluate_advance_conditions(self, metrics: dict) -> bool:
        alignment = metrics.get("alignment", 0.0)
        rc_veto_rate = metrics.get("rc_veto_rate", 1.0)
        emd_score = metrics.get("emd_score", 1.0)
        mra_score = metrics.get("mra_score", 0.0)
        passes = (
            alignment >= 0.55 and
            rc_veto_rate <= 0.15 and
            emd_score <= 0.3 and
            mra_score >= 0.3
        )
        return passes

    def _check_rollback_conditions(self, metrics: dict) -> bool:
        alignment = metrics.get("alignment", 1.0)
        rc_veto_rate = metrics.get("rc_veto_rate", 0.0)
        return alignment < 0.40 or rc_veto_rate > 0.25

    def _record_transition(self, from_state: str, to_state: str, metrics: dict, reason: str) -> None:
        import time as _t
        entry = {
            "timestamp": _t.strftime("%Y-%m-%dT%H:%M:%SZ", _t.gmtime()),
            "from_state": from_state,
            "to_state": to_state,
            "reason": reason,
            "metrics_snapshot": {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()},
        }
        self._transition_log.append(entry)

    def evaluate(self, metrics: dict) -> dict:
        self._stability_window.append(self._evaluate_advance_conditions(metrics))
        if len(self._stability_window) > self._min_cycles_before_advance:
            self._stability_window.pop(0)

        if self._check_rollback_conditions(metrics):
            prev = self._state
            if self._state != self.SHADOW:
                self._state = self.SHADOW
                self._record_transition(prev, self.SHADOW, metrics, "rollback_conditions_triggered")
            return {"state": self._state, "transition": prev != self._state, "direction": "ROLLBACK"}

        if self._state == self.SHADOW and len(self._stability_window) >= self._min_cycles_before_advance:
            stable_ratio = sum(self._stability_window) / len(self._stability_window)
            if stable_ratio >= 0.70:
                prev = self._state
                self._state = self.MICRO_CAPITAL
                self._record_transition(prev, self.MICRO_CAPITAL, metrics, f"stable_ratio={stable_ratio:.2f}")
                return {"state": self._state, "transition": True, "direction": "ADVANCE"}

        if self._state == self.MICRO_CAPITAL and len(self._stability_window) >= self._min_cycles_before_advance:
            stable_ratio = sum(self._stability_window) / len(self._stability_window)
            if stable_ratio >= 0.85:
                prev = self._state
                self._state = self.FULL_LIVE
                self._record_transition(prev, self.FULL_LIVE, metrics, f"stable_ratio={stable_ratio:.2f}")
                return {"state": self._state, "transition": True, "direction": "ADVANCE"}

        return {"state": self._state, "transition": False, "direction": "STAY"}

    def force_state(self, state: str) -> None:
        prev = self._state
        self._state = state
        self._record_transition(prev, state, {}, "manual_override")

    def get_transition_log(self) -> list[dict]:
        return list(self._transition_log)
