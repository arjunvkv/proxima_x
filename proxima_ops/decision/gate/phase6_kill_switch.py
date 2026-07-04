from __future__ import annotations

from typing import Any


KILL_SWITCH_THRESHOLDS = {
    "alignment_min": 0.40,
    "rc_veto_rate_max": 0.25,
    "mra_collapse_min": 0.20,
    "emd_spike_max": 0.50,
    "consecutive_failures_max": 3,
}


class Phase6KillSwitch:
    def __init__(self, thresholds: dict | None = None) -> None:
        self._thresholds = thresholds or KILL_SWITCH_THRESHOLDS
        self._triggered: bool = False
        self._consecutive_failures: int = 0
        self._incidents: list[dict] = []
        self._frozen: bool = False

    @property
    def is_triggered(self) -> bool:
        return self._triggered

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    def evaluate(self, metrics: dict) -> dict:
        if self._frozen:
            return {"triggered": True, "frozen": True, "reason": "system_frozen"}

        alignment = metrics.get("alignment", 1.0)
        rc_veto = metrics.get("rc_veto_rate", 0.0)
        mra = metrics.get("mra_score", 1.0)
        emd = metrics.get("emd_score", 0.0)

        failures: list[str] = []
        if alignment < self._thresholds["alignment_min"]:
            failures.append(f"alignment={alignment:.4f}<{self._thresholds['alignment_min']}")
        if rc_veto > self._thresholds["rc_veto_rate_max"]:
            failures.append(f"rc_veto={rc_veto:.4f}>{self._thresholds['rc_veto_rate_max']}")
        if mra < self._thresholds["mra_collapse_min"]:
            failures.append(f"mra={mra:.4f}<{self._thresholds['mra_collapse_min']}")
        if emd > self._thresholds["emd_spike_max"]:
            failures.append(f"emd={emd:.4f}>{self._thresholds['emd_spike_max']}")

        if failures:
            self._consecutive_failures += 1
        else:
            self._consecutive_failures = max(0, self._consecutive_failures - 1)

        if self._consecutive_failures >= self._thresholds["consecutive_failures_max"]:
            if not self._triggered:
                self._triggered = True
                incident = {
                    "event": "KILL_SWITCH",
                    "reason": "; ".join(failures),
                    "consecutive_failures": self._consecutive_failures,
                    "metrics_snapshot": {
                        "alignment": round(alignment, 4),
                        "rc_veto_rate": round(rc_veto, 4),
                        "mra_score": round(mra, 4),
                        "emd_score": round(emd, 4),
                    },
                }
                self._incidents.append(incident)
            return {"triggered": True, "failures": failures, "consecutive": self._consecutive_failures}

        self._triggered = False
        return {"triggered": False, "failures": [], "consecutive": self._consecutive_failures}

    def freeze(self) -> None:
        self._frozen = True
        self._triggered = True
        self._incidents.append({"event": "FREEZE", "reason": "manual_freeze"})

    def reset(self) -> None:
        self._triggered = False
        self._frozen = False
        self._consecutive_failures = 0

    def get_incidents(self) -> list[dict]:
        return list(self._incidents)
