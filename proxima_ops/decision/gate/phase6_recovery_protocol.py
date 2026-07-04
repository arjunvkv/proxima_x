from __future__ import annotations

from typing import Any


RECOVERY_PHASES = [
    {"phase": "COOLDOWN", "min_cycles": 10, "max_exposure": 0.0, "label": "no_trading"},
    {"phase": "REVALIDATION", "min_cycles": 20, "max_exposure": 0.0, "label": "shadow_only"},
    {"phase": "GRADUAL_REENTRY", "min_cycles": 30, "max_exposure": 0.01, "label": "one_percent"},
]


class Phase6RecoveryProtocol:
    def __init__(self) -> None:
        self._active: bool = False
        self._current_phase_idx: int = 0
        self._cycles_in_phase: int = 0
        self._kill_switch_triggered_at: int = 0
        self._recovery_log: list[dict] = []
        self._total_kill_switch_events: int = 0

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def current_phase(self) -> str:
        if not self._active:
            return "NORMAL"
        return RECOVERY_PHASES[self._current_phase_idx]["phase"]

    @property
    def max_exposure(self) -> float:
        if not self._active:
            return 1.0
        return RECOVERY_PHASES[self._current_phase_idx]["max_exposure"]

    def trigger(self, cycle: int) -> None:
        self._active = True
        self._current_phase_idx = 0
        self._cycles_in_phase = 0
        self._kill_switch_triggered_at = cycle
        self._total_kill_switch_events += 1
        entry = {
            "event": "RECOVERY_START",
            "phase": RECOVERY_PHASES[0]["phase"],
            "cycle": cycle,
            "total_events": self._total_kill_switch_events,
        }
        self._recovery_log.append(entry)

    def evaluate(self, cycle: int, alignment: float, rc_veto_rate: float) -> dict:
        if not self._active:
            return {"active": False, "phase": "NORMAL", "max_exposure": 1.0}

        self._cycles_in_phase += 1
        current = RECOVERY_PHASES[self._current_phase_idx]
        advanced = False

        if self._cycles_in_phase >= current["min_cycles"]:
            next_idx = self._current_phase_idx + 1
            if next_idx < len(RECOVERY_PHASES):
                if self._current_phase_idx >= 1:
                    align_ok = alignment >= 0.60
                    rc_ok = rc_veto_rate < 0.10
                    if not (align_ok and rc_ok):
                        return {
                            "active": True,
                            "phase": current["phase"],
                            "max_exposure": current["max_exposure"],
                            "advance_blocked": True,
                            "reason": f"alignment={alignment:.4f} rc={rc_veto_rate:.4f}",
                        }
                self._current_phase_idx = next_idx
                self._cycles_in_phase = 0
                advanced = True
                entry = {
                    "event": "RECOVERY_ADVANCE",
                    "phase": RECOVERY_PHASES[next_idx]["phase"],
                    "cycle": cycle,
                    "alignment": round(alignment, 4),
                    "rc_veto_rate": round(rc_veto_rate, 4),
                }
                self._recovery_log.append(entry)

        current = RECOVERY_PHASES[self._current_phase_idx]
        result = {
            "active": True,
            "phase": current["phase"],
            "max_exposure": current["max_exposure"],
            "cycles_in_phase": self._cycles_in_phase,
            "advance_blocked": not advanced and self._cycles_in_phase >= current["min_cycles"],
        }

        if self._current_phase_idx == len(RECOVERY_PHASES) - 1 and self._cycles_in_phase >= current["min_cycles"]:
            self._active = False
            entry = {"event": "RECOVERY_COMPLETE", "cycle": cycle}
            self._recovery_log.append(entry)
            result["recovered"] = True

        return result

    def get_log(self) -> list[dict]:
        return list(self._recovery_log)
