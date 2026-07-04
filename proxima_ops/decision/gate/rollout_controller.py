from __future__ import annotations

from typing import Any


class RolloutController:
    SHADOW = "SHADOW"
    PARTIAL = "PARTIAL_ENFORCEMENT"
    FULL = "FULL_LIVE"

    def __init__(self) -> None:
        self._mode = self.SHADOW
        self._cycles_in_mode: int = 0
        self._activation_cycle: int = 0
        self._total_cycles: int = 0

    def set_mode(self, mode: str) -> None:
        if mode in (self.SHADOW, self.PARTIAL, self.FULL):
            self._mode = mode
            self._cycles_in_mode = 0

    def get_mode(self) -> str:
        return self._mode

    def tick(self) -> None:
        self._total_cycles += 1
        self._cycles_in_mode += 1

    def should_enforce(self) -> bool:
        if self._mode == self.SHADOW:
            return False
        if self._mode == self.PARTIAL:
            return self._total_cycles % 2 == 0
        return True

    def get_partial_multiplier(self) -> float:
        if self._mode == self.PARTIAL:
            return 0.5
        return 1.0

    def can_transition_to(self, target: str) -> dict[str, Any]:
        if target == self.PARTIAL and self._mode == self.SHADOW:
            return {"allowed": True, "reason": "shadow -> partial"}
        if target == self.FULL and self._mode == self.PARTIAL:
            return {"allowed": True, "reason": "partial -> full"}
        if target == self.SHADOW:
            return {"allowed": True, "reason": "rollback to shadow"}
        return {"allowed": False, "reason": "invalid transition"}

    def get_state(self) -> dict[str, Any]:
        return {
            "mode": self._mode,
            "cycles_in_mode": self._cycles_in_mode,
            "total_cycles": self._total_cycles,
            "enforcing": self.should_enforce(),
        }
