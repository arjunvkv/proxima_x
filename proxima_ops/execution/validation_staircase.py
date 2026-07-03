"""Standalone volume staircase — PnL-invariant, based ONLY on completed trade count."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Optional

logger = logging.getLogger("proxima_ops.execution.validation_staircase")


class ValidationStaircase:
    """Volume staircase that progresses through phases based solely on
    completed trade count.  PnL-invariant and replay-consistent."""

    PHASE_VOLUMES: dict[int, float] = {1: 0.01, 2: 0.03, 3: 0.05, 4: 0.10}

    def __init__(self, state_path: str = "state/volume_staircase_state.json") -> None:
        self._state_path: str = state_path
        self._completed_trades: int = 0
        self._current_phase: int = 1
        self._last_transition: Optional[float] = None
        self._load_or_init()

    # ── Public properties ──────────────────────────────────────────────

    @property
    def completed_trades(self) -> int:
        return self._completed_trades

    @property
    def current_phase(self) -> int:
        return self._current_phase

    # ── Static helpers ─────────────────────────────────────────────────

    @staticmethod
    def phase_for(trades: int) -> int:
        # Relaxed thresholds for demonstration and validation
        if trades < 2:
            return 1
        if trades < 4:
            return 2
        if trades < 6:
            return 3
        return 4

    # ── Public interface ───────────────────────────────────────────────

    def get_volume(self) -> float:
        old_phase = self._current_phase
        self._current_phase = self.phase_for(self._completed_trades)
        vol = self.PHASE_VOLUMES[self._current_phase]
        if self._current_phase != old_phase:
            self._last_transition = time.time()
            self._save()
            logger.info(
                f"[STAIRCASE_PHASE_SHIFT] Phase {old_phase} -> "
                f"{self._current_phase} at completed_trades={self._completed_trades}"
            )
        return vol

    def increment_trades(self) -> None:
        old_phase = self._current_phase
        self._completed_trades += 1
        self._current_phase = self.phase_for(self._completed_trades)
        if self._current_phase != old_phase:
            self._last_transition = time.time()
        self._save()
        logger.info(
            f"[STAIRCASE] Trade closed. Total: {self._completed_trades}, "
            f"Phase: {self._current_phase}"
        )

    def set_trades(self, trades: int) -> None:
        """Set trade count directly (used by broker reconciliation).

        This is the only way to externally sync the staircase state
        (e.g. after reconciling with the broker's trade history).
        """
        old_phase = self._current_phase
        self._completed_trades = trades
        self._current_phase = self.phase_for(trades)
        if self._current_phase != old_phase:
            self._last_transition = time.time()
        self._save()
        logger.info(
            f"[STAIRCASE] set_trades({trades}) -> phase={self._current_phase}"
        )

    def describe(self) -> dict:
        return {
            "completed_trades": self._completed_trades,
            "current_phase": self._current_phase,
            "volume": self.get_volume(),
        }

    # ── State persistence ──────────────────────────────────────────────

    def _load_or_init(self) -> None:
        if os.path.exists(self._state_path):
            try:
                with open(self._state_path) as f:
                    state = json.load(f)
                self._completed_trades = state.get("completed_trades", 0)
                self._current_phase = state.get("current_phase", 1)
                self._last_transition = state.get("last_transition")
            except (json.JSONDecodeError, OSError):
                logger.warning(
                    f"[STAIRCASE] Corrupt state file {self._state_path}, "
                    "reinitialising from scratch."
                )
                self._completed_trades = 0
                self._current_phase = 1
                self._last_transition = None
                self._save()
        else:
            self._completed_trades = 0
            self._current_phase = 1
            self._last_transition = None
            self._save()
        logger.info(
            f"[STAIRCASE] Init: completed_trades={self._completed_trades} "
            f"phase={self._current_phase}"
        )

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self._state_path) or ".", exist_ok=True)
        with open(self._state_path, "w") as f:
            json.dump(
                {
                    "completed_trades": self._completed_trades,
                    "current_phase": self._current_phase,
                    "last_transition": self._last_transition,
                },
                f,
                indent=2,
            )
