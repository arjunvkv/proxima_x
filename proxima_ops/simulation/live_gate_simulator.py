"""LiveGateSimulator: Simulate decision flow without MT5 execution.

Pipeline: SIL → CORE → Confirm → Governor → VEL → EXECUTE?
Pure simulation — never calls MT5 or modifies real execution state.
"""

import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("proxima_ops.simulation.live_gate_simulator")


class LiveGateSimulator:
    """Simulate the full trade decision pipeline and report which gates pass or block.

    Each call to ``simulate()`` returns a snapshot of the decision flow,
    indicating whether execution *would* proceed and which stage (if any)
    blocks the pipeline.
    """

    def __init__(self) -> None:
        logger.debug("LiveGateSimulator initialised")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def simulate(
        self,
        cycle: int,
        best_signal: dict | None,
        confirm_counts: dict,
        governor_state: str,
        vel_allowed: bool,
        sil_scores: dict,
    ) -> dict[str, Any]:
        """Run all five gates and return the simulated decision outcome.

        Parameters
        ----------
        cycle : int
            Current cycle number (used for logging / traceability).
        best_signal : dict or None
            The best signal from CORE (expected to contain at least
            ``"confidence"``).
        confirm_counts : dict
            ``{symbol_direction: count}`` from the Confirm gate.
        governor_state : str
            Current Governor (SEGL) state (e.g. ``"ARMED"``, ``"OBSERVE"``).
        vel_allowed : bool
            Whether the Volume Expansion Layer allows execution.
        sil_scores : dict
            ``{symbol: score}`` from the SIL universe.

        Returns
        -------
        dict
            ``would_execute`` (bool), ``blocking_stage`` (str or None),
            ``confidence_path`` (list[dict]), ``simulated_at`` (str).
        """
        try:
            confidence_path: list[dict[str, Any]] = []

            # --- 1. SIL gate ---
            sil_passed, sil_detail = self._check_sil(sil_scores)
            confidence_path.append({
                "stage": "sil",
                "passed": sil_passed,
                "detail": sil_detail,
            })

            # --- 2. CORE gate ---
            core_passed, core_detail = self._check_core(best_signal)
            confidence_path.append({
                "stage": "core",
                "passed": core_passed,
                "detail": core_detail,
            })

            # --- 3. Confirm gate ---
            confirm_passed, confirm_detail = self._check_confirm(confirm_counts)
            confidence_path.append({
                "stage": "confirm",
                "passed": confirm_passed,
                "detail": confirm_detail,
            })

            # --- 4. Governor gate ---
            governor_passed, governor_detail = self._check_governor(governor_state)
            confidence_path.append({
                "stage": "governor",
                "passed": governor_passed,
                "detail": governor_detail,
            })

            # --- 5. VEL gate ---
            vel_passed, vel_detail = self._check_vel(vel_allowed)
            confidence_path.append({
                "stage": "vel",
                "passed": vel_passed,
                "detail": vel_detail,
            })

            # --- Determine blocking stage ---
            blocking_stage: str | None = None
            for entry in confidence_path:
                if not entry["passed"]:
                    blocking_stage = entry["stage"]
                    break

            would_execute = blocking_stage is None

            result: dict[str, Any] = {
                "would_execute": would_execute,
                "blocking_stage": blocking_stage,
                "confidence_path": confidence_path,
                "simulated_at": datetime.now(timezone.utc).isoformat(),
            }

            logger.debug(
                "Cycle %d | simulate complete: would_execute=%s blocking_stage=%s",
                cycle,
                would_execute,
                blocking_stage,
            )
            return result

        except Exception:
            logger.exception("LiveGateSimulator.simulate failed for cycle %d", cycle)
            return {
                "would_execute": False,
                "blocking_stage": "exception",
                "confidence_path": [
                    {"stage": "sil", "passed": False, "detail": "Exception raised"},
                    {"stage": "core", "passed": False, "detail": "Exception raised"},
                    {"stage": "confirm", "passed": False, "detail": "Exception raised"},
                    {"stage": "governor", "passed": False, "detail": "Exception raised"},
                    {"stage": "vel", "passed": False, "detail": "Exception raised"},
                ],
                "simulated_at": datetime.now(timezone.utc).isoformat(),
            }

    # ------------------------------------------------------------------
    # Internal gate checks
    # ------------------------------------------------------------------

    @staticmethod
    def _check_sil(sil_scores: dict) -> tuple[bool, str]:
        """SIL gate: passes if *sil_scores* is non-empty and max score > 0."""
        try:
            if not isinstance(sil_scores, dict) or not sil_scores:
                return False, "SIL scores dict is empty or None"
            max_score = max(sil_scores.values()) if sil_scores else 0
            if max_score > 0:
                return True, f"SIL max score {max_score} > 0"
            return False, f"SIL max score {max_score} <= 0"
        except Exception:
            logger.warning("SIL gate check failed", exc_info=True)
            return False, "Exception in SIL check"

    @staticmethod
    def _check_core(best_signal: dict | None) -> tuple[bool, str]:
        """CORE gate: passes if *best_signal* is not None and has confidence > 0."""
        try:
            if best_signal is None:
                return False, "best_signal is None"
            confidence = best_signal.get("confidence", 0)
            if not isinstance(confidence, (int, float)):
                return False, f"confidence type is {type(confidence).__name__}, expected numeric"
            if confidence > 0:
                return True, f"CORE confidence {confidence} > 0"
            return False, f"CORE confidence {confidence} <= 0"
        except Exception:
            logger.warning("CORE gate check failed", exc_info=True)
            return False, "Exception in CORE check"

    @staticmethod
    def _check_confirm(confirm_counts: dict) -> tuple[bool, str]:
        """Confirm gate: passes if any count >= 2."""
        try:
            if not isinstance(confirm_counts, dict) or not confirm_counts:
                return False, "confirm_counts dict is empty or None"
            max_count = max(confirm_counts.values()) if confirm_counts else 0
            if max_count >= 2:
                return True, f"max confirm count {max_count} >= 2"
            return False, f"max confirm count {max_count} < 2"
        except Exception:
            logger.warning("Confirm gate check failed", exc_info=True)
            return False, "Exception in Confirm check"

    @staticmethod
    def _check_governor(governor_state: str) -> tuple[bool, str]:
        """Governor gate: passes if *governor_state* is 'ARMED' (case insensitive)."""
        try:
            if not isinstance(governor_state, str) or not governor_state:
                return False, f"governor_state is empty or not a string (got {type(governor_state).__name__})"
            if governor_state.strip().upper() == "ARMED":
                return True, f"Governor state is '{governor_state}'"
            return False, f"Governor state is '{governor_state}', expected 'ARMED'"
        except Exception:
            logger.warning("Governor gate check failed", exc_info=True)
            return False, "Exception in Governor check"

    @staticmethod
    def _check_vel(vel_allowed: bool) -> tuple[bool, str]:
        """VEL gate: passes if *vel_allowed* is True."""
        try:
            if bool(vel_allowed):
                return True, "VEL allowed is True"
            return False, "VEL allowed is False"
        except Exception:
            logger.warning("VEL gate check failed", exc_info=True)
            return False, "Exception in VEL check"
