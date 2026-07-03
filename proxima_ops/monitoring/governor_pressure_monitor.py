"""
GovernorPressureMonitor
=======================
Detect when the governor is about to "release" trades. Track rule fatigue,
risk buffer expansion, and frequency budget relaxation windows.

No state changes, no execution interference. All calculations are wrapped in
try/except so a failure never crashes the calling cycle.
"""

import logging

logger = logging.getLogger("proxima_ops.monitoring.governor_pressure_monitor")

# ---------------------------------------------------------------------------
# Constants for pressure computation
# ---------------------------------------------------------------------------

CB_COOLING_THRESHOLD = 50       # cycles of uninterrupted cooling → full relief
CONFIRM_RELEASE_THRESHOLD = 2   # active confirm slots at or above this → high pressure
CONFIRM_HIGH_PRESSURE = 0.8     # confirm_pressure when slots >= threshold
CONFIRM_BASE_PRESSURE = 0.2     # confirm_pressure when slots == 0

# SEGL state → base pressure_relief mapping
#   ARMED      = low readiness  (governor waiting)
#   OBSERVE    = high readiness (governor ready to release)
#   any other  = full readiness (fallback)
SEGL_PRESSURE_MAP: dict[str, float] = {
    "ARMED": 0.2,
    "OBSERVE": 0.8,
}
SEGL_OTHER_PRESSURE = 1.0

# Weights for composite pressure_score
WEIGHT_CB = 0.35
WEIGHT_CONFIRM = 0.35
WEIGHT_SEGL = 0.30

# Duration boost ceiling: how much a long tenure in the same SEGL state
# can add to the base pressure_relief (as fraction of window).
SEGL_DURATION_BOOST_MAX = 0.3


# ---------------------------------------------------------------------------
# Monitor class
# ---------------------------------------------------------------------------

class GovernorPressureMonitor:
    """Detect when the governor is ready to release trades.

    Tracks circuit breaker cooling, confirm-gate slot saturation, and
    SEGL state-machine position to compute a composite *pressure score*
    that reflects how close the system is to releasing a trade.

    Parameters
    ----------
    window : int
        Rolling window for state-duration normalisation (default 20).
    """

    def __init__(self, window: int = 20) -> None:
        self.window = window

        # Cycle counter
        self._cycle: int = 0

        # Circuit-breaker tracking: last cycle a CB trigger was detected
        self._last_cb_cycle: int = -1  # -1 → never triggered

        # SEGL state tracking
        self._last_segl_state: str | None = None
        self._segl_state_start_cycle: int = 0

        # Cached result (returned by get_pressure() without recomputation)
        self._pressure_score: float = 0.0
        self._next_execution_likelihood: float = 0.5
        self._dominant_releasing_rule: str | None = None
        self._rule_states: dict = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, cycle_data: dict, pipeline_trace: dict) -> dict:
        """Evaluate governor pressure from the latest cycle data.

        Parameters
        ----------
        cycle_data : dict
            Executor cycle snapshot. Expected keys: ``cycle``, ``decision``,
            ``denial_reason``, ``segl_state``, ``open_positions``, etc.
        pipeline_trace : dict
            Pipeline trace snapshot. Expected keys: ``threshold_gate``,
            ``confirm_gate``, ``governor_gate``, ``execution``.

        Returns
        -------
        dict
            Pressure report with keys ``pressure_score``,
            ``next_execution_likelihood``, ``dominant_releasing_rule``,
            and ``rule_states`` (see module docstring for full schema).
        """
        # Advance cycle from input or fallback
        try:
            self._cycle = cycle_data.get("cycle", self._cycle + 1)
        except Exception:
            self._cycle += 1

        # Run computation with full try/except around everything
        try:
            result = self._compute(cycle_data, pipeline_trace)
        except Exception as exc:
            logger.warning(
                "GovernorPressureMonitor compute failed: %s", exc
            )
            result = self._empty_result()

        # Cache for get_pressure()
        self._pressure_score = result["pressure_score"]
        self._next_execution_likelihood = result["next_execution_likelihood"]
        self._dominant_releasing_rule = result["dominant_releasing_rule"]
        self._rule_states = result["rule_states"]

        logger.debug(
            "GovernorPressureMonitor cycle=%d pressure=%.4f likelihood=%.4f "
            "dominant=%s",
            self._cycle,
            self._pressure_score,
            self._next_execution_likelihood,
            self._dominant_releasing_rule,
        )

        return result

    def get_pressure(self) -> dict:
        """Return the latest pressure state without recomputing.

        Returns
        -------
        dict
            Same schema as ``update()`` return value.
        """
        return {
            "pressure_score": self._pressure_score,
            "next_execution_likelihood": self._next_execution_likelihood,
            "dominant_releasing_rule": self._dominant_releasing_rule,
            "rule_states": dict(self._rule_states),
        }

    # ------------------------------------------------------------------
    # Core computation  (each step individually wrapped in try/except)
    # ------------------------------------------------------------------

    def _compute(self, cycle_data: dict, pipeline_trace: dict) -> dict:
        """Run all three sub-computations and assemble the result."""
        # --- 1. Circuit breaker -------------------------------------------
        try:
            cb_cooling, cb_relief = self._compute_cb_pressure(cycle_data)
        except Exception as exc:
            logger.warning(
                "GovernorPressureMonitor CB pressure failed: %s", exc
            )
            cb_cooling, cb_relief = 0, 0.0

        # --- 2. Confirm gate ----------------------------------------------
        try:
            confirm_slots, confirm_pressure = self._compute_confirm_pressure(
                pipeline_trace
            )
        except Exception as exc:
            logger.warning(
                "GovernorPressureMonitor confirm pressure failed: %s", exc
            )
            confirm_slots, confirm_pressure = 0, 0.0

        # --- 3. SEGL state ------------------------------------------------
        try:
            segl_state, state_dur, segl_relief = self._compute_segl_pressure(
                cycle_data
            )
        except Exception as exc:
            logger.warning(
                "GovernorPressureMonitor SEGL pressure failed: %s", exc
            )
            segl_state, state_dur, segl_relief = "UNKNOWN", 0, 0.5

        # --- 4. Assemble rule states --------------------------------------
        rule_states = {
            "circuit_breaker": {
                "cooling": cb_cooling,
                "pressure_relief": round(cb_relief, 4),
            },
            "confirm_gate": {
                "active_confirm_slots": confirm_slots,
                "confirm_pressure": round(confirm_pressure, 4),
            },
            "segl_state": {
                "state": segl_state,
                "state_duration": state_dur,
                "pressure_relief": round(segl_relief, 4),
            },
        }

        # --- 5. Weighted pressure score ------------------------------------
        total_weight = WEIGHT_CB + WEIGHT_CONFIRM + WEIGHT_SEGL
        pressure_score = (
            cb_relief * WEIGHT_CB
            + confirm_pressure * WEIGHT_CONFIRM
            + segl_relief * WEIGHT_SEGL
        ) / total_weight
        pressure_score = round(min(max(pressure_score, 0.0), 1.0), 4)

        # --- 6. Dominant releasing rule ------------------------------------
        reliefs: dict[str, float] = {
            "circuit_breaker": cb_relief,
            "confirm_gate": confirm_pressure,
            "segl_state": segl_relief,
        }
        dominant: str | None = max(reliefs, key=reliefs.__getitem__)  # type: ignore[arg-type]
        if reliefs.get(dominant, 0.0) <= 0.0:
            dominant = None

        # --- 7. Execution likelihood ---------------------------------------
        next_execution_likelihood = round(1.0 - pressure_score, 4)

        return {
            "pressure_score": pressure_score,
            "next_execution_likelihood": next_execution_likelihood,
            "dominant_releasing_rule": dominant,
            "rule_states": rule_states,
        }

    # ------------------------------------------------------------------
    # Sub-computations  (individual rule pressure)
    # ------------------------------------------------------------------

    def _compute_cb_pressure(self, cycle_data: dict) -> tuple[int, float]:
        """Return (cooling, pressure_relief) for the circuit breaker.

        *cooling* is the number of cycles since the last CB trigger.
        *pressure_relief* is a linear ramp ``cooling / 50`` clamped to
        ``[0.0, 1.0]`` where 0.0 means "high pressure" (just triggered)
        and 1.0 means fully relieved.
        """
        denial_reason = cycle_data.get("denial_reason", "")

        # Detect CB trigger from denial_reason text
        if denial_reason and "circuit_breaker" in str(denial_reason).lower():
            self._last_cb_cycle = self._cycle

        # Also check the governor_gate in pipeline_trace (already extracted
        # upstream), but we re-use the same denial_reason heuristic here.
        # If no CB trigger has ever been recorded, treat as fully relieved.
        if self._last_cb_cycle < 0:
            return self._cycle, 1.0

        cooling = self._cycle - self._last_cb_cycle
        if cooling >= CB_COOLING_THRESHOLD:
            return cooling, 1.0

        return cooling, cooling / CB_COOLING_THRESHOLD

    def _compute_confirm_pressure(
        self, pipeline_trace: dict
    ) -> tuple[int, float]:
        """Return (active_confirm_slots, confirm_pressure).

        *confirm_pressure* is 0.8+ when ``active_confirm_slots >= 2``,
        with a linear ramp from 0.2 at 0 slots.
        """
        confirm_gate = pipeline_trace.get("confirm_gate", {})
        if not isinstance(confirm_gate, dict):
            return 0, 0.0

        slots = confirm_gate.get(
            "active_slots",
            confirm_gate.get(
                "active_confirm_slots",
                confirm_gate.get("slots", 0),
            ),
        )
        slots = int(max(0, slots))

        if slots >= CONFIRM_RELEASE_THRESHOLD:
            pressure = CONFIRM_HIGH_PRESSURE
        elif slots <= 0:
            pressure = CONFIRM_BASE_PRESSURE
        else:
            # Linear interpolation: 1 slot → somewhere between base and high
            pressure = CONFIRM_BASE_PRESSURE + (
                (CONFIRM_HIGH_PRESSURE - CONFIRM_BASE_PRESSURE)
                * (slots / CONFIRM_RELEASE_THRESHOLD)
            )

        return slots, min(pressure, 1.0)

    def _compute_segl_pressure(
        self, cycle_data: dict
    ) -> tuple[str, int, float]:
        """Return (state, state_duration, pressure_relief) for the SEGL FSM.

        *pressure_relief* starts from the base value of the current state
        (ARMED=0.2, OBSERVE=0.8, other=1.0) and is boosted by the portion
        of ``window`` that the state has been held (up to +0.3).
        """
        segl_state = cycle_data.get("segl_state", "UNKNOWN")
        if segl_state is None:
            segl_state = "UNKNOWN"
        segl_state = str(segl_state).upper()

        # Track state transitions and duration
        if segl_state != self._last_segl_state:
            self._last_segl_state = segl_state
            self._segl_state_start_cycle = self._cycle

        state_duration = self._cycle - self._segl_state_start_cycle

        # Base pressure from the state itself
        base = SEGL_PRESSURE_MAP.get(segl_state, SEGL_OTHER_PRESSURE)

        # Duration boost: longer in the same state increases relief
        duration_fraction = min(state_duration / max(self.window, 1), 1.0)
        boost = duration_fraction * SEGL_DURATION_BOOST_MAX

        return segl_state, state_duration, min(base + boost, 1.0)

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_result() -> dict:
        """Return a safe default result when the entire computation fails."""
        return {
            "pressure_score": 0.0,
            "next_execution_likelihood": 0.5,
            "dominant_releasing_rule": None,
            "rule_states": {
                "circuit_breaker": {
                    "cooling": 0,
                    "pressure_relief": 0.0,
                },
                "confirm_gate": {
                    "active_confirm_slots": 0,
                    "confirm_pressure": 0.0,
                },
                "segl_state": {
                    "state": "UNKNOWN",
                    "state_duration": 0,
                    "pressure_relief": 0.0,
                },
            },
        }
