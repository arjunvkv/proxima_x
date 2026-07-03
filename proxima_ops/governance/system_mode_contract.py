"""
Proxima System Mode Contract — unified mode governance layer.

Defines:
- SystemMode dataclass (single source of truth for system state)
- ModeValidator (enforces plane/execution/MOF/UI compatibility)
- StateTransitionGraph (allowed plane transitions)
- RuntimeInvariantChecker (per-cycle validation)
- ModeSnapshotLogger (periodic state recording)
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Optional

logger = logging.getLogger("proxima.system_mode")

# ── Plane definitions ──────────────────────────────────────────────────────

class Plane(Enum):
    RESEARCH = "RESEARCH"
    SIMULATION = "SIMULATION"
    EXECUTION = "EXECUTION"

class ExecutionMode(Enum):
    REAL = "REAL"
    PAPER = "PAPER"
    NONE = "NONE"

class UIMode(Enum):
    FULL_DIAGNOSTIC = "FULL_DIAGNOSTIC"
    TRADER_VIEW = "TRADER_VIEW"
    ACCEPTANCE = "ACCEPTANCE"

class MOFPolicy(Enum):
    STRICT = "STRICT"
    RELAXED_SIM = "RELAXED_SIM"
    NONE = "NONE"

# ── System Mode Contract ───────────────────────────────────────────────────

@dataclass
class SystemMode:
    """Single source of truth for Proxima mode state."""
    plane: Plane = Plane.SIMULATION
    execution: ExecutionMode = ExecutionMode.REAL
    ui: UIMode = UIMode.FULL_DIAGNOSTIC
    mof_policy: MOFPolicy = MOFPolicy.STRICT

    def __post_init__(self):
        """Auto-correct any invalid defaults via ModeValidator."""
        corrected = ModeValidator.correct(self)
        self.plane = corrected.plane
        self.execution = corrected.execution
        self.ui = corrected.ui
        self.mof_policy = corrected.mof_policy

    def snapshot(self) -> dict:
        return {k: v.value if isinstance(v, Enum) else v for k, v in asdict(self).items()}

    def __repr__(self) -> str:
        return (f"SystemMode(plane={self.plane.value}, "
                f"execution={self.execution.value}, "
                f"ui={self.ui.value}, "
                f"mof_policy={self.mof_policy.value})")

# ── Mode Validator ─────────────────────────────────────────────────────────

class ModeValidationError(Exception):
    """Raised on invalid mode configuration."""

class ModeValidator:
    """Enforces plane ↔ execution ↔ MOF ↔ UI compatibility rules."""

    RULES = [
        # (condition_fn, error_msg)
        (lambda m: m.plane == Plane.EXECUTION and m.execution != ExecutionMode.REAL,
         "EXECUTION plane requires execution=REAL"),
        (lambda m: m.plane == Plane.SIMULATION and m.execution == ExecutionMode.REAL,
         "SIMULATION plane cannot use execution=REAL (use PAPER)"),
        (lambda m: m.mof_policy == MOFPolicy.RELAXED_SIM and m.plane != Plane.SIMULATION,
         "RELAXED_SIM MOF policy is only valid in SIMULATION plane"),
        (lambda m: m.mof_policy == MOFPolicy.STRICT and m.plane == Plane.EXECUTION and m.execution != ExecutionMode.REAL,
         "STRICT MOF policy in EXECUTION plane requires execution=REAL"),
        (lambda m: m.ui == UIMode.ACCEPTANCE and m.plane == Plane.EXECUTION,
         "ACCEPTANCE UI mode is not valid in EXECUTION plane"),
    ]

    @classmethod
    def validate(cls, mode: SystemMode, raise_on_error: bool = False) -> list[str]:
        errors = []
        for condition_fn, msg in cls.RULES:
            if condition_fn(mode):
                errors.append(msg)
        if errors:
            logger.warning(f"[MODE_VALIDATOR] {len(errors)} violation(s): {'; '.join(errors)}")
            if raise_on_error:
                raise ModeValidationError("; ".join(errors))
        return errors

    @classmethod
    def correct(cls, mode: SystemMode) -> SystemMode:
        """Return a corrected mode if invalid. Logs each correction."""
        # Use __new__ to bypass __post_init__ and avoid recursion
        corrected = object.__new__(SystemMode)
        corrected.plane = mode.plane
        corrected.execution = mode.execution
        corrected.ui = mode.ui
        corrected.mof_policy = mode.mof_policy

        if corrected.plane == Plane.EXECUTION and corrected.execution != ExecutionMode.REAL:
            logger.warning(f"[MODE_VALIDATOR] Correcting: EXECUTION plane forced execution=REAL "
                           f"(was {corrected.execution.value})")
            corrected.execution = ExecutionMode.REAL
            corrected.mof_policy = MOFPolicy.STRICT

        if corrected.plane == Plane.SIMULATION and corrected.execution == ExecutionMode.REAL:
            logger.warning(f"[MODE_VALIDATOR] Correcting: SIMULATION plane forced execution=PAPER "
                           f"(was REAL)")
            corrected.execution = ExecutionMode.PAPER

        if corrected.plane != Plane.SIMULATION and corrected.mof_policy == MOFPolicy.RELAXED_SIM:
            logger.warning(f"[MODE_VALIDATOR] Correcting: RELAXED_SIM only valid in SIMULATION, "
                           f"setting mof_policy=STRICT (plane={corrected.plane.value})")
            corrected.mof_policy = MOFPolicy.STRICT

        return corrected

# ── State Transition Graph ─────────────────────────────────────────────────

class StateTransitionGraph:
    """Defines allowed transitions between planes."""

    ALLOWED = {
        Plane.RESEARCH: {Plane.SIMULATION},
        Plane.SIMULATION: {Plane.RESEARCH, Plane.EXECUTION},
        Plane.EXECUTION: {Plane.SIMULATION},  # only via safe downgrade
    }

    @classmethod
    def can_transition(cls, current: Plane, target: Plane) -> bool:
        allowed = cls.ALLOWED.get(current, set())
        return target in allowed

    @classmethod
    def assert_transition(cls, current: Plane, target: Plane, context: str = ""):
        if not cls.can_transition(current, target):
            msg = f"Invalid transition: {current.value} → {target.value}"
            if context:
                msg += f" ({context})"
            logger.error(f"[MODE_TRANSITION] {msg}")
            raise ModeValidationError(msg)
        logger.info(f"[MODE_TRANSITION] {current.value} → {target.value} {context}")

# ── Runtime Invariant Checker ──────────────────────────────────────────────

class RuntimeInvariantChecker:
    """Per-cycle validation of SYSTEM_MODE consistency."""

    def __init__(self):
        self._last_violations: list[str] = []

    def check(self, mode: SystemMode, mof_blocked: bool = False,
              replay_mode: bool = False) -> list[str]:
        violations = []

        violations.extend(ModeValidator.validate(mode))

        if mode.mof_policy == MOFPolicy.STRICT and mof_blocked and not replay_mode:
            pass  # expected — live MT5 missing, MOF blocks correctly

        if mode.mof_policy == MOFPolicy.STRICT and not mof_blocked and not replay_mode:
            violations.append("STRICT MOF but mof_blocked=False — possible enforcement gap")

        if mode.mof_policy == MOFPolicy.RELAXED_SIM and mof_blocked:
            violations.append("RELAXED_SIM but mof_blocked=True — override not applied")

        self._last_violations = violations
        for v in violations:
            logger.warning(f"[INVARIANT] {v}")
        return violations

    @property
    def has_violations(self) -> bool:
        return len(self._last_violations) > 0

# ── Mode Snapshot Logger ───────────────────────────────────────────────────

class ModeSnapshotLogger:
    """Periodic recording of SYSTEM_MODE state for traceability."""

    def __init__(self, interval: int = 60):
        self._interval = interval
        self._cycle_count = 0

    def maybe_log(self, mode: SystemMode, mof_state: str = "",
                  mof_score: float = 0.0) -> Optional[dict]:
        self._cycle_count += 1
        if self._cycle_count % self._interval != 0:
            return None
        snap = mode.snapshot()
        snap["mof_state"] = mof_state
        snap["mof_score"] = mof_score
        logger.info(f"[MODE_SNAPSHOT] {snap}")
        return snap
