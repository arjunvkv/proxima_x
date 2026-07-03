import time
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


class ExecutionState(Enum):
    OBSERVE = "OBSERVE"
    ARMED = "ARMED"
    EXECUTING = "EXECUTING"
    COOLDOWN = "COOLDOWN"
    LOCKED = "LOCKED"
    POST_TRADE_LOCK = "POST_TRADE_LOCK"


@dataclass
class TransitionEvent:
    timestamp: float = field(default_factory=time.time)
    from_state: ExecutionState = None
    to_state: ExecutionState = None
    reason: str = ""
    triggered_by: str = ""


@dataclass
class TransitionRule:
    guard_fn: callable
    key: str = ""
    reason: str = ""


class StateTransition:
    def __init__(self, source: ExecutionState, target: ExecutionState):
        self.source = source
        self.target = target
        self.guards: list[TransitionRule] = []

    def guard(self, fn: callable, reason: str = "", key: str = ""):
        self.guards.append(TransitionRule(key=key, guard_fn=fn, reason=reason))
        return self

    def check(self, context: dict) -> tuple[bool, str]:
        for g in self.guards:
            passed = g.guard_fn(context)
            state_str = " ".join(f"{k}={v}" for k, v in sorted(context.items()))
            if not passed:
                logger.info("[GOVERNOR] rule=%s passed=False reason=%s state=%s",
                            g.key, g.reason, state_str)
                return False, g.reason
            logger.info("[GOVERNOR] rule=%s passed=True state=%s", g.key, state_str)
        return True, ""


class ExecutionStateMachine:
    STATE_LABELS = {
        ExecutionState.OBSERVE: "Monitoring — no execution allowed",
        ExecutionState.ARMED: "Signal valid, no conflicts — eligible for execution consideration",
        ExecutionState.EXECUTING: "Actuation permitted — time-limited",
        ExecutionState.COOLDOWN: "Post-execution stabilization — no execution",
        ExecutionState.LOCKED: "Risk or instability detected — manual intervention required",
        ExecutionState.POST_TRADE_LOCK: "Post-trade lock — preventing immediate re-entry",
    }

    def __init__(self, lock_duration_cycles: int = 3):
        self._state: ExecutionState = ExecutionState.OBSERVE
        self._history: list[TransitionEvent] = []
        self._entered_at: float = time.time()
        self._transitions: dict[tuple[ExecutionState, ExecutionState], StateTransition] = {}
        self._context: dict = {}
        self._lock_duration_cycles = lock_duration_cycles
        self._cycles_in_state = 0
        self._build_default_transitions()

    def _build_default_transitions(self):
        self.add_transition(ExecutionState.OBSERVE, ExecutionState.ARMED, [
            ("signal_valid", "No valid signal present"),
            ("mof_baseline_ok", "MOF below STRUCTURE_LIMITED or gating inconsistent"),
            ("no_portfolio_conflicts", "Portfolio conflicts detected"),
            ("rf_drift_bounded", "RF drift exceeds threshold"),
        ])
        self.add_transition(ExecutionState.ARMED, ExecutionState.EXECUTING, [
            ("governance_pipeline_approves", "Governance pipeline rejected"),
            ("envelope_check_passes", "Execution envelope check failed"),
            ("within_frequency_budget", "Execution frequency budget exceeded"),
        ])
        t_armed_observe = StateTransition(ExecutionState.ARMED, ExecutionState.OBSERVE)
        self._transitions[(ExecutionState.ARMED, ExecutionState.OBSERVE)] = t_armed_observe
        self.add_transition(ExecutionState.EXECUTING, ExecutionState.COOLDOWN)
        self.add_transition(ExecutionState.COOLDOWN, ExecutionState.OBSERVE, [
            ("stabilization_cycles_elapsed", "Minimum stabilization not complete"),
            ("mof_recovered", "MOF not recovered to pre-execution band"),
            ("rf_stable_post", "RF drift detected during cooldown"),
        ])
        t_locked = StateTransition(ExecutionState.COOLDOWN, ExecutionState.LOCKED)
        t_locked.guard(
            lambda ctx: ctx.get("mof_degraded", False) or ctx.get("rf_drift_exceeded", False) or ctx.get("lifecycle_incoherent", False),
            "No instability detected during cooldown",
            key="mof_degraded/rf_drift_exceeded/lifecycle_incoherent",
        )
        self._transitions[(ExecutionState.COOLDOWN, ExecutionState.LOCKED)] = t_locked
        t_unlock = StateTransition(ExecutionState.LOCKED, ExecutionState.OBSERVE)
        t_unlock.guard(
            lambda ctx: ctx.get("manual_clearance", False),
            "Manual clearance required — system locked",
            key="manual_clearance",
        )
        self._transitions[(ExecutionState.LOCKED, ExecutionState.OBSERVE)] = t_unlock

        t_observe_cooldown = StateTransition(ExecutionState.OBSERVE, ExecutionState.COOLDOWN)
        t_observe_cooldown.guard(
            lambda ctx: ctx.get("position_closed_after_trade", False),
            "No position closure detected",
            key="position_closed_after_trade",
        )
        self._transitions[(ExecutionState.OBSERVE, ExecutionState.COOLDOWN)] = t_observe_cooldown

        t_cooldown_ptl = StateTransition(ExecutionState.COOLDOWN, ExecutionState.POST_TRADE_LOCK)
        t_cooldown_ptl.guard(
            lambda ctx: ctx.get("stabilization_cycles_elapsed", False),
            "Minimum stabilization not complete",
            key="stabilization_cycles_elapsed",
        )
        t_cooldown_ptl.guard(
            lambda ctx: ctx.get("mof_recovered", False),
            "MOF not recovered to pre-execution band",
            key="mof_recovered",
        )
        t_cooldown_ptl.guard(
            lambda ctx: ctx.get("rf_stable_post", False),
            "RF drift detected during cooldown",
            key="rf_stable_post",
        )
        t_cooldown_ptl.guard(
            lambda ctx: ctx.get("trade_was_executed", False),
            "No trade was executed in this cooldown",
            key="trade_was_executed",
        )
        self._transitions[(ExecutionState.COOLDOWN, ExecutionState.POST_TRADE_LOCK)] = t_cooldown_ptl

        t_ptl_observe = StateTransition(ExecutionState.POST_TRADE_LOCK, ExecutionState.OBSERVE)
        t_ptl_observe.guard(
            lambda ctx: ctx.get("lock_expired", False) or ctx.get("no_trade_executed_during_lock", False),
            "Post-trade lock still active and trade was executed during lock",
            key="lock_expired/no_trade_executed_during_lock",
        )
        self._transitions[(ExecutionState.POST_TRADE_LOCK, ExecutionState.OBSERVE)] = t_ptl_observe

    def _make_guard(self, key: str, fail_reason: str) -> callable:
        def guard(ctx: dict) -> bool:
            return ctx.get(key, False)
        return guard

    def add_transition(self, source: ExecutionState, target: ExecutionState,
                       guard_checks: list[tuple[str, str]] = None):
        t = StateTransition(source, target)
        if guard_checks:
            for key, reason in guard_checks:
                t.guard(self._make_guard(key, reason), reason, key=key)
        self._transitions[(source, target)] = t

    def can_transition(self, target: ExecutionState) -> tuple[bool, str]:
        key = (self._state, target)
        if key not in self._transitions:
            return False, f"No transition defined: {self._state.value} -> {target.value}"
        return self._transitions[key].check(self._context)

    def transition(self, target: ExecutionState, reason: str = "", trigger: str = "") -> bool:
        can, denial = self.can_transition(target)
        if not can:
            logger.warning("Transition denied %s -> %s: %s", self._state.value, target.value, denial)
            return False
        event = TransitionEvent(
            from_state=self._state,
            to_state=target,
            reason=reason,
            triggered_by=trigger,
        )
        self._history.append(event)
        self._state = target
        self._entered_at = time.time()
        self._cycles_in_state = 0
        logger.info("State transition: %s -> %s (%s)", event.from_state.value, target.value, reason)
        return True

    def set_context(self, key: str, value):
        self._context[key] = value

    def update_context(self, ctx: dict):
        self._context.update(ctx)

    @property
    def state(self) -> ExecutionState:
        return self._state

    @property
    def context(self) -> dict:
        return dict(self._context)

    @property
    def entered_at(self) -> float:
        return self._entered_at

    @property
    def elapsed(self) -> float:
        return time.time() - self._entered_at

    @property
    def history(self) -> list[TransitionEvent]:
        return list(self._history)

    def increment_cycle(self):
        self._cycles_in_state += 1

    @property
    def lock_duration_cycles(self) -> int:
        return self._lock_duration_cycles

    @property
    def cycles_in_state(self) -> int:
        return self._cycles_in_state

    def state_label(self, state: ExecutionState = None) -> str:
        s = state or self._state
        return self.STATE_LABELS.get(s, "")

    def reset(self):
        self._state = ExecutionState.OBSERVE
        self._entered_at = time.time()
        self._context = {}
        self._cycles_in_state = 0
        logger.info("State machine reset to OBSERVE")

    def describe(self) -> dict:
        return {
            "state": self._state.value,
            "entered_at": self._entered_at,
            "elapsed_seconds": self.elapsed,
            "description": self.state_label(),
            "context_keys": list(self._context.keys()),
            "transition_count": len(self._history),
            "last_transition": self._history[-1].reason if self._history else None,
        }
