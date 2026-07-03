import time
import logging
from typing import Optional

from .execution_state_machine import ExecutionStateMachine, ExecutionState
from .execution_frequency_controller import ExecutionFrequencyController
from .edge_governance_binding import EdgeGovernanceBinding, EdgeEvent

logger = logging.getLogger(__name__)


class SelectiveExecutionGovernor:
    def __init__(self):
        self.state_machine = ExecutionStateMachine()
        self.frequency_controller = ExecutionFrequencyController()
        self.edge_governance = EdgeGovernanceBinding()

    def process_signal(
        self,
        signal: dict,
        mof_state: str,
        mof_score: float,
        mof_gating_consistent: bool,
        portfolio_conflict: float,
        governance_pipeline_approved: bool,
        rf_drift: float,
        lifecycle_orphans: int,
        edge_envelope_pass: bool,
    ):
        signal_valid = signal.get("state") in ("ACTIVE", "PENDING")
        mof_baseline_ok = mof_state in ("STRUCTURE_LIMITED", "INFORMATION_RICH") and mof_gating_consistent

        logger.info("[GOVERNOR] process_signal: signal_valid=%s mof_baseline_ok=%s mof_state=%s mof_score=%.4f mof_gating_consistent=%s "
                    "portfolio_conflict=%.2f governance_pipeline_approved=%s rf_drift=%.4f lifecycle_orphans=%d "
                    "edge_envelope_pass=%s frequency_budget=%s",
                    signal_valid, mof_baseline_ok, mof_state, mof_score, mof_gating_consistent,
                    portfolio_conflict, governance_pipeline_approved, rf_drift, lifecycle_orphans,
                    edge_envelope_pass, self.frequency_controller.within_frequency_budget)

        self.state_machine.update_context({
            "signal_valid": signal_valid,
            "mof_baseline_ok": mof_baseline_ok,
            "no_portfolio_conflicts": portfolio_conflict <= 0.30,
            "rf_drift_bounded": rf_drift <= 0.05,
            "governance_pipeline_approves": governance_pipeline_approved,
            "envelope_check_passes": edge_envelope_pass,
            "within_frequency_budget": self.frequency_controller.within_frequency_budget,
            "signal_still_valid": signal_valid,
            "mof_still_baseline_ok": mof_baseline_ok,
        })

        if self.state_machine.state == ExecutionState.COOLDOWN:
            self.state_machine.update_context({
                "stabilization_cycles_elapsed": self.frequency_controller.stabilization_complete,
                "mof_recovered": self.frequency_controller.check_mof_recovery(mof_score),
                "rf_stable_post": self.frequency_controller.check_rf_recovery(rf_drift),
                "mof_degraded": mof_state in ("BLACKOUT", "DEGRADED", "NOISE"),
                "rf_drift_exceeded": rf_drift > 0.05,
                "lifecycle_incoherent": lifecycle_orphans > 0,
            })

        edge_event = self.edge_governance.evaluate_arming_eligibility(
            signal=signal,
            mof_state=mof_state,
            mof_score=mof_score,
            portfolio_conflict=portfolio_conflict,
            current_system_state=self.state_machine.state.value,
        )

        return edge_event

    def evaluate_system_state(self) -> dict:
        current = self.state_machine.state

        if current == ExecutionState.COOLDOWN:
            self.state_machine.set_context("stabilization_cycles_elapsed",
                                           self.frequency_controller.stabilization_complete)

        can_arm, denial = self.state_machine.can_transition(ExecutionState.ARMED)
        can_exec, exec_denial = self.state_machine.can_transition(ExecutionState.EXECUTING)
        can_cooldown, cd_denial = self.state_machine.can_transition(ExecutionState.COOLDOWN)
        can_observe, obs_denial = self.state_machine.can_transition(ExecutionState.OBSERVE)

        if current == ExecutionState.OBSERVE and can_arm:
            self.state_machine.transition(ExecutionState.ARMED, "Signal conditions met", "system_evaluation")

        if current == ExecutionState.ARMED:
            logger.info("GOV_FIX_ACTIVE=True evaluate_system_state: ARMED via direct context check")
            ctx = self.state_machine.context
            signal_still_valid = ctx.get("signal_valid", False)
            mof_still_ok = ctx.get("mof_baseline_ok", False)
            no_conflicts = ctx.get("no_portfolio_conflicts", True)
            rf_still_bounded = ctx.get("rf_drift_bounded", True)
            arm_conditions_hold = signal_still_valid and mof_still_ok and no_conflicts and rf_still_bounded
            if not arm_conditions_hold:
                self.state_machine.transition(ExecutionState.OBSERVE, "Signal conditions no longer met", "system_evaluation")

        if current == ExecutionState.COOLDOWN:
            if self.state_machine.can_transition(ExecutionState.LOCKED)[0]:
                self.state_machine.transition(ExecutionState.LOCKED,
                                              "Cooldown failure detected", "system_evaluation")
            elif self.state_machine.can_transition(ExecutionState.POST_TRADE_LOCK)[0]:
                self.state_machine.transition(ExecutionState.POST_TRADE_LOCK,
                                              "Cooldown complete, entering post-trade lock", "system_evaluation")
            elif self.state_machine.can_transition(ExecutionState.OBSERVE)[0]:
                self.state_machine.transition(ExecutionState.OBSERVE,
                                              "Cooldown complete", "system_evaluation")
                self.state_machine.set_context("position_closed_after_trade", False)

        if current == ExecutionState.LOCKED:
            if self.state_machine.can_transition(ExecutionState.OBSERVE)[0]:
                self.state_machine.transition(ExecutionState.OBSERVE,
                                              "Manual clearance received", "system_evaluation")

        if current == ExecutionState.POST_TRADE_LOCK:
            self.state_machine.increment_cycle()
            cycles = self.state_machine.cycles_in_state
            if cycles >= self.state_machine.lock_duration_cycles:
                self.state_machine.set_context("lock_expired", True)
                self.state_machine.set_context("no_trade_executed_during_lock", True)
            if self.state_machine.can_transition(ExecutionState.OBSERVE)[0]:
                self.state_machine.transition(ExecutionState.OBSERVE,
                                              "Post-trade lock expired", "system_evaluation")
                self.state_machine.set_context("trade_was_executed", False)
                self.state_machine.set_context("position_closed_after_trade", False)

        return {
            "state": self.state_machine.state.value,
            "transitions": {
                "to_armed": {"possible": can_arm, "denial": denial},
                "to_executing": {"possible": can_exec, "denial": exec_denial},
                "to_cooldown": {"possible": can_cooldown, "denial": cd_denial},
                "to_observe": {"possible": can_observe, "denial": obs_denial},
            },
        }

    def authorize_execution(self, signal: dict) -> tuple[bool, str]:
        ctx = self.state_machine.context
        state_str = " ".join(f"{k}={v}" for k, v in sorted(ctx.items()))

        is_armed = self.state_machine.state == ExecutionState.ARMED
        if not is_armed:
            logger.info("[GOVERNOR] rule=state_is_armed passed=False reason=System not ARMED state=%s",
                        state_str)
            return False, f"System not ARMED (state: {self.state_machine.state.value})"

        logger.info("[GOVERNOR] rule=state_is_armed passed=True state=%s", state_str)

        can_exec, denial = self.state_machine.can_transition(ExecutionState.EXECUTING)
        if not can_exec:
            logger.info("[GOVERNOR] rule=can_transition_to_executing passed=False reason=%s state=%s",
                        denial, state_str)
            return False, f"Execution denied: {denial}"

        logger.info("[GOVERNOR] rule=can_transition_to_executing passed=True state=%s", state_str)
        return True, "Execution authorized — transitioning to EXECUTING"

    def record_execution(self, signal_id: str = "", symbol: str = "", action: str = "",
                          mof_state: str = "", mof_score: float = 0.0,
                          rf_drift: float = 0.0, lifecycle_orphans: int = 0):
        if self.state_machine.state != ExecutionState.COOLDOWN:
            self.state_machine.transition(ExecutionState.COOLDOWN,
                                           f"Post-execution cooldown: {signal_id}",
                                           "execution_complete")
        self.frequency_controller.set_pre_execution_baseline({
            "mof_state": mof_state,
            "mof_score": mof_score,
        })
        from .execution_frequency_controller import ExecutionRecord
        self.frequency_controller.record_execution(
            ExecutionRecord(
                signal_id=signal_id,
                symbol=symbol,
                action=action,
                mof_state=mof_state,
                mof_score=mof_score,
                rf_drift=rf_drift,
                lifecycle_orphans=lifecycle_orphans,
            )
        )
        self.state_machine.update_context({
            "stabilization_cycles_elapsed": False,
            "mof_recovered": False,
            "rf_stable_post": False,
            "mof_degraded": mof_state in ("BLACKOUT", "DEGRADED", "NOISE"),
            "rf_drift_exceeded": rf_drift > 0.05,
            "lifecycle_incoherent": lifecycle_orphans > 0,
            "trade_was_executed": action not in ("CLOSE", "CLOSE_ALL"),
            "position_closed_after_trade": action in ("CLOSE", "CLOSE_ALL"),
        })

    def record_cycle(self):
        self.frequency_controller.record_cycle()
        self.state_machine.increment_cycle()

    def can_execute(self) -> tuple[bool, str]:
        return self.authorize_execution(None)

    def describe(self) -> dict:
        return {
            "state_machine": self.state_machine.describe(),
            "frequency_controller": self.frequency_controller.describe(),
            "edge_governance": self.edge_governance.describe(),
        }
