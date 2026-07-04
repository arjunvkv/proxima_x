from __future__ import annotations

from typing import Any

from proxima_ops.decision.shadow_engine.stre.stre_engine import STREngine, STRECoordinator
from proxima_ops.decision.shadow_engine.sof.objective import evaluate_system
from proxima_ops.decision.gate.phase6_rollout_controller import Phase6RolloutController
from proxima_ops.decision.gate.phase6_kill_switch import Phase6KillSwitch
from proxima_ops.decision.gate.phase6_scaling_engine import Phase6ScalingEngine
from proxima_ops.decision.gate.phase6_recovery_protocol import Phase6RecoveryProtocol
from proxima_ops.decision.gate.phase6_audit_logger import Phase6AuditLogger
from research.ucf.integration.ucf_propagation_schema import UCFPropagationField
from proxima_ops.orchestration.runtime_state import RuntimeState


class CycleEvaluator:
    def __init__(self, runtime_state: RuntimeState) -> None:
        self._runtime = runtime_state
        self._stre_engine = STREngine.load(window=100)
        self._stre_coordinator = STRECoordinator(self._stre_engine)
        self._last_stre_result: dict[str, Any] | None = None
        self._sof_evaluate = evaluate_system
        self._phase6_rollout = Phase6RolloutController()
        self._phase6_killswitch = Phase6KillSwitch()
        self._phase6_scaling = Phase6ScalingEngine()
        self._phase6_recovery = Phase6RecoveryProtocol()
        self._phase6_audit = Phase6AuditLogger()
        self._phase6_current_mult: float = 1.0

    def run_str_e(self, gt_sim: float, sy_sim: float, pnl_proxy: float) -> dict[str, Any]:
        _stre_result = self._stre_coordinator.step(gt_sim, sy_sim, pnl_proxy)
        self._last_stre_result = _stre_result
        _gt_signal: dict[str, float] = {"expected_move": 0.0, "p_cont": gt_sim}
        _sy_signal: dict[str, float] = {"expected_move": 0.0, "p_cont": 0.5}
        _sof_result = self._sof_evaluate(_gt_signal, _sy_signal, pnl_proxy, _stre_result)
        _stre_result["SOF"] = _sof_result["SOF"]
        _stre_result["execution_efficiency"] = _sof_result["execution_efficiency"]
        _stre_result["edge_preservation"] = _sof_result["edge_preservation"]
        return _stre_result

    def compute_ucf_alignment(self, ucf_field: UCFPropagationField | None) -> float:
        _ucf_coherence = ucf_field.field_coherence if ucf_field is not None else 0.0
        if _ucf_coherence > 0.01:
            return _ucf_coherence
        _stre = self._last_stre_result
        if _stre is not None and _stre.get("samples", 0) >= 5:
            _stas = abs(float(_stre.get("gt_corr", 0.0)) - float(_stre.get("sy_corr", 0.0)))
            _sample_confidence = min(1.0, _stre.get("samples", 0) / 200.0)
            _stas_score = max(0.0, 1.0 - _stas)
            return 0.3 + 0.7 * (_stas_score * _sample_confidence)
        return 0.5

    def evaluate_phase6(
        self,
        alignment: float,
        rc_veto_rate: float,
        mra_score: float,
        emd_score: float,
        cycle_id: int,
    ) -> dict[str, Any]:
        _p6_metrics: dict[str, float] = {
            "alignment": min(1.0, max(0.0, alignment)),
            "rc_veto_rate": rc_veto_rate,
            "mra_score": mra_score,
            "emd_score": emd_score,
        }
        _ks = self._phase6_killswitch.evaluate(_p6_metrics)
        if _ks["triggered"]:
            self._phase6_rollout.force_state("SHADOW")
            self._phase6_recovery.trigger(cycle_id)
            self._phase6_current_mult = 0.0
            self._phase6_audit.log_kill_switch(_p6_metrics, "; ".join(_ks.get("failures", [])))
        _roll = self._phase6_rollout.evaluate(_p6_metrics)
        if _roll.get("transition"):
            self._phase6_audit.log_transition(
                _roll.get("from_state", "SHADOW"),
                _roll["state"],
                _p6_metrics,
                _roll.get("reason", "state_change"),
            )
        _scaling = self._phase6_scaling.evaluate(
            _p6_metrics["alignment"],
            _p6_metrics["rc_veto_rate"],
            _p6_metrics["emd_score"],
        )
        self._phase6_current_mult = _scaling["position_size_multiplier"]
        if _roll["state"] == "SHADOW":
            self._phase6_current_mult = 0.0
        _pv6 = self._phase6_recovery.evaluate(cycle_id, _p6_metrics["alignment"], _p6_metrics["rc_veto_rate"])
        if _pv6.get("active"):
            self._phase6_current_mult = min(self._phase6_current_mult, _pv6.get("max_exposure", 1.0))
        return {
            "state": _roll["state"],
            "multiplier": self._phase6_current_mult,
            "transition": _roll.get("transition", False),
            "direction": _roll.get("direction", "STAY"),
            "kill_switch_triggered": _ks["triggered"],
            "kill_switch_failures": _ks.get("failures", []),
            "rollout_state": _roll["state"],
            "rollout_transition": _roll.get("transition", False),
            "stability_score": _scaling.get("stability_score", 0.0),
            "stability_tier": _scaling.get("stability_tier", "critical"),
            "position_size_multiplier": self._phase6_current_mult,
            "recovery_active": _pv6.get("active", False),
            "recovery_phase": _pv6.get("phase", "NORMAL"),
            "recovery_max_exposure": _pv6.get("max_exposure", 1.0),
            "metrics": _p6_metrics,
        }

    def save_state(self) -> None:
        if self._stre_engine is not None:
            self._stre_engine.save()
