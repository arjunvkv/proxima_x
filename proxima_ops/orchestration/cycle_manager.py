from __future__ import annotations

from typing import Any, List

from proxima_ops.orchestration.runtime_state import RuntimeState
from proxima_ops.orchestration.cycle_evaluator import CycleEvaluator


class CycleManager:
    def __init__(self, runtime_state: RuntimeState, evaluator: CycleEvaluator) -> None:
        self._runtime = runtime_state
        self._evaluator = evaluator
        self._cycle_id: int = 0
        self._cycle_log: List[dict[str, Any]] = []

    def run_cycle(self) -> dict[str, Any]:
        self._cycle_id += 1
        if self._runtime._cycle_id is not None:
            self._runtime._cycle_id = self._cycle_id

        _last = self._runtime._last_stre_result or {}
        _gt_sim = float(_last.get("gt_corr", 0.5))
        _sy_sim = float(_last.get("sy_corr", 0.5))
        _pnl_proxy = 0.0
        if self._runtime.positions is not None:
            _pos_list = getattr(self._runtime.positions, "positions", [])
            if _pos_list:
                _pnl_proxy = sum(float(p.get("profit", 0.0)) for p in _pos_list)

        _ucf_field = self._runtime._cycle_context.get("ucf_field") if self._runtime._cycle_context else None

        _stre_result = self._evaluator.run_str_e(_gt_sim, _sy_sim, _pnl_proxy)
        _alignment = self._evaluator.compute_ucf_alignment(_ucf_field)
        _rc_veto_rate = 0.0
        if self._runtime._gate_decisions:
            _total = len(self._runtime._gate_decisions)
            _vetoes = sum(1 for d in self._runtime._gate_decisions if d.get("veto", False))
            _rc_veto_rate = _vetoes / _total if _total > 0 else 0.0
        _mra_score = self._runtime._gate_mra.compute() if hasattr(self._runtime, "_gate_mra") and self._runtime._gate_mra is not None else 0.5
        _emd_score = self._runtime._gate_emd.compute() if hasattr(self._runtime, "_gate_emd") and self._runtime._gate_emd is not None else 0.5
        _p6_result = self._evaluator.evaluate_phase6(
            alignment=_alignment,
            rc_veto_rate=_rc_veto_rate,
            mra_score=float(_mra_score),
            emd_score=float(_emd_score),
            cycle_id=self._cycle_id,
        )

        _result: dict[str, Any] = {
            "cycle_id": self._cycle_id,
            "stre": _stre_result,
            "ucf_alignment": _alignment,
            "phase6": _p6_result,
        }
        self._cycle_log.append(_result)
        return _result

    def run_loop(self, max_cycles: int = 0) -> None:
        if max_cycles <= 0:
            while True:
                self.run_cycle()
        else:
            for _ in range(max_cycles):
                self.run_cycle()

    @property
    def cycle_log(self) -> List[dict[str, Any]]:
        return list(self._cycle_log)

    @property
    def cycle_count(self) -> int:
        return self._cycle_id
