from __future__ import annotations

from typing import Any

from execution.execution_router import ExecutionRouter as _ExecutionRouter
from proxima_ops.orchestration.runtime_state import RuntimeState


PHASE6_REGIME_RULES: dict[str, dict[str, Any]] = {
    "SHADOW": {
        "multiplier_cap": 0.0,
        "allow_execution": False,
        "description": "observability only — no trades",
    },
    "MICRO_CAPITAL": {
        "multiplier_cap": 0.05,
        "allow_execution": True,
        "description": "1–5% risk scaling",
    },
    "FULL_LIVE": {
        "multiplier_cap": 1.0,
        "allow_execution": True,
        "description": "unrestricted governed execution",
    },
}


class Phase6ExecutionRouter:
    def __init__(self, inner: _ExecutionRouter) -> None:
        self._inner = inner
        self._rejection_log: list[dict[str, Any]] = []

    def route(
        self,
        symbol: str,
        direction: int,
        volume: float,
        entry_price: float,
        sl_price: float,
        tp_price: float,
        observer_state: str,
        reality_score: float,
        calibration_ok: bool,
        account_balance: float,
        open_positions: list,
        runtime_state: RuntimeState | None = None,
    ) -> dict[str, Any]:
        phase6_state = "SHADOW"
        phase6_mult = 1.0
        if runtime_state is not None:
            phase6_state = getattr(runtime_state, "_phase6_state", "SHADOW")
            phase6_mult = getattr(runtime_state, "_phase6_current_mult", 1.0)
        regime = PHASE6_REGIME_RULES.get(phase6_state, PHASE6_REGIME_RULES["SHADOW"])
        if not regime["allow_execution"]:
            _rejection = {
                "symbol": symbol,
                "direction": direction,
                "reason": f"phase6_state={phase6_state} blocks execution",
                "phase6_state": phase6_state,
                "phase6_mult": phase6_mult,
            }
            self._rejection_log.append(_rejection)
            return {
                "executed": False,
                "trade_id": "",
                "rejection_reason": _rejection["reason"],
                "phase6_state": phase6_state,
                "phase6_mult": phase6_mult,
            }
        capped_volume = volume * min(1.0, regime["multiplier_cap"] / max(0.01, phase6_mult))
        if capped_volume <= 0.0:
            _rejection = {
                "symbol": symbol,
                "direction": direction,
                "reason": f"volume capped to zero by phase6 mult={phase6_mult} cap={regime['multiplier_cap']}",
                "phase6_state": phase6_state,
                "phase6_mult": phase6_mult,
            }
            self._rejection_log.append(_rejection)
            return {
                "executed": False,
                "trade_id": "",
                "rejection_reason": _rejection["reason"],
                "phase6_state": phase6_state,
                "phase6_mult": phase6_mult,
            }
        return self._inner.route(
            symbol=symbol,
            direction=direction,
            volume=capped_volume,
            entry_price=entry_price,
            sl_price=sl_price,
            tp_price=tp_price,
            observer_state=observer_state,
            reality_score=reality_score,
            calibration_ok=calibration_ok,
            account_balance=account_balance,
            open_positions=open_positions,
        )

    @property
    def rejection_log(self) -> list[dict[str, Any]]:
        return list(self._rejection_log)
