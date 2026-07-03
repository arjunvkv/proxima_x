from __future__ import annotations

from typing import Dict, Optional

from execution.trigger_provenance import TriggerProvenance


class ExecutionRouter:
    __slots__ = ("provenance", "risk_manager", "position_manager", "order_manager")

    def __init__(self, risk_manager, position_manager, order_manager) -> None:
        self.provenance = TriggerProvenance()
        self.risk_manager = risk_manager
        self.position_manager = position_manager
        self.order_manager = order_manager

    def route(self, symbol: str, direction: int, volume: float,
              entry_price: float, sl_price: float, tp_price: float,
              observer_state: str, reality_score: float,
              calibration_ok: bool, account_balance: float,
              open_positions: list) -> Dict:
        trade_id = f"EXEC_{symbol}_{id(self)}_{len(self.provenance._log)}"

        if observer_state != "EXECUTE":
            return self._reject(trade_id, symbol, "OBSERVER", observer_state,
                                calibration_ok, reality_score,
                                f"observer_state={observer_state} (requires EXECUTE)")

        if reality_score < 0.15:
            return self._reject(trade_id, symbol, "REALITY_GATE", observer_state,
                                calibration_ok, reality_score,
                                f"reality_score={reality_score:.4f} < 0.15")

        direction_str = "BUY" if direction > 0 else "SELL"
        risk_check = self.risk_manager.pre_order_check(
            symbol, volume, entry_price, account_balance,
            open_positions, direction=direction_str
        )
        if not risk_check.get("allowed", False):
            return self._reject(trade_id, symbol, "RISK", observer_state,
                                calibration_ok, reality_score,
                                f"risk:{risk_check.get('reason','unknown')}")

        if len(open_positions) >= 5:
            return self._reject(trade_id, symbol, "POSITION_LIMIT", observer_state,
                                calibration_ok, reality_score,
                                f"open_positions={len(open_positions)} >= 5")

        order_result = self.order_manager.place_order(
            symbol=symbol,
            order_type="buy" if direction > 0 else "sell",
            volume=volume,
            price=entry_price,
            sl=sl_price,
            tp=tp_price,
        )

        result = self.provenance.record(
            trade_id=trade_id,
            symbol=symbol,
            trigger_layer="OBSERVER",
            observer_state=observer_state,
            calibration_state="PASS" if calibration_ok else "FAIL",
            reality_score=reality_score,
            rejection_reason="",
        )

        return {
            "executed": True,
            "trade_id": trade_id,
            "order_result": order_result,
            "provenance": result,
            "latency_stages": {
                "observer_to_execution": 0,
            },
        }

    def _reject(self, trade_id: str, symbol: str, trigger_layer: str,
                observer_state: str, calibration_ok: bool,
                reality_score: float, reason: str) -> Dict:
        self.provenance.record(
            trade_id=trade_id,
            symbol=symbol,
            trigger_layer=trigger_layer,
            observer_state=observer_state,
            calibration_state="PASS" if calibration_ok else "FAIL",
            reality_score=reality_score,
            rejection_reason=reason,
        )
        return {
            "executed": False,
            "trade_id": trade_id,
            "rejection_reason": reason,
            "provenance": self.provenance.get_log()[-1],
        }
