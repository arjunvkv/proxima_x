import logging
from datetime import datetime
from typing import Optional
from proxima_ops.monitoring.signal_funnel import SignalFunnel

logger = logging.getLogger("proxima_ops.audit")


class LifecycleAudit:
    def __init__(self, funnel: SignalFunnel):
        self._funnel = funnel

    def record_generated(self, symbol: str, es: float, residual: float,
                         at: float, regime: str) -> str:
        signal_id = self._funnel.make_signal_id(symbol)
        self._funnel.generate(signal_id, symbol, es, residual, at, regime)
        return signal_id

    def record_threshold_passed(self, signal_id: str):
        self._funnel.transition(signal_id, "THRESHOLD_PASSED")

    def record_triggered(self, signal_id: str):
        self._funnel.transition(signal_id, "TRIGGERED")

    def record_blocked(self, signal_id: str, reason: str):
        state_map = {
            "SPREAD": "BLOCKED_SPREAD",
            "SPREAD_NORM": "BLOCKED_SPREAD",
            "INVALID_SPREAD": "BLOCKED_SPREAD",
            "POSITION_EXISTS": "BLOCKED_POSITION_EXISTS",
            "RISK_LIMIT": "BLOCKED_RISK_LIMIT",
            "MAX_POSITIONS": "BLOCKED_MAX_POSITIONS",
            "POSITION_LOCK": "BLOCKED_POSITION_LOCK",
            "NOT_IN_TOP3": "BLOCKED_NOT_IN_TOP3",
            "THRESHOLD_NOT_MET": "BLOCKED_THRESHOLD",
            "REJECTED": "ORDER_REJECTED",
            "RHL_BLOCKED": "BLOCKED_RHL",
            "H20": "BLOCKED_H20",
            "FLIP": "BLOCKED_FLIP",
            "FLIP_COOLDOWN": "BLOCKED_FLIP",
            "FLIP_TOO_YOUNG": "BLOCKED_FLIP",
            "ORDER_REJECTED": "ORDER_REJECTED",
            "EQUITY_PROTECTION": "BLOCKED_EQUITY_PROTECTION",
            "NO_TICK": "BLOCKED_NO_TICK",
            "ECONOMICALLY_UNVIABLE": "BLOCKED_SPREAD"}
        if reason not in state_map:
            logger.warning(f"Unknown block reason '{reason}' — mapping to BLOCKED_UNKNOWN")
        state = state_map.get(reason, "BLOCKED_UNKNOWN")
        self._funnel.transition(signal_id, state, rejection_reason=reason)

    def record_submitted(self, signal_id: str):
        self._funnel.transition(signal_id, "ORDER_SUBMITTED")

    def record_accepted(self, signal_id: str, ticket: int):
        self._funnel.transition(signal_id, "ORDER_ACCEPTED", mt5_ticket=ticket)

    def record_rejected(self, signal_id: str, retcode: int, comment: str):
        self._funnel.transition(signal_id, "ORDER_REJECTED",
                                rejection_reason=f"retcode={retcode}: {comment}")

    def record_opened(self, signal_id: str, ticket: int):
        self._funnel.transition(signal_id, "POSITION_OPENED", mt5_ticket=ticket)

    def record_closed(self, signal_id: str, pnl_points: float, pnl_money: float):
        self._funnel.transition(signal_id, "POSITION_CLOSED",
                                pnl_points=pnl_points, pnl_money=pnl_money)

    def record_sync_failure(self, signal_id: str, detail: str):
        self._funnel.transition(signal_id, "POSITION_SYNC_FAILURE",
                                rejection_reason=detail)

    def get_signal(self, signal_id: str) -> Optional[dict]:
        return self._funnel.get(signal_id)

    def funnel_summary(self) -> dict:
        return self._funnel.summary()
