import json
import os
import logging
from datetime import datetime
from typing import Optional
from proxima_ops.monitoring.deployment_context import DeploymentContext

logger = logging.getLogger("proxima_ops.funnel")


class SignalFunnel:
    def __init__(self, save_path: str = None, deployment_context: DeploymentContext = None):
        self._save_path = save_path
        self._ctx = deployment_context
        self._signals: dict[str, dict] = {}
        self._counts = {
            "GENERATED": 0, "THRESHOLD_PASSED": 0, "TRIGGERED": 0,
            "ORDER_SUBMITTED": 0, "ORDER_ACCEPTED": 0,
            "POSITION_OPENED": 0, "POSITION_CLOSED": 0,
            "BLOCKED_SPREAD": 0, "BLOCKED_POSITION_EXISTS": 0,
            "BLOCKED_RISK_LIMIT": 0,
            "BLOCKED_MAX_POSITIONS": 0, "BLOCKED_POSITION_LOCK": 0,
            "BLOCKED_NOT_IN_TOP3": 0, "BLOCKED_THRESHOLD": 0,
            "BLOCKED_RHL": 0, "BLOCKED_H20": 0, "BLOCKED_FLIP": 0,
            "BLOCKED_EQUITY_PROTECTION": 0, "BLOCKED_NO_TICK": 0,
            "BLOCKED_UNKNOWN": 0,
            "ORDER_REJECTED": 0, "ORDER_TIMEOUT": 0,
            "POSITION_SYNC_FAILURE": 0}
        if self._save_path:
            self.load()

    def save(self):
        if not self._save_path:
            return
        try:
            os.makedirs(os.path.dirname(self._save_path), exist_ok=True)
            with open(self._save_path, "w", encoding="utf-8") as f:
                json.dump({"signals": self._signals, "counts": self._counts}, f, indent=4)
        except Exception:
            pass

    def load(self):
        if not self._save_path or not os.path.exists(self._save_path):
            return
        try:
            with open(self._save_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._signals = data.get("signals", {})
            self._counts = data.get("counts", self._counts)
        except Exception:
            pass

    def make_signal_id(self, symbol: str) -> str:
        now = datetime.now()
        return f"SIG_{now.strftime('%Y%m%d_%H%M%S')}_{symbol}"

    def generate(self, signal_id: str, symbol: str, es: float, residual: float,
                 at: float, regime: str, price_at_generation: float = None) -> dict:
        rec = {
            "signal_id": signal_id, "symbol": symbol,
            "timestamp_generated": datetime.now().isoformat(),
            "timestamp_triggered": None, "timestamp_submitted": None,
            "timestamp_accepted": None, "timestamp_opened": None,
            "timestamp_closed": None,
            "final_state": "GENERATED", "rejection_reason": None,
            "mt5_ticket": None, "pnl_points": 0.0, "pnl_money": 0.0,
            "es": es, "residual": residual, "at": at, "regime": regime,
            "price_at_generation": price_at_generation,
            "forward_return_H5": None,
            "forward_return_H20": None,
            "forward_return_H50": None}
        if self._ctx:
            rec["deployment_id"] = self._ctx.deployment_id
            rec["session_id"] = self._ctx.session_id
        self._signals[signal_id] = rec
        self._counts["GENERATED"] += 1
        self.save()
        return rec

    def transition(self, signal_id: str, new_state: str,
                   rejection_reason: str = None, **kwargs) -> Optional[dict]:
        rec = self._signals.get(signal_id)
        if rec is None:
            logger.warning(f"Signal {signal_id} not found for transition to {new_state}")
            return None
        ts_field = {
            "THRESHOLD_PASSED": "timestamp_triggered",
            "TRIGGERED": "timestamp_triggered",
            "ORDER_SUBMITTED": "timestamp_submitted",
            "ORDER_ACCEPTED": "timestamp_accepted",
            "POSITION_OPENED": "timestamp_opened",
            "POSITION_CLOSED": "timestamp_closed"}.get(new_state)
        if ts_field:
            rec[ts_field] = datetime.now().isoformat()
        rec["final_state"] = new_state
        if rejection_reason:
            rec["rejection_reason"] = rejection_reason
        for k, v in kwargs.items():
            rec[k] = v
        self._counts[new_state] = self._counts.get(new_state, 0) + 1
        self.save()
        return rec

    def get(self, signal_id: str) -> Optional[dict]:
        return self._signals.get(signal_id)

    def get_all(self) -> list[dict]:
        return list(self._signals.values())

    def summary(self) -> dict:
        return dict(self._counts)

    def reset(self):
        self._signals.clear()
        for k in self._counts:
            self._counts[k] = 0
        self.save()
