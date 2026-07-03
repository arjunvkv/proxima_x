import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger("proxima_ops.ard.confidence")


class ConfidenceEngine:
    def __init__(self, hypothesis_tracker):
        self._tracker = hypothesis_tracker

    def evidence_strength(self, n_trades: int, days_active: int) -> float:
        if n_trades < 5:
            return round(n_trades / 10.0, 3)
        trade_score = min(n_trades / 50.0, 1.0)
        time_score = min(days_active / 30.0, 1.0)
        return round((trade_score * 0.6 + time_score * 0.4), 3)

    def research_confidence(self) -> float:
        hyp = self._tracker.all_confidences()
        values = list(hyp.values())
        return round(sum(values) / len(values), 3) if values else 0.5

    def deployment_confidence(self, asr: float, exec_quality: str,
                               n_trades: int) -> float:
        base = 0.3
        base += asr * 0.3
        q_map = {"EXCELLENT": 0.25, "GOOD": 0.15, "DEGRADED": 0.05, "CRITICAL": 0.0, "NO_DATA": 0.0}
        base += q_map.get(exec_quality, 0.0)
        base += min(n_trades / 100.0, 1.0) * 0.15
        return round(max(0.0, min(base, 1.0)), 3)

    def update_from_evidence(self, ate: float, sharpe: float, pp: float,
                              exec_quality: str, leakage_rate: float,
                              n_trades: int):
        if ate > 0.75:
            self._tracker.record_evidence("energy_storage_alpha", True, 0.05)
        else:
            self._tracker.record_evidence("energy_storage_alpha", False, 0.03)

        if pp > 0.55:
            self._tracker.record_evidence("residual_alpha", True, 0.05)
        else:
            self._tracker.record_evidence("residual_alpha", False, 0.03)

        if leakage_rate < 0.30:
            self._tracker.record_evidence("frequency_controller", True, 0.05)
        elif leakage_rate > 0.60:
            self._tracker.record_evidence("frequency_controller", False, 0.05)

        if exec_quality in ("EXCELLENT", "GOOD"):
            self._tracker.record_evidence("at_overlay", True, 0.03)
        elif exec_quality == "DEGRADED":
            self._tracker.record_evidence("at_overlay", False, 0.03)

    def summary(self) -> dict:
        return {
            "evidence_strength": 0.0,
            "research_confidence": self.research_confidence(),
            "deployment_confidence": 0.0,
            "hypotheses": self._tracker.all_confidences()}
