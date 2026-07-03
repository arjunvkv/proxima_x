import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger("proxima_ops.ard.recommendation")

RECOMMENDATIONS = [
    "NO_ACTION",
    "OBSERVE_LONGER",
    "INVESTIGATE_EXECUTION",
    "INVESTIGATE_FREQUENCY",
    "INVESTIGATE_PERSISTENCE",
    "READY_FOR_SMALL_LIVE_CAPITAL",
    "LIVE_DEPLOYABLE"]


class RecommendationEngine:
    def __init__(self):
        self._history: list[dict] = []

    def evaluate(self, n_trades: int, score: float,
                  contradiction_count: int, evidence_strength: float,
                  research_conf: float, deployment_conf: float,
                  ate: float, health_index: float,
                  freq_match: float) -> dict:
        reasons = []
        rec = "NO_ACTION"

        if n_trades < 10:
            rec = "OBSERVE_LONGER"
            reasons.append(f"Insufficient trades ({n_trades} < 10)")
        elif n_trades < 30:
            rec = "OBSERVE_LONGER"
            reasons.append(f"Limited trade sample ({n_trades} < 30)")
        elif evidence_strength < 0.5:
            rec = "OBSERVE_LONGER"
            reasons.append(f"Weak evidence ({evidence_strength} < 0.5)")

        if contradiction_count > 3 and evidence_strength > 0.3:
            if freq_match < 0.5:
                rec = "INVESTIGATE_FREQUENCY"
                reasons.append(f"High contradictions ({contradiction_count}) + low freq match ({freq_match})")
            elif deployment_conf < 0.5:
                rec = "INVESTIGATE_EXECUTION"
                reasons.append(f"High contradictions ({contradiction_count}) + low deployment conf ({deployment_conf})")
            else:
                rec = "INVESTIGATE_PERSISTENCE"
                reasons.append(f"High contradictions ({contradiction_count}) unexplained")

        if evidence_strength >= 0.6 and research_conf >= 0.7 and health_index >= 70 and ate >= 0.75:
            rec = "READY_FOR_SMALL_LIVE_CAPITAL"
            reasons.append("Strong evidence across all dimensions")

        if evidence_strength >= 0.8 and research_conf >= 0.85 and deployment_conf >= 0.8 and health_index >= 85:
            rec = "LIVE_DEPLOYABLE"
            reasons.append("Research and reality fully converge")

        entry = {
            "timestamp": datetime.now().isoformat(),
            "recommendation": rec, "reasons": reasons,
            "n_trades": n_trades, "score": score,
            "contradictions": contradiction_count,
            "evidence_strength": evidence_strength,
            "research_conf": research_conf,
            "deployment_conf": deployment_conf,
            "ate": ate, "health_index": health_index,
            "freq_match": freq_match}
        self._history.append(entry)
        if len(self._history) > 500:
            self._history = self._history[-100:]
        return entry

    def current(self) -> Optional[dict]:
        return self._history[-1] if self._history else None
