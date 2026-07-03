import logging
from typing import Optional

logger = logging.getLogger("proxima_ops.ard.advisor")


class DeploymentAdvisor:
    def __init__(self, confidence_engine):
        self._engine = confidence_engine

    def biggest_risk(self, hypotheses: dict) -> str:
        if not hypotheses:
            return "Unknown"
        sorted_h = sorted(hypotheses.items(), key=lambda x: x[1])
        return sorted_h[0][0] if sorted_h[0][1] < 0.5 else sorted_h[0][0]

    def biggest_strength(self, hypotheses: dict) -> str:
        if not hypotheses:
            return "Unknown"
        sorted_h = sorted(hypotheses.items(), key=lambda x: -x[1])
        return sorted_h[0][0]

    def recommend(self, rec: dict, risk: str, strength: str) -> dict:
        return {
            "recommendation": rec.get("recommendation", "NO_ACTION"),
            "reason": rec.get("reasons", ["No reason"])[0] if rec.get("reasons") else "No reason",
            "biggest_risk": risk,
            "biggest_strength": strength,
            "evidence_strength": 0.0,
            "research_confidence": self._engine.research_confidence(),
            "deployment_confidence": 0.0,
            "alpha_transfer": rec.get("ate", 0.0)}
