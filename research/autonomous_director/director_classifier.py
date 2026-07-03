import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger("proxima_ops.ard.classifier")

FINAL_CLASSES = [
    "RESEARCH_PENDING",
    "RESEARCH_CONFIRMED",
    "DEPLOYMENT_PENDING",
    "LIVE_READY"]


class DirectorClassifier:
    def __init__(self):
        self._history: list[dict] = []

    def classify(self, n_trades: int, evidence_strength: float,
                  research_conf: float, deployment_conf: float,
                  ate: float, health_index: float,
                  contradiction_count: int) -> dict:
        cls = "RESEARCH_PENDING"

        if n_trades < 10:
            cls = "RESEARCH_PENDING"
        elif evidence_strength < 0.3 or n_trades < 25:
            cls = "COLLECTING_EVIDENCE"
        elif ate >= 0.75 and research_conf >= 0.7 and evidence_strength >= 0.5:
            cls = "RESEARCH_CONFIRMED"
        else:
            cls = "COLLECTING_EVIDENCE"

        entry = {
            "timestamp": datetime.now().isoformat(),
            "classification": cls,
            "evidence_strength": evidence_strength,
            "research_conf": research_conf,
            "deployment_conf": deployment_conf,
            "ate": ate,
            "health_index": health_index,
            "contradictions": contradiction_count}
        self._history.append(entry)
        if len(self._history) > 500:
            self._history = self._history[-100:]
        return entry

    def current(self) -> Optional[dict]:
        return self._history[-1] if self._history else None
