from typing import Dict, Any


class SystemSelfOptimizationLoop:
    """
    SSOL — System Self-Optimization Loop

    Global meta-controller for entire trading system.
    """

    def __init__(self):
        self.exploration = 0.5
        self.stability = 0.5
        self.learning_rate = 0.05

    def update(self,
               lct_score: float,
               drift_scores: Dict[str, float],
               mso_state: Dict[str, float],
               drl_state: Dict[str, float]) -> Dict[str, float]:

        avg_drift = sum(drift_scores.values()) / len(drift_scores) if drift_scores else 0.0

        oscillation_penalty = 0.0
        if mso_state.get("regularization", 1.0) < 1.0:
            oscillation_penalty = 0.2

        if lct_score > 0.3:
            self.exploration *= 0.95
            self.stability *= 1.05
        elif lct_score < -0.3:
            self.exploration *= 1.10
            self.stability *= 0.90

        if avg_drift > 0.6:
            self.stability *= 1.1
            self.exploration *= 0.9

        if oscillation_penalty > 0:
            self.stability *= 1.1
            self.learning_rate *= 0.9

        self.learning_rate *= drl_state.get("regularization", 1.0)

        self.exploration = max(0.1, min(0.9, self.exploration))
        self.stability = max(0.1, min(0.9, self.stability))
        self.learning_rate = max(0.01, min(0.2, self.learning_rate))

        return {
            "exploration": self.exploration,
            "stability": self.stability,
            "learning_rate": self.learning_rate,
        }
