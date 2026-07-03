from typing import Dict, Any


class AFLFeedbackEngine:
    """
    Alignment Feedback Loop (AFL)

    Uses delayed outcomes to adjust system sensitivity.
    """

    def __init__(self,
                 learning_rate: float = 0.05,
                 entropy_sensitivity: float = 1.0,
                 rotation_sensitivity: float = 1.0):
        self.lr = learning_rate
        self.entropy_sensitivity = entropy_sensitivity
        self.rotation_sensitivity = rotation_sensitivity

    def update(self,
               doa_results: Dict[str, float]) -> Dict[str, Any]:
        if not doa_results:
            return {
                "entropy_sensitivity": self.entropy_sensitivity,
                "rotation_sensitivity": self.rotation_sensitivity,
            }

        avg_score = sum(doa_results.values()) / len(doa_results)

        if avg_score > 0.2:
            self.entropy_sensitivity *= (1 + self.lr)
            self.rotation_sensitivity *= (1 + self.lr)
        elif avg_score < -0.2:
            self.entropy_sensitivity *= (1 - self.lr)
            self.rotation_sensitivity *= (1 - self.lr)

        self.entropy_sensitivity = max(0.5, min(2.0, self.entropy_sensitivity))
        self.rotation_sensitivity = max(0.5, min(2.0, self.rotation_sensitivity))

        return {
            "entropy_sensitivity": self.entropy_sensitivity,
            "rotation_sensitivity": self.rotation_sensitivity,
        }
