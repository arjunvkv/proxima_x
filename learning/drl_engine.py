from typing import Dict, Any


class DriftResolutionLayer:
    """
    DRL — Drift Resolution Layer

    Converts CDM instability into adaptive learning weights.
    """

    def __init__(self,
                 base_cal_weight: float = 0.5,
                 base_tca_weight: float = 0.5):

        self.base_cal = base_cal_weight
        self.base_tca = base_tca_weight

        self.cal_weight = base_cal_weight
        self.tca_weight = base_tca_weight

        self.regularization = 1.0

    def adapt(self,
              drift_scores: Dict[str, float]) -> Dict[str, float]:
        if not drift_scores:
            return self._state()

        avg_drift = sum(drift_scores.values()) / len(drift_scores)

        if avg_drift > 0.75:
            self.cal_weight *= 0.90
            self.tca_weight *= 1.10
            self.regularization = 0.70
        elif avg_drift < 0.25:
            self.cal_weight *= 1.10
            self.tca_weight *= 0.90
            self.regularization = 1.10
        else:
            self.cal_weight *= 1.02
            self.tca_weight *= 1.02
            self.regularization = 1.0

        total = self.cal_weight + self.tca_weight
        self.cal_weight /= total
        self.tca_weight /= total

        return self._state()

    def _state(self) -> Dict[str, float]:
        return {
            "cal_weight": self.cal_weight,
            "tca_weight": self.tca_weight,
            "regularization": self.regularization,
        }
