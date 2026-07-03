from typing import Dict, List


class MetaStabilityOptimizer:
    """
    MSO — Meta-Stability Optimizer

    Detects oscillations in learning weights and suppresses instability.
    """

    def __init__(self,
                 window: int = 10,
                 oscillation_threshold: float = 0.25):

        self.window = window
        self.oscillation_threshold = oscillation_threshold

        self.history: List[Dict[str, float]] = []

    def record(self, state: Dict[str, float]):
        self.history.append(state.copy())
        if len(self.history) > self.window:
            self.history.pop(0)

    def detect_oscillation(self) -> bool:
        if len(self.history) < self.window:
            return False

        cal_series = [h.get("cal_weight", 0.5) for h in self.history]
        tca_series = [h.get("tca_weight", 0.5) for h in self.history]

        cal_var = self._variance(cal_series)
        tca_var = self._variance(tca_series)

        return (cal_var > self.oscillation_threshold and
                tca_var > self.oscillation_threshold)

    def _variance(self, series: List[float]) -> float:
        mean = sum(series) / len(series)
        return sum((x - mean) ** 2 for x in series) / len(series)

    def stabilize(self, state: Dict[str, float]) -> Dict[str, float]:
        if not self.detect_oscillation():
            return state

        damped = {
            "cal_weight": 0.5 * state.get("cal_weight", 0.5) + 0.5 * 0.5,
            "tca_weight": 0.5 * state.get("tca_weight", 0.5) + 0.5 * 0.5,
            "regularization": max(1.0, state.get("regularization", 1.0)),
        }

        return damped
