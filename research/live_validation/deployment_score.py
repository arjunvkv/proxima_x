import numpy as np

class DeploymentScore:
    def __init__(self):
        self._history: list[float] = []

    def compute(self, signal_health: float, frequency_stability: float,
                residual_strength: float, persistence_accuracy: float,
                threshold_stability: float) -> float:
        weights = {
            "signal_health": 0.15,
            "frequency_stability": 0.20,
            "residual_strength": 0.25,
            "persistence_accuracy": 0.20,
            "threshold_stability": 0.20}
        score = (
            weights["signal_health"] * np.clip(signal_health, 0, 1) +
            weights["frequency_stability"] * np.clip(frequency_stability, 0, 1) +
            weights["residual_strength"] * np.clip(residual_strength, 0, 1) +
            weights["persistence_accuracy"] * np.clip(persistence_accuracy, 0, 1) +
            weights["threshold_stability"] * np.clip(threshold_stability, 0, 1))
        self._history.append(float(score))
        return float(score)

    @property
    def classification(self) -> str:
        if not self._history:
            return "INSUFFICIENT_DATA"
        latest = self._history[-1]
        if latest >= 0.75:
            return "LIVE_HEALTHY"
        elif latest >= 0.50:
            return "LIVE_WARNING"
        else:
            return "LIVE_CRITICAL"

    @property
    def trend(self) -> str:
        if len(self._history) < 7:
            return "STABLE"
        recent = self._history[-7:]
        slope = np.polyfit(range(len(recent)), recent, 1)[0]
        if slope > 0.01:
            return "IMPROVING"
        elif slope < -0.01:
            return "DEGRADING"
        return "STABLE"

    def summary(self) -> dict:
        return {
            "current_score": round(self._history[-1], 3) if self._history else 0.0,
            "classification": self.classification,
            "trend": self.trend,
            "mean_30d": round(float(np.mean(self._history[-30:])), 3) if len(self._history) >= 30 else 0.0,
            "min_90d": round(float(np.min(self._history[-90:])), 3) if len(self._history) >= 90 else 0.0,
            "n_updates": len(self._history)}
