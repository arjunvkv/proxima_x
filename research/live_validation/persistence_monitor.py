import numpy as np
from collections import deque

class PersistenceMonitor:
    def __init__(self, alert_da: float = 0.55):
        self._predicted: list[str] = []
        self._actual: list[str] = []
        self._predicted_durations: list[int] = []
        self._actual_durations: list[int] = []
        self._alerts: list[dict] = []
        self._alert_da = alert_da

    def record(self, predicted_class: str, actual_class: str,
               predicted_bars: int, actual_bars: int, timestamp: int):
        self._predicted.append(predicted_class)
        self._actual.append(actual_class)
        self._predicted_durations.append(predicted_bars)
        self._actual_durations.append(actual_bars)

    @property
    def directional_accuracy(self) -> float:
        if len(self._predicted) == 0:
            return 0.0
        correct = sum(1 for p, a in zip(self._predicted, self._actual) if p == a)
        return correct / len(self._predicted)

    @property
    def duration_error(self) -> float:
        if len(self._predicted_durations) == 0:
            return 0.0
        arr = np.array(self._predicted_durations) - np.array(self._actual_durations)
        return float(np.mean(np.abs(arr)))

    @property
    def forecast_decay(self) -> float:
        if len(self._predicted) < 20:
            return 0.0
        half = len(self._predicted) // 2
        first_half = self._predicted[:half]
        first_actual = self._actual[:half]
        second_half = self._predicted[half:2*half]
        second_actual = self._actual[half:2*half]
        da_first = sum(1 for p, a in zip(first_half, first_actual) if p == a) / max(len(first_half), 1)
        da_second = sum(1 for p, a in zip(second_half, second_actual) if p == a) / max(len(second_half), 1)
        return da_first - da_second

    def check_alert(self) -> dict | None:
        da = self.directional_accuracy
        if da < self._alert_da and len(self._predicted) >= 20:
            alert = {
                "type": "persistence_collapse",
                "directional_accuracy": da,
                "duration_error": self.duration_error,
                "forecast_decay": self.forecast_decay,
                "severity": "WARNING" if da >= 0.40 else "CRITICAL"}
            self._alerts.append(alert)
            return alert
        return None

    def get_alerts(self) -> list[dict]:
        return self._alerts

    def summary(self) -> dict:
        return {
            "directional_accuracy": self.directional_accuracy,
            "duration_error": self.duration_error,
            "forecast_decay": self.forecast_decay,
            "n_predictions": len(self._predicted),
            "n_alerts": len(self._alerts)}
