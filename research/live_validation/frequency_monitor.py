import numpy as np
from collections import deque

class FrequencyMonitor:
    def __init__(self, target: int = 30, alert_cv: float = 0.50):
        self._target = target
        self._alert_cv = alert_cv
        self._monthly_counts: list[int] = []
        self._current_month = 0
        self._current_count = 0
        self._alerts: list[dict] = []
        self._daily_signals: deque = deque(maxlen=720)

    def record_signal(self, timestamp: int):
        month = timestamp // 504
        if month != self._current_month:
            if self._current_count > 0:
                self._monthly_counts.append(self._current_count)
            self._current_month = month
            self._current_count = 0
        self._current_count += 1
        self._daily_signals.append(timestamp)

    def finalize(self):
        if self._current_count > 0:
            self._monthly_counts.append(self._current_count)

    @property
    def actual_frequency(self) -> float:
        if not self._monthly_counts:
            return 0.0
        return float(np.mean(self._monthly_counts))

    @property
    def frequency_cv(self) -> float:
        if len(self._monthly_counts) < 2:
            return 0.0
        arr = np.array(self._monthly_counts, dtype=float)
        return float(np.std(arr) / max(np.mean(arr), 1))

    @property
    def on_target(self) -> bool:
        return abs(self.actual_frequency - self._target) / max(self._target, 1) < 0.25

    def check_alert(self) -> dict | None:
        cv = self.frequency_cv
        if cv > self._alert_cv and len(self._monthly_counts) >= 2:
            alert = {
                "type": "frequency_instability",
                "cv": cv,
                "actual": self.actual_frequency,
                "target": self._target,
                "on_target": self.on_target,
                "severity": "WARNING" if cv < 0.75 else "CRITICAL"}
            self._alerts.append(alert)
            return alert
        return None

    def get_alerts(self) -> list[dict]:
        return self._alerts

    def summary(self) -> dict:
        return {
            "target": self._target,
            "actual_frequency": round(self.actual_frequency, 1),
            "frequency_cv": round(self.frequency_cv, 3),
            "on_target": self.on_target,
            "n_months": len(self._monthly_counts),
            "n_alerts": len(self._alerts)}
