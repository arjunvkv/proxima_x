import numpy as np
from collections import deque

class ThresholdMonitor:
    def __init__(self, window_long: int = 504, alert_pct: float = 0.25):
        self._history: list[float] = []
        self._window = window_long
        self._alert_pct = alert_pct
        self._alerts: list[dict] = []

    def record(self, threshold: float, timestamp: int):
        self._history.append(threshold)

    @property
    def current(self) -> float:
        return self._history[-1] if self._history else 0.80

    @property
    def rolling(self) -> float:
        if len(self._history) < 2:
            return self.current
        return float(np.mean(self._history[-min(len(self._history), self._window):]))

    @property
    def baseline(self) -> float:
        if len(self._history) < self._window:
            return self.rolling
        return float(np.mean(self._history[-self._window:]))

    @property
    def velocity(self) -> float:
        if len(self._history) < 2:
            return 0.0
        recent = self._history[-min(20, len(self._history)):]
        return float(np.polyfit(range(len(recent)), recent, 1)[0])

    @property
    def deviation(self) -> float:
        bl = self.baseline
        if bl == 0:
            return 0.0
        return abs(self.current - bl) / bl

    def check_alert(self) -> dict | None:
        dev = self.deviation
        if dev > self._alert_pct:
            alert = {
                "type": "threshold_drift",
                "current": self.current,
                "baseline": self.baseline,
                "deviation": dev,
                "velocity": self.velocity,
                "severity": "WARNING" if dev < 0.50 else "CRITICAL"}
            self._alerts.append(alert)
            return alert
        return None

    def get_alerts(self) -> list[dict]:
        return self._alerts

    def summary(self) -> dict:
        return {
            "current": self.current,
            "rolling": self.rolling,
            "baseline": self.baseline,
            "velocity": self.velocity,
            "deviation": self.deviation,
            "n_alerts": len(self._alerts)}
