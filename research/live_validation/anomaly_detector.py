import numpy as np
from collections import deque
from dataclasses import dataclass, field

@dataclass
class AnomalyRecord:
    timestamp: int
    anomaly_type: str
    value: float
    threshold: float
    severity: str
    details: str

class AnomalyDetector:
    def __init__(self):
        self._anomalies: list[AnomalyRecord] = []
        self._signal_gaps: deque = deque(maxlen=100)
        self._threshold_history: list[float] = []
        self._persistence_da_history: list[float] = []
        self._residual_sharpe_history: list[float] = []
        self._volatility_history: list[float] = []

    def check_signal_drought(self, signals_last_24h: int, timestamp: int,
                              threshold: int = 3):
        self._signal_gaps.append(signals_last_24h)
        if signals_last_24h < threshold and len(self._signal_gaps) > 10:
            record = AnomalyRecord(
                timestamp, "signal_drought", float(signals_last_24h),
                float(threshold), "WARNING",
                f"Only {signals_last_24h} signals in last 24h")
            self._anomalies.append(record)
            return record
        return None

    def check_threshold_explosion(self, threshold: float, timestamp: int,
                                   baseline: float = 0.80, max_dev: float = 0.25):
        self._threshold_history.append(threshold)
        if len(self._threshold_history) >= 10:
            dev = abs(threshold - baseline) / max(baseline, 1e-10)
            if dev > max_dev:
                record = AnomalyRecord(
                    timestamp, "threshold_explosion", threshold, baseline * (1 + max_dev),
                    "CRITICAL" if dev > 0.50 else "WARNING",
                    f"Threshold {threshold:.2f} deviates {dev:.1%} from baseline {baseline}")
                self._anomalies.append(record)
                return record
        return None

    def check_persistence_collapse(self, da: float, timestamp: int,
                                    threshold: float = 0.55):
        self._persistence_da_history.append(da)
        if da < threshold and len(self._persistence_da_history) >= 10:
            record = AnomalyRecord(
                timestamp, "persistence_collapse", da, threshold,
                "CRITICAL" if da < 0.40 else "WARNING",
                f"Persistence DA {da:.3f} below {threshold}")
            self._anomalies.append(record)
            return record
        return None

    def check_residual_collapse(self, residual_sharpe: float, timestamp: int,
                                 es_sharpe: float, threshold_ratio: float = 0.50):
        self._residual_sharpe_history.append(residual_sharpe)
        if abs(es_sharpe) > 1e-10 and len(self._residual_sharpe_history) >= 5:
            ratio = residual_sharpe / es_sharpe
            if ratio < threshold_ratio:
                record = AnomalyRecord(
                    timestamp, "residual_collapse", ratio, threshold_ratio,
                    "CRITICAL" if ratio < 0.25 else "WARNING",
                    f"Residual/ES sharpe ratio {ratio:.3f} below {threshold_ratio}")
                self._anomalies.append(record)
                return record
        return None

    def check_regime_shock(self, regime: str, timestamp: int):
        if regime == "SHOCK":
            record = AnomalyRecord(
                timestamp, "regime_shock", 1.0, 0.0,
                "WARNING", "Volatility regime shock detected")
            self._anomalies.append(record)
            return record
        return None

    def check_frequency(self, cv: float, actual_freq: float, timestamp: int,
                         target: float = 30.0, max_cv: float = 0.50):
        if cv > max_cv and len(self._anomalies) < 100:
            record = AnomalyRecord(
                timestamp, "frequency_instability", cv, max_cv,
                "CRITICAL" if cv > 0.75 else "WARNING",
                f"Frequency CV {cv:.3f} exceeds {max_cv}, actual={actual_freq:.0f}/mo")
            self._anomalies.append(record)
            return record
        return None

    def get_anomalies(self, severity: str | None = None) -> list[AnomalyRecord]:
        if severity is None:
            return self._anomalies
        return [a for a in self._anomalies if a.severity == severity]

    def summary(self) -> dict:
        types = {}
        severities = {}
        for a in self._anomalies:
            types[a.anomaly_type] = types.get(a.anomaly_type, 0) + 1
            severities[a.severity] = severities.get(a.severity, 0) + 1
        return {
            "total_anomalies": len(self._anomalies),
            "by_type": types,
            "by_severity": severities,
            "recent": [{"type": a.anomaly_type, "severity": a.severity, "timestamp": a.timestamp}
                       for a in self._anomalies[-5:]]}
