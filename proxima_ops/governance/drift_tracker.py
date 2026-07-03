import time
import json
import os
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class DriftSnapshot:
    timestamp: float = field(default_factory=time.time)
    cycle: int = 0
    rf_mean_prob: float = 0.0
    rf_std_prob: float = 0.0
    rf_variance: float = 0.0
    rf_drift_from_baseline: float = 0.0
    mof_score: float = 0.0
    mof_state: str = ""
    mof_drift_from_baseline: float = 0.0
    edge_04_confidence: float = 0.0
    edge_04_direction: str = ""
    edge_04_signature_hash: str = ""
    segl_state: str = ""
    segl_elapsed: float = 0.0
    portfolio_conflict: float = 0.0
    execution_count_total: int = 0


class DriftTracker:
    MAX_HISTORY = 1000
    RF_DRIFT_WARN = 0.05
    RF_DRIFT_CRIT = 0.10
    MOF_DRIFT_WARN = 0.10
    MOF_DRIFT_CRIT = 0.20
    EDGE_CONFIDENCE_DRIFT_WARN = 0.10
    EDGE_CONFIDENCE_DRIFT_CRIT = 0.20

    def __init__(self, state_dir: str = None):
        self._history: deque[DriftSnapshot] = deque(maxlen=self.MAX_HISTORY)
        self._baseline: Optional[DriftSnapshot] = None
        self._state_dir = state_dir or os.path.join("state", "drift_audit_logs")
        os.makedirs(self._state_dir, exist_ok=True)

    def set_baseline(self, snapshot: DriftSnapshot):
        self._baseline = snapshot
        logger.info("Drift baseline set at cycle %d (RF mean=%.4f, MOF=%.4f)",
                     snapshot.cycle, snapshot.rf_mean_prob, snapshot.mof_score)

    def record(self, snapshot: DriftSnapshot):
        if self._baseline:
            snapshot.rf_drift_from_baseline = abs(snapshot.rf_mean_prob - self._baseline.rf_mean_prob)
            snapshot.mof_drift_from_baseline = abs(snapshot.mof_score - self._baseline.mof_score)
        self._history.append(snapshot)
        self._save_snapshot(snapshot)

    def _save_snapshot(self, snap: DriftSnapshot):
        path = os.path.join(self._state_dir, f"drift_snap_{snap.cycle:04d}_{int(snap.timestamp)}.json")
        with open(path, "w") as f:
            json.dump({
                "timestamp": snap.timestamp,
                "cycle": snap.cycle,
                "rf_mean_prob": snap.rf_mean_prob,
                "rf_std_prob": snap.rf_std_prob,
                "rf_variance": snap.rf_variance,
                "rf_drift_from_baseline": snap.rf_drift_from_baseline,
                "mof_score": snap.mof_score,
                "mof_state": snap.mof_state,
                "mof_drift_from_baseline": snap.mof_drift_from_baseline,
                "edge_04_confidence": snap.edge_04_confidence,
                "edge_04_direction": snap.edge_04_direction,
                "edge_04_signature_hash": snap.edge_04_signature_hash,
                "segl_state": snap.segl_state,
                "segl_elapsed": snap.segl_elapsed,
                "portfolio_conflict": snap.portfolio_conflict,
                "execution_count_total": snap.execution_count_total,
                "drift_alerts": self._check_alerts(snap),
            }, f, indent=2, default=str)

    def _check_alerts(self, snap: DriftSnapshot) -> list:
        alerts = []
        if snap.rf_drift_from_baseline > self.RF_DRIFT_CRIT:
            alerts.append(f"RF_DRIFT_CRITICAL: {snap.rf_drift_from_baseline:.4f} > {self.RF_DRIFT_CRIT}")
        elif snap.rf_drift_from_baseline > self.RF_DRIFT_WARN:
            alerts.append(f"RF_DRIFT_WARNING: {snap.rf_drift_from_baseline:.4f} > {self.RF_DRIFT_WARN}")
        if snap.mof_drift_from_baseline > self.MOF_DRIFT_CRIT:
            alerts.append(f"MOF_DRIFT_CRITICAL: {snap.mof_drift_from_baseline:.4f} > {self.MOF_DRIFT_CRIT}")
        elif snap.mof_drift_from_baseline > self.MOF_DRIFT_WARN:
            alerts.append(f"MOF_DRIFT_WARNING: {snap.mof_drift_from_baseline:.4f} > {self.MOF_DRIFT_WARN}")
        return alerts

    @property
    def history(self) -> list[DriftSnapshot]:
        return list(self._history)

    def trend_analysis(self) -> dict:
        if len(self._history) < 2:
            return {"status": "INSUFFICIENT_DATA", "samples": len(self._history)}
        snaps = list(self._history)
        rf_drift_values = [s.rf_drift_from_baseline for s in snaps if s.rf_drift_from_baseline > 0]
        mof_drift_values = [s.mof_drift_from_baseline for s in snaps if s.mof_drift_from_baseline > 0]
        rf_trend = (rf_drift_values[-1] - rf_drift_values[0]) / len(rf_drift_values) if len(rf_drift_values) >= 2 else 0
        mof_trend = (mof_drift_values[-1] - mof_drift_values[0]) / len(mof_drift_values) if len(mof_drift_values) >= 2 else 0
        latest = snaps[-1]
        latest_alerts = self._check_alerts(latest)
        status = "STABLE"
        if latest_alerts:
            critical = [a for a in latest_alerts if "CRITICAL" in a]
            status = "CRITICAL" if critical else "WARNING"
        return {
            "status": status,
            "samples": len(snaps),
            "rf_drift": {
                "current": snaps[-1].rf_drift_from_baseline,
                "mean": sum(rf_drift_values) / len(rf_drift_values) if rf_drift_values else 0,
                "max": max(rf_drift_values) if rf_drift_values else 0,
                "trend_per_cycle": rf_trend,
            },
            "mof_drift": {
                "current": snaps[-1].mof_drift_from_baseline,
                "mean": sum(mof_drift_values) / len(mof_drift_values) if mof_drift_values else 0,
                "max": max(mof_drift_values) if mof_drift_values else 0,
                "trend_per_cycle": mof_trend,
            },
            "alerts": latest_alerts,
        }

    def describe(self) -> dict:
        return {
            "baseline_set": self._baseline is not None,
            "records_captured": len(self._history),
            "storage_dir": self._state_dir,
            "rf_drift_warn": self.RF_DRIFT_WARN,
            "rf_drift_crit": self.RF_DRIFT_CRIT,
            "mof_drift_warn": self.MOF_DRIFT_WARN,
            "mof_drift_crit": self.MOF_DRIFT_CRIT,
        }
