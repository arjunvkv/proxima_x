"""
Reality Integrity Dashboard — Stop guessing, start measuring.

Tracks the 4 core reality metrics:
  1. SPR — Signal Preservation Rate (executed / generated)
  2. GFR — Gate False Rejection Rate (non-PnL rejections / total)
  3. ECS — Execution Coherence Score (entry/exit alignment)
  4. MAI — Microstructure Alignment Index (model vs broker reality)

Produces real-time reports and persistent history.
"""
import os
import json
import time
import logging
from collections import defaultdict, deque
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple, Callable

import numpy as np

logger = logging.getLogger("reality_dashboard")


class RealityIntegrityDashboard:
    """
    Central dashboard for reality and integrity metrics.

    Aggregates data from:
      - GateAuditLogger (SPR, GFR)
      - ExecutionLifecycleManager (ECS)
      - MicrostructureCalibrator (MAI)

    Produces:
      - Real-time metric snapshots
      - Historical metric trends
      - Overall integrity score
      - Alert conditions
    """

    def __init__(self, history_max: int = 1000,
                 alert_thresholds: Optional[Dict[str, float]] = None):
        """
        Args:
            history_max: Max metric history entries to keep
            alert_thresholds: Custom alert thresholds per metric
        """
        self._history_max = history_max

        # Alert thresholds (defaults)
        self._alert_thresholds = alert_thresholds or {
            "spr": 0.20,       # Signal Preservation Rate < 20% = alert
            "gfr": 0.50,       # Gate False Rejection Rate > 50% = alert
            "ecs": 0.70,       # Execution Coherence < 0.70 = alert
            "mai": 0.50,       # Microstructure Alignment < 0.50 = alert
            "overall": 0.40,   # Overall integrity < 0.40 = alert
        }

        # Metric history: metric_name -> [(timestamp, value)]
        self._history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=history_max)
        )

        # Current metric values
        self._current: Dict[str, float] = {
            "spr": 0.0,
            "gfr": 0.0,
            "ecs": 1.0,
            "mai": 0.0,
            "overall": 0.0,
        }

        # Alert log
        self._alerts: deque = deque(maxlen=100)

        # Component references (optional, set externally)
        self._gate_audit = None
        self._lifecycle_mgr = None
        self._micro_calibrator = None

        # Persistence
        self._persist_path = os.path.join(
            os.getcwd(), "state", "reality_integrity.json"
        )
        self._load_persisted()

        logger.info("[REALITY_DASH] Initialized")

    def _load_persisted(self):
        """Load persisted state."""
        if not os.path.exists(self._persist_path):
            return
        try:
            with open(self._persist_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._current = data.get("current", self._current)
            # Replay history
            for metric, entries in data.get("history", {}).items():
                for ts, val in entries[-self._history_max:]:
                    self._history[metric].append((ts, val))
            logger.info(f"[REALITY_DASH] Loaded persisted metrics")
        except Exception as e:
            logger.warning(f"[REALITY_DASH] Could not load: {e}")

    def _persist(self):
        """Persist current metrics."""
        try:
            os.makedirs(os.path.dirname(self._persist_path), exist_ok=True)
            data = {
                "current": self._current,
                "history": {
                    metric: list(entries)
                    for metric, entries in self._history.items()
                },
                "last_updated": time.time(),
            }
            with open(self._persist_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception:
            pass

    # --- Component registration ---

    def register_gate_audit(self, gate_audit):
        """Register GateAuditLogger instance."""
        self._gate_audit = gate_audit

    def register_lifecycle_manager(self, lifecycle_mgr):
        """Register ExecutionLifecycleManager instance."""
        self._lifecycle_mgr = lifecycle_mgr

    def register_micro_calibrator(self, micro_calibrator):
        """Register MicrostructureCalibrator instance."""
        self._micro_calibrator = micro_calibrator

    # --- Metric computation ---

    def _compute_spr(self) -> float:
        """
        Signal Preservation Rate.

        SPR = executed_signals / generated_signals
        Measures how many generated signals survive to execution.
        """
        if self._gate_audit is not None:
            try:
                return self._gate_audit.signal_preservation_rate()
            except Exception:
                pass
        return self._current.get("spr", 0.0)

    def _compute_gfr(self) -> float:
        """
        Gate False Rejection Rate.

        GFR = non-PnL rejections / total evaluated
        Non-PnL gates are structural/logistical rejections.
        """
        if self._gate_audit is not None:
            try:
                return self._gate_audit.false_rejection_rate()
            except Exception:
                pass
        return self._current.get("gfr", 0.0)

    def _compute_ecs(self) -> float:
        """
        Execution Coherence Score.

        ECS = alignment(entry_logic, exit_logic, state_tracking)
        """
        if self._lifecycle_mgr is not None:
            try:
                return self._lifecycle_mgr.execution_coherence_score()
            except Exception:
                pass
        return self._current.get("ecs", 1.0)

    def _compute_mai(self) -> float:
        """
        Microstructure Alignment Index.

        MAI = model_assumptions vs broker_reality
        """
        if self._micro_calibrator is not None:
            try:
                return self._micro_calibrator.compute_mai()
            except Exception:
                pass
        return self._current.get("mai", 0.0)

    def _compute_overall(self, spr: float, gfr: float, ecs: float, mai: float) -> float:
        """
        Overall integrity score.

        Weighted combination of all 4 metrics:
          - SPR: 25% (how much signal survives)
          - GFR: 25% (how much is falsely rejected)
          - ECS: 25% (execution quality)
          - MAI: 25% (microstructure alignment)
        """
        # GFR is inverted (lower is better)
        gfr_score = 1.0 - gfr

        overall = 0.25 * spr + 0.25 * gfr_score + 0.25 * ecs + 0.25 * mai
        return round(max(0.0, min(1.0, overall)), 4)

    def _check_alerts(self, spr: float, gfr: float, ecs: float, mai: float, overall: float):
        """Check and log alert conditions."""
        checks = [
            ("spr", spr, self._alert_thresholds["spr"], "below", lambda v, t: v < t),
            ("gfr", gfr, self._alert_thresholds["gfr"], "above", lambda v, t: v > t),
            ("ecs", ecs, self._alert_thresholds["ecs"], "below", lambda v, t: v < t),
            ("mai", mai, self._alert_thresholds["mai"], "below", lambda v, t: v < t),
            ("overall", overall, self._alert_thresholds["overall"], "below", lambda v, t: v < t),
        ]

        for name, value, threshold, direction, condition in checks:
            if condition(value, threshold):
                alert = {
                    "ts": time.time(),
                    "metric": name,
                    "value": value,
                    "threshold": threshold,
                    "direction": direction,
                    "message": (f"[ALERT] {name.upper()} = {value:.3f} "
                                f"({direction} threshold {threshold:.3f})"),
                }
                self._alerts.append(alert)
                logger.warning(alert["message"])

    # --- Main update ---

    def update(self) -> dict:
        """
        Compute all metrics and return current snapshot.

        Call this every evaluation cycle (60s).
        """
        spr = self._compute_spr()
        gfr = self._compute_gfr()
        ecs = self._compute_ecs()
        mai = self._compute_mai()
        overall = self._compute_overall(spr, gfr, ecs, mai)

        now = time.time()

        # Update current
        self._current = {
            "spr": spr,
            "gfr": gfr,
            "ecs": ecs,
            "mai": mai,
            "overall": overall,
            "last_updated": now,
        }

        # Record history
        for metric, value in [("spr", spr), ("gfr", gfr), ("ecs", ecs),
                              ("mai", mai), ("overall", overall)]:
            self._history[metric].append((now, value))

        # Check alerts
        self._check_alerts(spr, gfr, ecs, mai, overall)

        # Persist
        self._persist()

        return self.snapshot()

    # --- Reporting ---

    def snapshot(self) -> dict:
        """Return current metric snapshot."""
        overall = self._current.get("overall", 0.0)
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": {
                "spr": {
                    "value": self._current.get("spr", 0.0),
                    "label": "Signal Preservation Rate",
                    "description": "executed_signals / generated_signals",
                    "threshold": self._alert_thresholds["spr"],
                    "status": "OK" if self._current.get("spr", 0) >= self._alert_thresholds["spr"] else "ALERT",
                },
                "gfr": {
                    "value": self._current.get("gfr", 0.0),
                    "label": "Gate False Rejection Rate",
                    "description": "non-PnL rejections / total evaluated",
                    "threshold": self._alert_thresholds["gfr"],
                    "status": "OK" if self._current.get("gfr", 0) <= self._alert_thresholds["gfr"] else "ALERT",
                },
                "ecs": {
                    "value": self._current.get("ecs", 1.0),
                    "label": "Execution Coherence Score",
                    "description": "entry/exit/state alignment",
                    "threshold": self._alert_thresholds["ecs"],
                    "status": "OK" if self._current.get("ecs", 1) >= self._alert_thresholds["ecs"] else "ALERT",
                },
                "mai": {
                    "value": self._current.get("mai", 0.0),
                    "label": "Microstructure Alignment Index",
                    "description": "model assumptions vs broker reality",
                    "threshold": self._alert_thresholds["mai"],
                    "status": "OK" if self._current.get("mai", 0) >= self._alert_thresholds["mai"] else "ALERT",
                },
            },
            "overall": {
                "value": overall,
                "label": "Overall Integrity Score",
                "threshold": self._alert_thresholds["overall"],
                "classification": self._classify(overall),
                "status": "OK" if overall >= self._alert_thresholds["overall"] else "ALERT",
            },
            "alerts": [
                {
                    "ts": datetime.fromtimestamp(a["ts"]).isoformat(),
                    "metric": a["metric"],
                    "value": a["value"],
                    "threshold": a["threshold"],
                    "message": a["message"],
                }
                for a in list(self._alerts)[-20:]  # Last 20 alerts
            ],
        }

    def _classify(self, overall: float) -> str:
        """Classify the overall integrity score."""
        if overall >= 0.80:
            return "EXCELLENT"
        elif overall >= 0.60:
            return "GOOD"
        elif overall >= 0.40:
            return "FAIR"
        elif overall >= 0.20:
            return "POOR"
        else:
            return "CRITICAL"

    def history(self, metric: str = None,
                since: float = None) -> Dict[str, List[Tuple[float, float]]]:
        """Return metric history, optionally filtered."""
        if metric:
            if metric not in self._history:
                return {}
            entries = list(self._history[metric])
            if since:
                entries = [(ts, v) for ts, v in entries if ts >= since]
            return {metric: entries}

        result = {}
        for m, entries in self._history.items():
            filtered = list(entries)
            if since:
                filtered = [(ts, v) for ts, v in filtered if ts >= since]
            result[m] = filtered
        return result

    def trend(self, metric: str, window: int = 10) -> str:
        """Return trend direction for a metric."""
        entries = list(self._history.get(metric, []))
        if len(entries) < window:
            return "INSUFFICIENT_DATA"

        recent = [v for _, v in entries[-window:]]
        if len(recent) < 2:
            return "INSUFFICIENT_DATA"

        slope = np.polyfit(range(len(recent)), recent, 1)[0]
        if slope > 0.01:
            return "IMPROVING"
        elif slope < -0.01:
            return "DEGRADING"
        else:
            return "STABLE"

    def report(self) -> str:
        """Generate a human-readable report."""
        snap = self.snapshot()
        lines = [
            "=" * 60,
            "  REALITY INTEGRITY DASHBOARD",
            f"  {snap['timestamp']}",
            "=" * 60,
            "",
            f"  OVERALL: {snap['overall']['value']:.3f} "
            f"[{snap['overall']['classification']}] "
            f"{'⚠ ALERT' if snap['overall']['status'] == 'ALERT' else '✓ OK'}",
            "",
            "  --- Metrics ---",
        ]

        for name, metric in snap['metrics'].items():
            status_symbol = "⚠" if metric['status'] == 'ALERT' else "✓"
            trend_str = self.trend(name)
            trend_symbol = {"IMPROVING": "↑", "DEGRADING": "↓", "STABLE": "→",
                           "INSUFFICIENT_DATA": "?"}.get(trend_str, "?")
            lines.append(
                f"  {status_symbol} {name.upper()}: {metric['value']:.3f} "
                f"(threshold: {metric['threshold']:.2f}) "
                f"{trend_symbol} {trend_str}"
            )

        if snap['alerts']:
            lines.extend(["", "  --- Active Alerts ---"])
            for alert in snap['alerts'][-5:]:
                lines.append(f"  ⚠ {alert['message']}")

        lines.extend([
            "",
            "=" * 60,
        ])

        return "\n".join(lines)


# Singleton
_INSTANCE: Optional[RealityIntegrityDashboard] = None


def get_reality_dashboard() -> RealityIntegrityDashboard:
    """Get or create the global RealityIntegrityDashboard instance."""
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = RealityIntegrityDashboard()
    return _INSTANCE
