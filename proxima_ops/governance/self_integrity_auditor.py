import time
import json
import os
import logging
from typing import Optional

from .drift_tracker import DriftTracker, DriftSnapshot
from .consistency_auditor import SelfConsistencyAuditor
from .execution_behavior_analyzer import ExecutionBehaviorAnalyzer, BehaviorRecord
from .regime_survival_simulator import RegimeSurvivalSimulator

logger = logging.getLogger(__name__)


class SelfIntegrityAuditor:
    def __init__(self, state_dir: str = None):
        base = state_dir or "state"
        self.drift_tracker = DriftTracker(os.path.join(base, "drift_audit_logs"))
        self.consistency_auditor = SelfConsistencyAuditor(os.path.join(base, "integrity_audit_logs"))
        self.behavior_analyzer = ExecutionBehaviorAnalyzer(os.path.join(base, "behavior_audit_logs"))
        self.regime_simulator = RegimeSurvivalSimulator(os.path.join(base, "regime_simulation_logs"))
        self._audit_history: list[dict] = []
        self._integrity_score: float = 1.0
        self._consecutive_failures: int = 0

    def record_cycle(
        self,
        cycle: int,
        rf_mean: float,
        rf_std: float,
        rf_variance: float,
        mof_score: float,
        mof_state: str,
        edge_04_confidence: float,
        edge_04_direction: str,
        segl_state: str,
        segl_elapsed: float,
        portfolio_conflict: float,
        execution_count: int,
    ):
        snap = DriftSnapshot(
            cycle=cycle,
            rf_mean_prob=rf_mean,
            rf_std_prob=rf_std,
            rf_variance=rf_variance,
            mof_score=mof_score,
            mof_state=mof_state,
            edge_04_confidence=edge_04_confidence,
            edge_04_direction=edge_04_direction,
            segl_state=segl_state,
            segl_elapsed=segl_elapsed,
            portfolio_conflict=portfolio_conflict,
            execution_count_total=execution_count,
        )
        if not self.drift_tracker._baseline and cycle == 0:
            self.drift_tracker.set_baseline(snap)
        self.drift_tracker.record(snap)

        self.behavior_analyzer.record(BehaviorRecord(
            cycle=cycle,
            state=segl_state,
            elapsed_in_state=segl_elapsed,
            execution_authorized=(segl_state == "EXECUTING"),
        ))

        drift_analysis = self.drift_tracker.trend_analysis()
        if drift_analysis["status"] == "CRITICAL":
            self._consecutive_failures += 1
        else:
            self._consecutive_failures = 0

        self._integrity_score = self._compute_integrity(drift_analysis)

        audit_entry = {
            "cycle": cycle,
            "integrity_score": self._integrity_score,
            "drift_status": drift_analysis["status"],
            "consecutive_failures": self._consecutive_failures,
        }
        self._audit_history.append(audit_entry)
        return audit_entry

    def _compute_integrity(self, drift_analysis: dict) -> float:
        base = 1.0
        rf_drift = drift_analysis.get("rf_drift", {}).get("current", 0)
        mof_drift = drift_analysis.get("mof_drift", {}).get("current", 0)
        rf_penalty = min(rf_drift * 5, 0.3)
        mof_penalty = min(mof_drift * 3, 0.3)
        consecutive_penalty = min(self._consecutive_failures * 0.05, 0.2)
        return round(max(0.0, base - rf_penalty - mof_penalty - consecutive_penalty), 4)

    def run_consistency_audit(self, edge_data: dict, rf_data: dict,
                                mof_state: str, mof_score: float,
                                portfolio_conflict: float) -> dict:
        edge_baseline_path = os.path.join("state", "edge_04_identity_lock.json")
        if os.path.exists(edge_baseline_path):
            self.consistency_auditor.load_edge_baseline(edge_baseline_path)
        rf_baseline_path = os.path.join("state", "rf_rehydration_report.json")
        if os.path.exists(rf_baseline_path):
            self.consistency_auditor.load_rf_baseline(rf_baseline_path)
        self.consistency_auditor.test_edge_identity(edge_data)
        self.consistency_auditor.test_mof_meaning_stability(mof_state, mof_score, portfolio_conflict)
        self.consistency_auditor.test_rf_consistency(rf_data)
        return self.consistency_auditor.summary()

    def run_regime_survival_test(self, regimes: list[str] = None,
                                  cycles_per_regime: int = 20) -> dict:
        if regimes is None:
            regimes = list(self.regime_simulator.REGIMES.keys())
        for regime in regimes:
            self.regime_simulator.simulate(regime, cycles_per_regime)
        return self.regime_simulator.summary()

    def integrity_verdict(self) -> dict:
        drift = self.drift_tracker.trend_analysis()
        consistency = self.consistency_auditor.summary()
        behavior = self.behavior_analyzer.summary()
        regime = self.regime_simulator.summary()

        drift_ok = drift["status"] == "STABLE"
        consistency_ok = consistency["status"] == "ALL_PASSED" if consistency.get("total", 0) > 0 else True
        behavior_ok = all(
            b.get("status") in ("NORMAL", "INSUFFICIENT_DATA")
            for b in [behavior.get("arming_bias", {}), behavior.get("locking_bias", {}),
                       behavior.get("execution_frequency", {})]
        )
        regime_ok = all(
            r.get("final_stable", False) and not r.get("collapse_detected", True)
            for r in regime.get("results", [])
        ) if regime.get("total_simulations", 0) > 0 else True

        checks_passed = sum([drift_ok, consistency_ok, behavior_ok, regime_ok])
        total_checks = 4

        if checks_passed == total_checks:
            verdict = "TRUSTED — system identity preserved across all layers"
        elif checks_passed >= total_checks - 1:
            verdict = "DEGRADED — minor drift detected, system still coherent"
        elif checks_passed >= total_checks - 2:
            verdict = "WARNING — significant drift detected, corrective action recommended"
        else:
            verdict = "UNTRUSTED — system identity compromised, intervention required"

        return {
            "verdict": verdict,
            "integrity_score": self._integrity_score,
            "checks_passed": checks_passed,
            "total_checks": total_checks,
            "details": {
                "drift": {"status": drift["status"], "alerts": drift.get("alerts", [])},
                "consistency": {"status": consistency.get("status", "NO_DATA")},
                "behavior": {
                    "arming": behavior.get("arming_bias", {}).get("status", "NO_DATA"),
                    "locking": behavior.get("locking_bias", {}).get("status", "NO_DATA"),
                    "exec_freq": behavior.get("execution_frequency", {}).get("status", "NO_DATA"),
                },
                "regime_survival": {
                    "simulations": regime.get("total_simulations", 0),
                    "all_stable": regime_ok,
                },
            },
            "audit_history": self._audit_history[-10:] if self._audit_history else [],
        }

    def describe(self) -> dict:
        return {
            "integrity_score": self._integrity_score,
            "consecutive_failures": self._consecutive_failures,
            "drift": self.drift_tracker.describe(),
            "consistency": self.consistency_auditor.summary(),
            "behavior": self.behavior_analyzer.summary(),
        }
