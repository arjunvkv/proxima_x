import time
import json
import os
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ConsistencyResult:
    timestamp: float = field(default_factory=time.time)
    test_name: str = ""
    layer: str = ""
    passed: bool = False
    score: float = 0.0
    threshold: float = 0.0
    detail: str = ""


class SelfConsistencyAuditor:
    RF_IDENTITY_THRESHOLD = 0.80
    MOF_MEANING_THRESHOLD = 0.75
    EDGE_SIGNATURE_THRESHOLD = 0.85

    def __init__(self, state_dir: str = None):
        self._results: list[ConsistencyResult] = []
        self._state_dir = state_dir or os.path.join("state", "integrity_audit_logs")
        os.makedirs(self._state_dir, exist_ok=True)
        self._edge_baseline: Optional[dict] = None
        self._rf_baseline: Optional[dict] = None

    def load_edge_baseline(self, path: str):
        if os.path.exists(path):
            with open(path) as f:
                self._edge_baseline = json.load(f)
            logger.info("Edge baseline loaded from %s", path)

    def load_rf_baseline(self, path: str):
        if os.path.exists(path):
            with open(path) as f:
                self._rf_baseline = json.load(f)
            logger.info("RF baseline loaded from %s", path)

    def test_edge_identity(self, current_edge_state: dict) -> ConsistencyResult:
        result = ConsistencyResult(
            test_name="edge_04_identity_preservation",
            layer="edge_04",
            threshold=self.EDGE_SIGNATURE_THRESHOLD,
        )
        if not self._edge_baseline:
            result.passed = False
            result.score = 0.0
            result.detail = "No edge baseline loaded"
            self._results.append(result)
            return result

        baseline_conf = self._edge_baseline.get("mean_confidence", 0.0)
        current_conf = current_edge_state.get("confidence", 0.0)
        if current_conf > 0:
            score = min(current_conf / baseline_conf, baseline_conf / current_conf)
        else:
            score = 0.0

        baseline_dir = self._edge_baseline.get("direction", "")
        current_dir = current_edge_state.get("direction", "")
        dir_match = baseline_dir == current_dir

        baseline_signature = self._edge_baseline.get("signature_hash", "")
        if baseline_signature:
            current_signature = self._compute_signature(current_edge_state)
            sig_match = baseline_signature == current_signature
        else:
            sig_match = False

        composite = score * 0.7 + (0.3 if dir_match else 0.0)
        result.score = round(composite, 4)
        result.passed = composite >= self.EDGE_SIGNATURE_THRESHOLD
        result.detail = (
            f"conf_ratio={score:.4f}, dir_match={dir_match}, "
            f"sig_match={sig_match}, composite={composite:.4f}"
        )
        self._results.append(result)
        return result

    def test_mof_meaning_stability(self, current_mof_state: str, current_mof_score: float,
                                    current_portfolio_conflict: float) -> ConsistencyResult:
        result = ConsistencyResult(
            test_name="mof_meaning_stability",
            layer="mof",
            threshold=self.MOF_MEANING_THRESHOLD,
        )
        expected_scores = {"INFORMATION_RICH": (0.65, 1.0), "STRUCTURE_LIMITED": (0.35, 0.65), "INFORMATION_DEGRADED": (0.0, 0.35)}
        if current_mof_state in expected_scores:
            low, high = expected_scores[current_mof_state]
            score = 1.0 if low <= current_mof_score <= high else 0.3
        else:
            score = 0.0
        conflict_ok = current_portfolio_conflict <= 0.30 if current_mof_state == "STRUCTURE_LIMITED" else True
        composite = score * 0.7 + (0.3 if conflict_ok else 0.0)
        result.score = round(composite, 4)
        result.passed = composite >= self.MOF_MEANING_THRESHOLD
        result.detail = (
            f"score_bounds_check={'OK' if score > 0.5 else 'FAIL'}, "
            f"conflict_ok={conflict_ok}, composite={composite:.4f}"
        )
        self._results.append(result)
        return result

    def test_rf_consistency(self, current_rf_data: dict) -> ConsistencyResult:
        result = ConsistencyResult(
            test_name="rf_output_consistency",
            layer="rf",
            threshold=self.RF_IDENTITY_THRESHOLD,
        )
        if not self._rf_baseline:
            result.passed = False
            result.score = 0.0
            result.detail = "No RF baseline loaded"
            self._results.append(result)
            return result

        baseline_mean = self._rf_baseline.get("mean_probability", 0.618)
        current_mean = current_rf_data.get("mean_probability", baseline_mean)
        if current_mean > 0:
            ratio = min(current_mean / baseline_mean, baseline_mean / current_mean)
        else:
            ratio = 0.0

        baseline_ready = self._rf_baseline.get("ready_count", 28)
        current_ready = current_rf_data.get("ready_count", baseline_ready)
        ready_ratio = min(current_ready, baseline_ready) / max(current_ready, baseline_ready) if max(current_ready, baseline_ready) > 0 else 1.0

        composite = ratio * 0.6 + ready_ratio * 0.4
        result.score = round(composite, 4)
        result.passed = composite >= self.RF_IDENTITY_THRESHOLD
        result.detail = (
            f"prob_ratio={ratio:.4f}, ready_ratio={ready_ratio:.4f}, composite={composite:.4f}"
        )
        self._results.append(result)
        return result

    def _compute_signature(self, state: dict) -> str:
        parts = [
            str(state.get("direction", "")),
            str(state.get("strategy", "")),
            str(state.get("symbol", "")),
            str(round(state.get("confidence", 0), 2)),
        ]
        return hashlib.md5("|".join(parts).encode()).hexdigest()[:8]

    def summary(self) -> dict:
        if not self._results:
            return {"status": "NO_TESTS_RUN", "passed": 0, "total": 0}
        passed = sum(1 for r in self._results if r.passed)
        total = len(self._results)
        return {
            "status": "ALL_PASSED" if passed == total else f"{passed}/{total} PASSED",
            "passed": passed,
            "total": total,
            "score": round(passed / total, 4) if total > 0 else 0,
            "results": [
                {"test": r.test_name, "layer": r.layer, "passed": r.passed,
                 "score": r.score, "detail": r.detail}
                for r in self._results
            ],
        }
