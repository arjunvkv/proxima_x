"""RQ10: What is the root physics model? Is compression the deepest generator?"""

from __future__ import annotations

from typing import Any

import numpy as np

from research.compression_physics.compression_validator import CompressionValidator, CPIResult


class RootPhysicsModel:
    def __init__(self, validator: CompressionValidator, asset: str = "EURJPY"):
        self.validator = validator
        self.asset = asset
        self._max_lag = 200

    def run(self) -> CPIResult:
        data = self.validator.load_asset_data(self.asset)
        signals = self.validator.compute_signals(data)

        # Assemble evidence from all prior RQs
        rq_evidence = {}
        if hasattr(self.validator, "results"):
            for rq_name, result in self.validator.results.items():
                if hasattr(result, "status") and hasattr(result, "metrics"):
                    rq_evidence[rq_name] = {
                        "status": result.status,
                        "metrics": result.metrics,
                    }

        RQ_LABELS = {
            "RQ1: Origins": "origins",
            "RQ2: Lifecycle": "lifecycle",
            "RQ3: Necessity": "necessity",
            "RQ4: Mediation": "mediation",
            "RQ5: Asset Universality": "universality",
            "RQ6: Time Stability": "stability",
            "RQ7: Generator Tournament": "tournament",
            "RQ8: Minimal Chain": "minimal_chain",
            "RQ9: Hidden Driver": "hidden_driver",
        }
        evidence = {}
        for label, key in RQ_LABELS.items():
            evidence[key] = rq_evidence.get(label, {}).get("status", "UNKNOWN")

        passed = sum(1 for v in evidence.values() if v == "PASSED")
        failed = sum(1 for v in evidence.values() if v == "FAILED")
        inconclusive = sum(1 for v in evidence.values() if v == "INCONCLUSIVE")
        total = len([v for v in evidence.values() if v != "UNKNOWN"])

        if total == 0:
            return CPIResult("root_physics_model", "INCONCLUSIVE",
                             metrics={"error": "no RQ results available"})

        # Score evidence: convert statuses to numeric
        def _score(status: str) -> float:
            if status == "PASSED":    return 1.0
            if status == "INCONCLUSIVE": return 0.5
            if status == "FAILED":    return 0.0
            return 0.0

        evidence_scores = {k: _score(v) for k, v in evidence.items() if v != "UNKNOWN"}
        avg_score = sum(evidence_scores.values()) / max(len(evidence_scores), 1)

        tournament_score = _score(evidence.get("tournament", "UNKNOWN"))
        hidden_score = _score(evidence.get("hidden_driver", "UNKNOWN"))
        necessity_score = _score(evidence.get("necessity", "UNKNOWN"))
        stability_score = _score(evidence.get("stability", "UNKNOWN"))

        # Only PASSED (1.0) counts as evidence FOR root/lead status
        root_conditions = sum([
            tournament_score >= 1.0,
            hidden_score >= 1.0,
            evidence.get("universality", "UNKNOWN") == "PASSED",
            evidence.get("stability", "UNKNOWN") == "PASSED",
            evidence.get("necessity", "UNKNOWN") == "PASSED",
        ])

        # FAILURE vetoes root/lead status
        any_root_veto = any([
            evidence.get("necessity", "") == "FAILED",
            evidence.get("stability", "") == "FAILED",
        ])

        if root_conditions >= 4 and not any_root_veto:
            classification = "ROOT_GENERATOR"
            confidence = avg_score
            summary = "Compression is the root generator"
        elif root_conditions >= 2 and not any_root_veto:
            classification = "LEAD_GENERATOR"
            confidence = avg_score
            summary = "Compression is a lead but not confirmed root"
        elif any_root_veto:
            classification = "OBSERVABLE_CONSEQUENCE"
            confidence = 1.0 - avg_score
            summary = "Compression is an observable consequence, not root"
        else:
            classification = "INCONCLUSIVE"
            confidence = 0.5
            summary = "Evidence insufficient to classify compression's role"

        metrics = {
            "evidence": evidence,
            "passed": passed,
            "failed": failed,
            "inconclusive": inconclusive,
            "total": total,
            "confidence": confidence,
            "classification": classification,
            "summary": summary,
            "tournament_score": tournament_score,
            "hidden_driver_score": hidden_score,
            "necessity_score": necessity_score,
            "stability_score": stability_score,
        }

        print(f"  Root Physics Model:")
        print(f"    Passed: {passed}/{total}, Failed: {failed}, Inconclusive: {inconclusive}")
        print(f"    Confidence: {confidence:.2f}")
        print(f"    Classification: {classification}")
        print(f"    Summary: {summary}")
        print()

        for rq, status in evidence.items():
            if status != "UNKNOWN":
                print(f"    {rq:20s}: {status}")

        return CPIResult("root_physics_model", "COMPLETE", metrics=metrics)
