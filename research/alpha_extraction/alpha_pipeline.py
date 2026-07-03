"""AEL pipeline: run all 10 alpha extraction research questions."""

from __future__ import annotations

import time
import json
from pathlib import Path

from research.alpha_extraction.alpha_validator import AlphaValidator, AELResult
from research.alpha_extraction.outcome_distributions import OutcomeDistributions
from research.alpha_extraction.expectancy_surface import ExpectancySurface
from research.alpha_extraction.interaction_discovery import InteractionDiscovery
from research.alpha_extraction.conditional_expectancy import ConditionalExpectancy
from research.alpha_extraction.distribution_shift import DistributionShift
from research.alpha_extraction.state_combinations import StateCombinations
from research.alpha_extraction.candidate_generation import CandidateGeneration
from research.alpha_extraction.cross_asset_validation import CrossAssetValidation
from research.alpha_extraction.cross_time_validation import CrossTimeValidation
from research.alpha_extraction.alpha_classifier import AlphaClassifier


class AlphaPipeline:
    def __init__(self, asset: str = "EURJPY"):
        self.asset = asset
        self.validator = AlphaValidator()
        self.results: dict[str, object] = {}

    def run_all(self) -> dict[str, object]:
        print(f"Alpha Extraction Lab (AEL) Pipeline")
        print(f"Asset: {self.asset}")
        print(f"{'='*60}")

        # RQ1-7: single-asset analysis
        runners = [
            ("RQ1: Outcome Distributions", OutcomeDistributions(self.validator, self.asset)),
            ("RQ2: Expectancy Surface", ExpectancySurface(self.validator, self.asset)),
            ("RQ3: Interaction Discovery", InteractionDiscovery(self.validator, self.asset)),
            ("RQ4: Conditional Expectancy", ConditionalExpectancy(self.validator, self.asset)),
            ("RQ5: Distribution Shift", DistributionShift(self.validator, self.asset)),
            ("RQ6: State Combinations", StateCombinations(self.validator, self.asset)),
            ("RQ7: Candidate Generation", CandidateGeneration(self.validator, self.asset)),
        ]

        for label, runner in runners:
            t0 = time.time()
            print(f"\n  [{label}]")
            try:
                result = runner.run()
                elapsed = time.time() - t0
                print(f"  -> Status: {result.status} ({elapsed:.2f}s)")
                self.results[label] = result
            except Exception as e:
                elapsed = time.time() - t0
                print(f"  -> ERROR: {e} ({elapsed:.2f}s)")
                self.results[label] = AELResult(label, "ERROR", metrics={"error": str(e)})

        # RQ8: Cross-asset
        print(f"\n  [RQ8: Cross-Asset Validation]")
        t0 = time.time()
        try:
            ca = CrossAssetValidation(self.validator)
            ca_result = ca.run()
            elapsed = time.time() - t0
            print(f"  -> Status: {ca_result.status} ({elapsed:.2f}s)")
            self.results["RQ8: Cross-Asset Validation"] = ca_result
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  -> ERROR: {e} ({elapsed:.2f}s)")
            self.results["RQ8: Cross-Asset Validation"] = AELResult(
                "RQ8", "ERROR", metrics={"error": str(e)})
            ca_result = None

        # RQ9: Cross-time
        print(f"\n  [RQ9: Cross-Time Validation]")
        t0 = time.time()
        try:
            ct = CrossTimeValidation(self.validator, self.asset)
            ct_result = ct.run()
            elapsed = time.time() - t0
            print(f"  -> Status: {ct_result.status} ({elapsed:.2f}s)")
            self.results["RQ9: Cross-Time Validation"] = ct_result
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  -> ERROR: {e} ({elapsed:.2f}s)")
            self.results["RQ9: Cross-Time Validation"] = AELResult(
                "RQ9", "ERROR", metrics={"error": str(e)})
            ct_result = None

        # RQ10: Alpha Classifier (uses RQ8+RQ9 cross-reference)
        ca_metrics = ca_result.metrics if ca_result else {}
        ct_metrics = ct_result.metrics if ct_result else {}

        print(f"\n  [RQ10: Alpha Classifier]")
        t0 = time.time()
        try:
            ac = AlphaClassifier(self.validator, self.asset, ca_metrics, ct_metrics)
            ac_result = ac.run()
            elapsed = time.time() - t0
            print(f"  -> Status: {ac_result.status} ({elapsed:.2f}s)")
            self.results["RQ10: Alpha Classifier"] = ac_result
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  -> ERROR: {e} ({elapsed:.2f}s)")
            self.results["RQ10: Alpha Classifier"] = AELResult(
                "RQ10", "ERROR", metrics={"error": str(e)})

        return self.results

    def save(self, path: str | Path) -> None:
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        serializable = {}
        for k, v in self.results.items():
            if hasattr(v, "status") and hasattr(v, "metrics"):
                serializable[k] = {"status": v.status, "metrics": v.metrics}
            else:
                serializable[k] = {"status": "UNKNOWN", "metrics": {}}

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, default=str)

        print(f"\nResults saved to {save_path}")


def run_alpha_pipeline(asset: str = "EURJPY", save_path: str | None = None) -> AlphaPipeline:
    pipeline = AlphaPipeline(asset)
    pipeline.run_all()

    if save_path:
        pipeline.save(save_path)

    return pipeline
