"""ARL pipeline: run all 10 alpha reality destruction tests."""

from __future__ import annotations

import time
import json
from pathlib import Path

from research.alpha_reality.arl_validator import ARLValidator, ARLResult
from research.alpha_reality.trend_independence import TrendIndependence
from research.alpha_reality.volatility_independence import VolatilityIndependence
from research.alpha_reality.randomization_test import RandomizationTest
from research.alpha_reality.execution_reality import ExecutionReality
from research.alpha_reality.cross_asset_transfer import CrossAssetTransfer
from research.alpha_reality.cross_time_transfer import CrossTimeTransfer
from research.alpha_reality.threshold_stability import ThresholdStability
from research.alpha_reality.interaction_necessity import InteractionNecessity
from research.alpha_reality.capacity_analysis import CapacityAnalysis
from research.alpha_reality.alpha_adjudication import AlphaAdjudication


class ARLPipeline:
    def __init__(self, asset: str = "EURJPY"):
        self.asset = asset
        self.validator = ARLValidator()
        self.results: dict[str, ARLResult] = {}

    def run_all(self) -> dict[str, ARLResult]:
        print(f"Alpha Reality Lab (ARL) — Destruction Testing")
        print(f"Asset: {self.asset}")
        print(f"{'='*60}")

        runners = [
            ("RQ1: Trend Independence", TrendIndependence(self.validator, self.asset)),
            ("RQ2: Volatility Independence", VolatilityIndependence(self.validator, self.asset)),
            ("RQ3: Randomization Test", RandomizationTest(self.validator, self.asset)),
            ("RQ4: Execution Reality", ExecutionReality(self.validator, self.asset)),
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
                import traceback
                print(f"  -> ERROR: {e} ({elapsed:.2f}s)")
                traceback.print_exc()
                self.results[label] = ARLResult(label, "ERROR", metrics={"error": str(e)})

        # RQ5: Cross-asset (needs validator only)
        print(f"\n  [RQ5: Cross-Asset Transfer]")
        t0 = time.time()
        try:
            ca = CrossAssetTransfer(self.validator)
            r5 = ca.run()
            elapsed = time.time() - t0
            print(f"  -> Status: {r5.status} ({elapsed:.2f}s)")
            self.results["RQ5: Cross-Asset Transfer"] = r5
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  -> ERROR: {e} ({elapsed:.2f}s)")
            self.results["RQ5: Cross-Asset Transfer"] = ARLResult("RQ5", "ERROR", metrics={"error": str(e)})

        # RQ6: Cross-time
        print(f"\n  [RQ6: Cross-Time Transfer]")
        t0 = time.time()
        try:
            ct = CrossTimeTransfer(self.validator, self.asset)
            r6 = ct.run()
            elapsed = time.time() - t0
            print(f"  -> Status: {r6.status} ({elapsed:.2f}s)")
            self.results["RQ6: Cross-Time Transfer"] = r6
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  -> ERROR: {e} ({elapsed:.2f}s)")
            self.results["RQ6: Cross-Time Transfer"] = ARLResult("RQ6", "ERROR", metrics={"error": str(e)})

        # RQ7-9: on single asset
        runners2 = [
            ("RQ7: Threshold Stability", ThresholdStability(self.validator, self.asset)),
            ("RQ8: Interaction Necessity", InteractionNecessity(self.validator, self.asset)),
            ("RQ9: Capacity Analysis", CapacityAnalysis(self.validator, self.asset)),
        ]

        for label, runner in runners2:
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
                self.results[label] = ARLResult(label, "ERROR", metrics={"error": str(e)})

        # RQ10: Final adjudication
        print(f"\n  [RQ10: Alpha Adjudication]")
        t0 = time.time()
        try:
            adj = AlphaAdjudication(self.validator, self.asset, self.results)
            r10 = adj.run()
            elapsed = time.time() - t0
            print(f"  -> Status: {r10.status} ({elapsed:.2f}s)")
            self.results["RQ10: Alpha Adjudication"] = r10
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  -> ERROR: {e} ({elapsed:.2f}s)")
            self.results["RQ10: Alpha Adjudication"] = ARLResult("RQ10", "ERROR", metrics={"error": str(e)})

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


def run_arl_pipeline(asset: str = "EURJPY", save_path: str | None = None) -> ARLPipeline:
    pipeline = ARLPipeline(asset)
    pipeline.run_all()

    if save_path:
        pipeline.save(save_path)

    return pipeline
