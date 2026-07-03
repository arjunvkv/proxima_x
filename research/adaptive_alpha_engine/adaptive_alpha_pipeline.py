"""Adaptive Alpha Engine — orchestrate all 10 research questions."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from research.adaptive_alpha_engine.aae_validator import AAEValidator, AAEResult
from research.adaptive_alpha_engine.threshold_drift import ThresholdDrift
from research.adaptive_alpha_engine.adaptive_percentiles import AdaptivePercentiles
from research.adaptive_alpha_engine.walk_forward import WalkForward
from research.adaptive_alpha_engine.alpha_decay import AlphaDecay
from research.adaptive_alpha_engine.portfolio_builder import PortfolioBuilder
from research.adaptive_alpha_engine.adaptive_time_overlay import AdaptiveTimeOverlay
from research.adaptive_alpha_engine.execution_stress import ExecutionStress
from research.adaptive_alpha_engine.capacity_model import CapacityModel
from research.adaptive_alpha_engine.live_system import LiveSystem
from research.adaptive_alpha_engine.deployment_validator import DeploymentValidator


class AdaptiveAlphaPipeline:
    def __init__(self, asset: str = "EURJPY"):
        self.asset = asset
        self.validator = AAEValidator()
        self.results: dict[str, AAEResult] = {}

    def run_all(self) -> dict[str, AAEResult]:
        print(f"Adaptive Alpha Engine — Full Pipeline")
        print(f"Asset: {self.asset}")
        print(f"{'='*60}")

        runners: list[tuple[str, Any]] = [
            ("RQ1: Threshold Drift", ThresholdDrift(self.validator, self.asset)),
            ("RQ2: Adaptive Percentiles", AdaptivePercentiles(self.validator, self.asset)),
            ("RQ3: Walk-Forward", WalkForward(self.validator, self.asset)),
            ("RQ4: Alpha Decay", AlphaDecay(self.validator, self.asset)),
            ("RQ5: Portfolio Builder", PortfolioBuilder(self.validator)),
            ("RQ6: Adaptive Time Overlay", AdaptiveTimeOverlay(self.validator, self.asset)),
            ("RQ7: Execution Stress", ExecutionStress(self.validator, self.asset)),
            ("RQ8: Capacity Model", CapacityModel(self.validator, self.asset)),
            ("RQ9: Live System", LiveSystem(self.validator, self.asset)),
        ]

        for label, runner in runners:
            t0 = time.time()
            print(f"\n  [{label}]")
            try:
                result = runner.run()
                elapsed = time.time() - t0
                status = getattr(result, "status", "UNKNOWN")
                print(f"  -> Status: {status} ({elapsed:.2f}s)")
                self.results[label] = result
            except Exception as e:
                elapsed = time.time() - t0
                import traceback
                print(f"  -> ERROR: {e} ({elapsed:.2f}s)")
                traceback.print_exc()
                self.results[label] = AAEResult(
                    rq_name=label, status="ERROR", metrics={"error": str(e)},
                )

        # RQ10: Deployment Validator (uses all prior results)
        print(f"\n  [RQ10: Deployment Validator]")
        t0 = time.time()
        try:
            adj = DeploymentValidator(self.validator, self.asset, self.results)
            r10 = adj.run()
            elapsed = time.time() - t0
            status = getattr(r10, "status", "UNKNOWN")
            print(f"  -> Status: {status} ({elapsed:.2f}s)")
            self.results["RQ10: Deployment Validator"] = r10
        except Exception as e:
            elapsed = time.time() - t0
            import traceback
            print(f"  -> ERROR: {e} ({elapsed:.2f}s)")
            traceback.print_exc()
            self.results["RQ10: Deployment Validator"] = AAEResult(
                rq_name="RQ10: Deployment Validator",
                status="ERROR",
                metrics={"error": str(e)},
            )

        return self.results

    def save(self, path: str | Path) -> None:
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        serializable: dict[str, dict[str, Any]] = {}
        for k, v in self.results.items():
            if hasattr(v, "rq_name") and hasattr(v, "status") and hasattr(v, "metrics"):
                serializable[k] = {
                    "rq_name": v.rq_name,
                    "status": v.status,
                    "metrics": v.metrics,
                }
            else:
                serializable[k] = {"status": "UNKNOWN", "metrics": {}}

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, default=str)

        print(f"\nResults saved to {save_path}")
