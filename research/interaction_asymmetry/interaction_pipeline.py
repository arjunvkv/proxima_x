from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from research.interaction_asymmetry.interaction_validator import InteractionValidator, IAEResult
from research.interaction_asymmetry.divergence_engine import DivergenceEngine
from research.interaction_asymmetry.synchronization_engine import SynchronizationEngine
from research.interaction_asymmetry.friction_engine import FrictionEngine
from research.interaction_asymmetry.leadership_engine import LeadershipEngine
from research.interaction_asymmetry.contradiction_engine import ContradictionEngine
from research.interaction_asymmetry.tension_surface import TensionSurface
from research.interaction_asymmetry.transition_pressure import TransitionPressure
from research.interaction_asymmetry.cross_asset_validator import CrossAssetValidator
from research.interaction_asymmetry.cross_time_validator import CrossTimeValidator
from research.interaction_asymmetry.interaction_classifier import InteractionClassifier
import numpy as np


def _clean(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    if isinstance(obj, tuple):
        return list(_clean(v) for v in obj)
    if isinstance(obj, (np.ndarray, np.generic)):
        return obj.tolist() if hasattr(obj, 'tolist') else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if hasattr(obj, 'to_dict'):
        return obj.to_dict()
    return obj


class InteractionAsymmetryPipeline:
    def __init__(self, asset: str = "EURJPY"):
        self.asset = asset
        self.validator = InteractionValidator(asset)
        self.results: dict[str, IAEResult] = {}

    def run_all(self) -> dict[str, IAEResult]:
        print(f"Interaction Asymmetry Lab — Full Pipeline")
        print(f"Asset: {self.asset}")
        print(f"{'='*60}")

        runners: list[tuple[str, Any]] = [
            ("RQ1: Divergence Alpha", DivergenceEngine(self.validator, self.asset)),
            ("RQ2: Synchronization States", SynchronizationEngine(self.validator, self.asset)),
            ("RQ3: Temporal Friction", FrictionEngine(self.validator, self.asset)),
            ("RQ4: Leadership Rotation", LeadershipEngine(self.validator, self.asset)),
            ("RQ5: Hidden Contradictions", ContradictionEngine(self.validator, self.asset)),
            ("RQ6: Tension Surface", TensionSurface(self.validator, self.asset)),
            ("RQ7: Transition Pressure", TransitionPressure(self.validator, self.asset)),
            ("RQ8: Cross-Asset Universality", CrossAssetValidator(self.validator)),
            ("RQ9: Cross-Time Survival", CrossTimeValidator(self.validator)),
        ]

        for label, runner in runners:
            t0 = time.time()
            print(f"\n{'='*72}")
            print(f"  [{label}]")
            print(f"{'='*72}")
            try:
                result = runner.run()
                elapsed = time.time() - t0
                status = getattr(result, "status", "UNKNOWN")
                print(f"\n  -> Status: {status} ({elapsed:.2f}s)")
                self.results[label] = result
            except Exception as e:
                elapsed = time.time() - t0
                import traceback
                traceback.print_exc()
                print(f"\n  -> ERROR: {e} ({elapsed:.2f}s)")
                self.results[label] = IAEResult(
                    rq_name=label, status="ERROR", metrics={"error": str(e)},
                )

        print(f"\n{'='*72}")
        print(f"  [RQ10: Interaction Alpha Adjudication]")
        print(f"{'='*72}")
        t0 = time.time()
        try:
            classifier = InteractionClassifier(self.validator, self.results)
            r10 = classifier.run()
            elapsed = time.time() - t0
            status = getattr(r10, "status", "UNKNOWN")
            print(f"\n  -> Status: {status} ({elapsed:.2f}s)")
            self.results["RQ10: Interaction Adjudication"] = r10
        except Exception as e:
            elapsed = time.time() - t0
            import traceback
            traceback.print_exc()
            print(f"\n  -> ERROR: {e} ({elapsed:.2f}s)")
            self.results["RQ10: Interaction Adjudication"] = IAEResult(
                rq_name="RQ10: Interaction Adjudication", status="ERROR", metrics={"error": str(e)},
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
                    "metrics": _clean(v.metrics),
                }
            else:
                serializable[k] = {"status": "UNKNOWN", "metrics": {}}

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, default=str)

        print(f"\nResults saved to {save_path}")
