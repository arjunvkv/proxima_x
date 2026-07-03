from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from research.residual_energy.residual_validator import ResidualEnergyValidator, REPResult
from research.residual_energy.residual_constructor import ResidualConstructor
from research.residual_energy.residual_validation import ResidualValidation
from research.residual_energy.residual_decomposition import ResidualDecomposition
from research.residual_energy.residual_alpha import ResidualAlpha
from research.residual_energy.residual_reality import ResidualReality
from research.residual_energy.residual_transfer import ResidualTransfer
from research.residual_energy.residual_deployment import ResidualDeployment
from research.residual_energy.residual_classifier import ResidualClassifier


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
    return obj


class ResidualEnergyPipeline:
    def __init__(self, asset: str = "EURJPY"):
        self.asset = asset
        self.validator = ResidualEnergyValidator(asset)
        self.results: dict[str, REPResult] = {}

    def run_all(self) -> dict[str, REPResult]:
        print(f"Residual Energy Project — Full Pipeline")
        print(f"Primary Asset: {self.asset}")
        print(f"{'='*60}")

        runners: list[tuple[str, Any]] = [
            ("REP-1: Residual Constructor", ResidualConstructor(self.validator, self.asset)),
            ("REP-2: Residual Validation", ResidualValidation(self.validator, self.asset)),
            ("REP-3: Residual Decomposition", ResidualDecomposition(self.validator, self.asset)),
            ("REP-4: Orthogonality Test", ResidualAlpha(self.validator, self.asset)),
            ("REP-5+6: Residual Reality", ResidualReality(self.validator)),
            ("REP-7: Walk-Forward", ResidualTransfer(self.validator, self.asset)),
            ("REP-8+9: Deployment", ResidualDeployment(self.validator, self.asset)),
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
                self.results[label] = REPResult(rq_name=label, status="ERROR", metrics={"error": str(e)})

        print(f"\n{'='*72}")
        print(f"  [REP-10: Final Adjudication]")
        print(f"{'='*72}")
        t0 = time.time()
        try:
            classifier = ResidualClassifier(self.validator, self.results)
            r10 = classifier.run()
            elapsed = time.time() - t0
            status = getattr(r10, "status", "UNKNOWN")
            print(f"\n  -> Status: {status} ({elapsed:.2f}s)")
            self.results["REP-10: Final Adjudication"] = r10
        except Exception as e:
            elapsed = time.time() - t0
            import traceback
            traceback.print_exc()
            print(f"\n  -> ERROR: {e} ({elapsed:.2f}s)")
            self.results["REP-10: Final Adjudication"] = REPResult(
                rq_name="REP-10: Final Adjudication", status="ERROR", metrics={"error": str(e)},
            )

        return self.results

    def save(self, path: str | Path) -> None:
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        serializable: dict[str, dict[str, Any]] = {}
        for k, v in self.results.items():
            if hasattr(v, "rq_name") and hasattr(v, "status") and hasattr(v, "metrics"):
                serializable[k] = {"rq_name": v.rq_name, "status": v.status, "metrics": _clean(v.metrics)}
            else:
                serializable[k] = {"status": "UNKNOWN", "metrics": {}}
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, default=str)
        print(f"\nResults saved to {save_path}")
