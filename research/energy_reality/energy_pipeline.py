from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from research.energy_reality.energy_validator import EnergyValidator, ERLResult
from research.energy_reality.volatility_redundancy import VolatilityRedundancy
from research.energy_reality.residual_alpha import ResidualAlpha
from research.energy_reality.benchmark_comparison import BenchmarkComparison
from research.energy_reality.fragility import Fragility
from research.energy_reality.live_degradation import LiveDegradation
from research.energy_reality.universal_energy import UniversalEnergy


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


class EnergyRealityPipeline:
    def __init__(self, asset: str = "EURJPY"):
        self.asset = asset
        self.validator = EnergyValidator(asset)
        self.results: dict[str, ERLResult] = {}

    def run_all(self) -> dict[str, ERLResult]:
        print(f"Energy Reality Lab — Full Falsification Pipeline")
        print(f"Primary Asset: {self.asset}")
        print(f"{'='*60}")

        runners: list[tuple[str, Any]] = [
            ("ERL-1: Volatility Redundancy", VolatilityRedundancy(self.validator)),
            ("ERL-2: Residual Alpha", ResidualAlpha(self.validator, self.asset)),
            ("ERL-3: Benchmark Comparison", BenchmarkComparison(self.validator)),
            ("ERL-4: Fragility", Fragility(self.validator, self.asset)),
            ("ERL-5: Live Degradation", LiveDegradation(self.validator, self.asset)),
            ("ERL-6: Universal Energy", UniversalEnergy(self.validator)),
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
                self.results[label] = ERLResult(rq_name=label, status="ERROR", metrics={"error": str(e)})

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
