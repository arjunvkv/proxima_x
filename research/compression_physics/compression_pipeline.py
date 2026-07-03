"""CPI pipeline: run all 10 compression physics investigations."""

from __future__ import annotations

import time
import json
from pathlib import Path

from research.compression_physics.compression_validator import CompressionValidator
from research.compression_physics.compression_origins import CompressionOrigins
from research.compression_physics.compression_lifecycle import CompressionLifecycle
from research.compression_physics.compression_necessity import CompressionNecessity
from research.compression_physics.compression_mediation import CompressionMediation
from research.compression_physics.asset_universality import AssetUniversality
from research.compression_physics.time_stability import TimeStability
from research.compression_physics.generator_tournament import GeneratorTournament
from research.compression_physics.minimal_chain import MinimalChain
from research.compression_physics.hidden_driver_search import HiddenDriverSearch
from research.compression_physics.root_physics_model import RootPhysicsModel


class CompressionPipeline:
    def __init__(self, asset: str = "EURJPY"):
        self.asset = asset
        self.validator = CompressionValidator()
        self.results: dict[str, object] = {}

    def run_all(self) -> dict[str, object]:
        print(f"Compression Physics Investigation (CPI) Pipeline")
        print(f"Asset: {self.asset}")
        print(f"{'='*60}")

        runners = [
            ("RQ1: Origins", CompressionOrigins(self.validator, self.asset)),
            ("RQ2: Lifecycle", CompressionLifecycle(self.validator, self.asset)),
            ("RQ3: Necessity", CompressionNecessity(self.validator, self.asset)),
            ("RQ4: Mediation", CompressionMediation(self.validator, self.asset)),
            ("RQ5: Asset Universality", AssetUniversality(self.validator)),
            ("RQ6: Time Stability", TimeStability(self.validator, self.asset)),
            ("RQ7: Generator Tournament", GeneratorTournament(self.validator, self.asset)),
            ("RQ8: Minimal Chain", MinimalChain(self.validator, self.asset)),
            ("RQ9: Hidden Driver", HiddenDriverSearch(self.validator, self.asset)),
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
                from research.compression_physics.compression_validator import CPIResult
                self.results[label] = CPIResult(label, "ERROR", metrics={"error": str(e)})

        # Attach results to validator for RQ10 cross-reference
        self.validator.results = {}
        for k, v in self.results.items():
            self.validator.results[k] = v

        # RQ10 uses all prior results
        print(f"\n  [RQ10: Root Physics Model]")
        try:
            root_model = RootPhysicsModel(self.validator, self.asset)
            root_result = root_model.run()
            self.results["RQ10: Root Physics Model"] = root_result
            self.validator.results["RQ10: Root Physics Model"] = root_result
            print(f"  -> Status: {root_result.status}")
        except Exception as e:
            print(f"  -> ERROR: {e}")

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


def run_compression_pipeline(asset: str = "EURJPY", save_path: str | None = None) -> CompressionPipeline:
    pipeline = CompressionPipeline(asset)
    pipeline.run_all()

    if save_path:
        pipeline.save(save_path)

    return pipeline
