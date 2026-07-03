from __future__ import annotations

from typing import Any

import numpy as np

from research.memory_physics.memory_validator import MemoryValidator, MPRResult
from research.temporal_reality.causality_analysis import AdaptiveTimeCausality


class MemoryCollapse:
    """RQ6: Which removal hurts the chain more — memory_conflict or memory_density?

    Remove each individually and measure adaptive_time_loss, mutation_loss, regime_loss.
    """

    def __init__(self, validator: MemoryValidator, asset: str = "EURJPY"):
        self.validator = validator
        self.asset = asset
        self._max_lag = 200

    def run(self) -> MPRResult:
        data = self.validator.load_asset_data(self.asset)
        signals = self.validator.compute_signals(data)

        targets = ["adaptive_time", "state_mutation_rate", "regime_change_probability"]
        removals = [
            ("memory_conflict", {"memory_conflict"}),
            ("memory_density", {"memory_density"}),
        ]

        # Baseline: best predictor for each target using ALL sources
        all_sources = ["energy_storage", "memory_density", "memory_conflict", "adaptive_time",
                       "state_mutation_rate", "regime_change_probability"]

        def _best_prediction(target: str, excluded: set[str]) -> float:
            best = 0.0
            for src in all_sources:
                if src == target or src in excluded:
                    continue
                if src not in signals:
                    continue
                flow = self.validator.information_flow(src, target, signals)
                best = max(best, flow)
            return best

        baseline = {t: _best_prediction(t, set()) for t in targets}
        results = []

        for removal_name, removed_set in removals:
            losses = {}
            for t in targets:
                before = baseline[t]
                after = _best_prediction(t, removed_set)
                loss = (before - after) / max(before, 1e-12)
                losses[f"{t}_loss"] = loss
                losses[f"{t}_before"] = before
                losses[f"{t}_after"] = after
            results.append({"removed": removal_name, **losses})

        conflict_losses = results[0]
        density_losses = results[1]

        conflict_total = sum(conflict_losses.get(f"{t}_loss", 0) for t in targets)
        density_total = sum(density_losses.get(f"{t}_loss", 0) for t in targets)

        metrics = {
            "baseline_predictions": baseline,
            "conflict_removal": {k: v for k, v in conflict_losses.items()},
            "density_removal": {k: v for k, v in density_losses.items()},
            "conflict_total_loss": conflict_total,
            "density_total_loss": density_total,
            "which_hurts_more": "memory_density" if density_total > conflict_total else "memory_conflict",
        }

        if density_total > conflict_total * 1.2:
            status = "FAILED"  # density hurts more -> conflict is less important
            print(f"  Density removal hurts more (d={density_total:.3f} vs c={conflict_total:.3f})")
        elif conflict_total > density_total * 1.2:
            status = "PASSED"  # conflict hurts more -> conflict is essential
            print(f"  Conflict removal hurts more (c={conflict_total:.3f} vs d={density_total:.3f})")
        else:
            status = "INCONCLUSIVE"
            print(f"  Similar impact (c={conflict_total:.3f} vs d={density_total:.3f})")

        return MPRResult("memory_collapse", status, metrics=metrics)
