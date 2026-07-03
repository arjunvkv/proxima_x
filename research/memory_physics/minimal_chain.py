from __future__ import annotations

from typing import Any

import numpy as np

from research.memory_physics.memory_validator import MemoryValidator, MPRResult
from research.temporal_reality.causality_analysis import AdaptiveTimeCausality


class MinimalChain:
    """RQ10: Find the smallest chain that still explains adaptive_time, state_mutation, regime_change.

    Test whether memory_density, memory_conflict, and adaptive_time can be removed.
    """

    def __init__(self, validator: MemoryValidator, asset: str = "EURJPY"):
        self.validator = validator
        self.asset = asset
        self._max_lag = 200

    def run(self) -> MPRResult:
        data = self.validator.load_asset_data(self.asset)
        signals = self.validator.compute_signals(data)

        all_vars = ["energy_storage", "memory_density", "memory_conflict", "adaptive_time",
                    "state_mutation_rate", "regime_change_probability"]
        targets = ["adaptive_time", "state_mutation_rate", "regime_change_probability"]

        def _predict_score(target: str, predictors: list[str]) -> float:
            best = 0.0
            for p in predictors:
                if p == target or p not in signals:
                    continue
                flow = self.validator.information_flow(p, target, signals)
                best = max(best, flow)
            return best

        def _chain_score(predictors: list[str]) -> dict[str, float]:
            scores = {}
            for t in targets:
                scores[t] = _predict_score(t, predictors)
            scores["composite"] = float(np.mean(list(scores.values())))
            return scores

        # Full chain
        full_preds = [v for v in all_vars if v not in targets]
        full_score = _chain_score(full_preds)
        baseline = full_score["composite"]

        removal_tests = [
            ("remove_memory_density", "memory_density"),
            ("remove_memory_conflict", "memory_conflict"),
            ("remove_adaptive_time", "adaptive_time"),
            ("remove_both_memory", ["memory_density", "memory_conflict"]),
            ("remove_energy_storage", "energy_storage"),
            ("minimal_chain", ["energy_storage"]),
            ("conflict_only", ["memory_conflict"]),
            ("density_only", ["memory_density"]),
        ]

        results = []
        for name, removed in removal_tests:
            if isinstance(removed, str):
                remaining = [v for v in full_preds if v != removed]
            else:
                remaining = [v for v in full_preds if v not in removed]
            if not remaining:
                results.append({"test": name, "composite": 0.0, "loss_vs_baseline": 1.0, "scores": {}})
                continue
            scores = _chain_score(remaining)
            loss = (baseline - scores["composite"]) / max(baseline, 1e-12)
            results.append({"test": name, "composite": scores["composite"],
                            "loss_vs_baseline": max(0.0, loss), "scores": scores,
                            "remaining_predictors": remaining})

        metrics = {
            "baseline_composite": baseline,
            "full_predictors": full_preds,
            "removal_results": results,
        }

        for r in results:
            print(f"  {r['test']}: composite={r['composite']:.4f}, loss={r.get('loss_vs_baseline', 0):.4f}")

        # Determine minimal chain: find smallest set that keeps loss < 0.3
        minimal = None
        for r in results:
            if r.get("loss_vs_baseline", 1.0) < 0.3:
                minimal = r
                break

        if minimal:
            metrics["minimal_chain_predictors"] = minimal.get("remaining_predictors", [])
            metrics["minimal_chain_composite"] = minimal["composite"]
            metrics["can_remove_memory_density"] = any(
                r["test"] == "remove_memory_density" and r.get("loss_vs_baseline", 1.0) < 0.3
                for r in results)
            metrics["can_remove_memory_conflict"] = any(
                r["test"] == "remove_memory_conflict" and r.get("loss_vs_baseline", 1.0) < 0.3
                for r in results)
            print(f"  Minimal chain: {minimal.get('remaining_predictors', [])}")

        status = "PASSED" if minimal is not None else "FAILED"

        return MPRResult("minimal_chain", status, metrics=metrics)
