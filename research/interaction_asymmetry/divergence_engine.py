from __future__ import annotations

from typing import Any

import numpy as np

from research.interaction_asymmetry.interaction_validator import InteractionValidator, IAEResult, PAIRS
from research.adaptive_alpha_engine.aae_validator import HORIZONS, _zscore


class DivergenceEngine:
    def __init__(self, validator: InteractionValidator, asset: str = "EURJPY"):
        self.validator = validator
        self.asset = asset

    def run(self) -> IAEResult:
        self.validator.load(self.asset)
        md_z = self.validator.md_z()
        es_z = self.validator.es_z()
        at_z = self.validator.at_z()

        var_map = {"memory_density": md_z, "energy_storage": es_z, "adaptive_time": at_z}
        methods = ["difference", "zscore_difference", "acceleration_difference"]

        divergence_results = []
        for pair in PAIRS:
            a, b = pair
            for method in methods:
                div_signal = self.validator.divergence(var_map[a], var_map[b], method)
                div_signal = np.nan_to_num(div_signal, nan=0.0, posinf=0.0, neginf=0.0)
                if len(div_signal) < 10:
                    continue
                alpha = self.validator.eval_alpha(div_signal, 2)
                divergence_results.append({
                    "pair": f"{a}_vs_{b}",
                    "method": method,
                    "mean": alpha.get("mean", 0.0),
                    "pp": alpha.get("pp", 0.5),
                    "sharpe": alpha.get("sharpe", 0.0),
                    "std": alpha.get("std", 0.0),
                    "n": alpha.get("n", 0),
                })

        benchmark_es = self.validator.benchmark_es_alpha()
        b_pp = benchmark_es.get("pp", 0.5)
        b_sharpe = benchmark_es.get("sharpe", 0.0)

        for r in divergence_results:
            beats = r["pp"] > b_pp or r["sharpe"] > b_sharpe * 1.1
            r["beats_ES"] = beats

        print(f"{'Method':>30s} | {'Pair':>30s} | {'Mean':>10s} | {'PP':>8s} | {'Sharpe':>8s} | {'N':>6s} | {'Beats_ES':>9s}")
        print("-" * 110)
        for r in divergence_results:
            print(f"{r['method']:>30s} | {r['pair']:>30s} | {r['mean']:>10.6f} | {r['pp']:>8.4f} | {r['sharpe']:>8.4f} | {r['n']:>6d} | {str(r['beats_ES']):>9s}")
        print(f"\nBenchmark ES alpha (H20): mean={benchmark_es.get('mean', 0.0):.6f} pp={b_pp:.4f} sharpe={b_sharpe:.4f}")

        n_beats = sum(1 for r in divergence_results if r["beats_ES"])
        best = max(divergence_results, key=lambda x: x["sharpe"]) if divergence_results else {}

        if best:
            print(f"\nBest divergence: {best['method']} | {best['pair']} | sharpe={best['sharpe']:.4f}")
        print(f"Divergences beating ES: {n_beats} / {len(divergence_results)}")

        metrics: dict[str, Any] = {
            "benchmark_es_alpha": benchmark_es,
            "divergence_results": divergence_results,
            "best_divergence": best,
            "n_beats_es": n_beats,
        }

        return IAEResult(rq_name="RQ1_Divergence_Alpha", status="COMPLETE", metrics=metrics)
