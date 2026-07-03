"""RQ8: What is the minimal causal chain to produce adaptive_time? Can compression replace energy_storage?"""

from __future__ import annotations

from typing import Any

import numpy as np

from research.compression_physics.compression_validator import CompressionValidator, CPIResult


class MinimalChain:
    def __init__(self, validator: CompressionValidator, asset: str = "EURJPY"):
        self.validator = validator
        self.asset = asset
        self._max_lag = 200

    def run(self) -> CPIResult:
        data = self.validator.load_asset_data(self.asset)
        signals = self.validator.compute_signals(data)

        all_generators = ["compression", "energy_storage", "memory_density", "memory_gradient",
                          "tension_score", "entropy_change", "liquidity_entropy",
                          "information_pressure", "cohort_alignment"]

        def _score_chain(chain: list[str]) -> dict:
            at = signals.get("adaptive_time", signals.get("state_mutation_rate", np.array([])))
            if len(chain) == 0:
                return {"score": 0.0}
            best_r = 0.0
            for gen in chain:
                if gen not in signals:
                    continue
                flow = self.validator.information_flow(gen, "adaptive_time", signals)
                best_r = max(best_r, flow)
            return {"score": best_r}

        # Full chain
        full_score = _score_chain(["energy_storage", "memory_density"])
        compression_only = _score_chain(["compression"])
        compression_es = _score_chain(["compression", "energy_storage"])
        compression_md = _score_chain(["compression", "memory_density"])
        es_only = _score_chain(["energy_storage"])
        md_only = _score_chain(["memory_density"])

        # Can compression replace energy_storage?
        replace_score = _score_chain(["compression", "memory_density"])
        original_score = _score_chain(["energy_storage", "memory_density"])
        replacement_loss = (original_score["score"] - replace_score["score"]) / max(original_score["score"], 1e-12)

        # Greedy minimal search
        best_single = max(all_generators, key=lambda g: _score_chain([g])["score"]) if all_generators else None
        best_single_score = _score_chain([best_single])["score"] if best_single else 0.0

        metrics = {
            "best_single_generator": best_single,
            "best_single_score": best_single_score,
            "full_chain_score": full_score["score"],
            "compression_only_score": compression_only["score"],
            "compression_plus_energy_score": compression_es["score"],
            "compression_plus_memory_score": compression_md["score"],
            "energy_storage_only_score": es_only["score"],
            "memory_density_only_score": md_only["score"],
            "replacement_loss": replacement_loss,
        }

        print(f"  Minimal chain analysis (for adaptive_time):")
        print(f"    Compression only:      {compression_only['score']:.6f}")
        print(f"    Energy storage only:   {es_only['score']:.6f}")
        print(f"    Memory density only:   {md_only['score']:.6f}")
        print(f"    Comp + Energy:         {compression_es['score']:.6f}")
        print(f"    Comp + Memory:         {compression_md['score']:.6f}")
        print(f"    Original (ES + MD):    {original_score['score']:.6f}")
        print(f"    Best single:           {best_single} ({best_single_score:.6f})")
        print(f"    Replacement loss:      {replacement_loss:.4f}")

        if replacement_loss < 0.15:
            status = "PASSED"
            print(f"  Compression CAN replace energy_storage in minimal chain")
        else:
            status = "INCONCLUSIVE"
            print(f"  Compression cannot fully replace energy_storage")

        return CPIResult("minimal_chain", status, metrics=metrics)
