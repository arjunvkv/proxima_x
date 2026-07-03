"""RQ9: Is there a hidden variable deeper than compression that drives the physics?"""

from __future__ import annotations

from typing import Any

import numpy as np

from research.compression_physics.compression_validator import CompressionValidator, CPIResult


class HiddenDriverSearch:
    def __init__(self, validator: CompressionValidator, asset: str = "EURJPY"):
        self.validator = validator
        self.asset = asset
        self._max_lag = 200

    def run(self) -> CPIResult:
        data = self.validator.load_asset_data(self.asset)
        signals = self.validator.compute_signals(data)

        # Check if any generator predicts compression better than compression predicts downstream
        at = np.asarray(signals.get("adaptive_time", signals.get("state_mutation_rate", np.zeros(100))), dtype=np.float64)

        # Score: how well does compression predict adaptive_time?
        from research.compression_physics.compression_validator import _find_peak_lag
        compression = np.asarray(signals["compression"], dtype=np.float64)
        _, compression_to_at = _find_peak_lag(compression, at, self._max_lag)

        # Now check if any other generator compresses before compression itself
        all_generators = ["entropy_change", "behavior_density", "cohort_alignment",
                          "cohort_conflict", "memory_alignment", "memory_gradient",
                          "information_pressure", "liquidity_entropy", "tension_score"]
        hidden_candidates = []
        for gen in all_generators:
            if gen not in signals:
                continue
            sig = np.asarray(signals[gen], dtype=np.float64)
            lag, r = _find_peak_lag(sig, compression, self._max_lag)
            flow = self.validator.information_flow(gen, "compression", signals)
            hidden_candidates.append({
                "generator": gen,
                "lag_to_compression": lag,
                "r_to_compression": r,
                "information_flow": flow,
                "precedes_compression": lag < 0,
            })

        preceding = [c for c in hidden_candidates if c["precedes_compression"]]
        preceding.sort(key=lambda x: abs(x["r_to_compression"]), reverse=True)

        # Hidden driver test: do residuals from compression still have predictive info?
        from research.causal_physics.latent_driver_search import (
            LatentDriverSearch,
        )
        residuals = np.zeros_like(compression)
        pred = np.roll(compression, -14)  # approximate lag-based prediction
        min_len = min(len(compression), len(pred))
        residuals[:min_len] = compression[:min_len] - pred[:min_len]

        # Test residuals against AT
        _, res_r = _find_peak_lag(residuals, at[:len(residuals)], self._max_lag)

        metrics = {
            "compression_to_adaptive_time_r": compression_to_at,
            "residual_to_adaptive_time_r": res_r,
            "hidden_candidates": hidden_candidates,
            "n_preceding_generators": len(preceding),
            "best_preceding": preceding[0] if preceding else None,
            "residual_predictive_power": res_r,
        }

        print(f"  Hidden driver search:")
        print(f"    Compression -> AT:     r={compression_to_at:.4f}")
        print(f"    Residual -> AT:        r={res_r:.4f}")

        if preceding:
            print(f"    Generators preceding compression:")
            for c in preceding[:3]:
                print(f"      {c['generator']:25s}: lag={c['lag_to_compression']:+4d}, r={c['r_to_compression']:.4f}")

        # Test: if residuals still predict AT, there's hidden info beyond compression
        if abs(res_r) > 0.15:
            status = "FAILED"
            print(f"  Residuals contain predictive info -> there IS a deeper driver (residual r={res_r:.4f})")
        elif abs(res_r) > 0.08:
            status = "INCONCLUSIVE"
        else:
            status = "PASSED"
            print(f"  Compression is near-complete -> no deeper driver needed")

        return CPIResult("hidden_driver_search", status, metrics=metrics)
