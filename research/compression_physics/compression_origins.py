"""RQ1: Which variables consistently precede compression?"""

from __future__ import annotations

from typing import Any

import numpy as np

from research.compression_physics.compression_validator import CompressionValidator, CPIResult


CANDIDATE_GENERATORS = [
    "entropy_change", "behavior_density", "cohort_alignment",
    "cohort_conflict", "memory_alignment", "memory_gradient",
    "information_pressure", "liquidity_entropy", "tension_score",
]


class CompressionOrigins:
    def __init__(self, validator: CompressionValidator, asset: str = "EURJPY"):
        self.validator = validator
        self.asset = asset
        self._max_lag = 200

    def run(self) -> CPIResult:
        data = self.validator.load_asset_data(self.asset)
        signals = self.validator.compute_signals(data)

        compression = np.asarray(signals["compression"], dtype=np.float64)
        n = len(compression)

        results = []
        for gen in CANDIDATE_GENERATORS:
            if gen not in signals:
                continue
            sig = np.asarray(signals[gen], dtype=np.float64)
            common = min(len(sig), n)
            if common < self._max_lag * 2 + 1:
                continue

            from research.compression_physics.compression_validator import _find_peak_lag
            lag, corr = _find_peak_lag(sig[:common], compression[:common], self._max_lag)
            flow = self.validator.information_flow(gen, "compression", signals)

            results.append({
                "generator": gen,
                "peak_lag": lag,
                "peak_corr": corr,
                "information_flow": flow,
                "leads_compression": lag < 0,
            })

        results.sort(key=lambda x: abs(x["peak_corr"]) if x["leads_compression"] else 0, reverse=True)

        lead_generators = [r for r in results if r["leads_compression"]]
        best_generator = lead_generators[0] if lead_generators else (results[0] if results else None)

        metrics = {
            "all_candidates": results,
            "n_lead_generators": len(lead_generators),
            "best_generator": best_generator["generator"] if best_generator else None,
            "best_lag": best_generator["peak_lag"] if best_generator else None,
            "best_corr": best_generator["peak_corr"] if best_generator else None,
        }

        print("  Compression origin candidates (sorted by |corr|, leaders only):")
        for r in lead_generators[:5]:
            print(f"    {r['generator']:25s}: lag={r['peak_lag']:4d}, corr={r['peak_corr']:.4f}, flow={r['information_flow']:.6f}")

        if best_generator and abs(best_generator["peak_corr"]) > 0.1:
            status = "PASSED"
            print(f"  Best pre-compression generator: {best_generator['generator']} (lag={best_generator['peak_lag']}, corr={best_generator['peak_corr']:.4f})")
        elif lead_generators:
            status = "INCONCLUSIVE"
        else:
            status = "FAILED"

        return CPIResult("compression_origins", status, metrics=metrics)
