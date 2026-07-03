from __future__ import annotations

from typing import Any

import numpy as np

from research.memory_physics.memory_validator import MemoryValidator, MPRResult


class ConflictMutationAnalysis:
    """RQ3: Compare memory_conflict vs memory_density as predictors of state_mutation."""

    def __init__(self, validator: MemoryValidator, asset: str = "EURJPY"):
        self.validator = validator
        self.asset = asset
        self._max_lag = 200

    def run(self) -> MPRResult:
        data = self.validator.load_asset_data(self.asset)
        signals = self.validator.compute_signals(data)

        smr = np.asarray(signals["state_mutation_rate"], dtype=np.float64)
        conflict = np.asarray(signals["memory_conflict"], dtype=np.float64)
        density = np.asarray(signals["memory_density"], dtype=np.float64)
        n = min(len(smr), len(conflict), len(density))

        if n < self._max_lag * 2 + 1:
            return MPRResult("conflict_mutation", "FAILED", metrics={"error": "insufficient data"})

        from research.memory_physics.memory_validator import _find_peak_lag
        lag_conflict, r_conflict = _find_peak_lag(conflict[:n], smr[:n], self._max_lag)
        lag_density, r_density = _find_peak_lag(density[:n], smr[:n], self._max_lag)

        flow_conflict_smr = self.validator.information_flow("memory_conflict", "state_mutation_rate", signals)
        flow_density_smr = self.validator.information_flow("memory_density", "state_mutation_rate", signals)

        if abs(r_density) > 0:
            relative_power = abs(r_conflict) / abs(r_density)
        else:
            relative_power = float('inf')

        if abs(r_conflict) > abs(r_density):
            better_predictor = "memory_conflict"
        else:
            better_predictor = "memory_density"

        metrics = {
            "conflict_to_smr_peak_lag": lag_conflict,
            "conflict_to_smr_peak_corr": r_conflict,
            "density_to_smr_peak_lag": lag_density,
            "density_to_smr_peak_corr": r_density,
            "flow_conflict_to_smr": flow_conflict_smr,
            "flow_density_to_smr": flow_density_smr,
            "relative_corr_power": relative_power,
            "better_predictor": better_predictor,
        }

        if abs(r_conflict) > abs(r_density) * 1.1:
            status = "PASSED"
            print(f"  Conflict beats density for mutation (r_c={r_conflict:.4f} vs r_d={r_density:.4f})")
        elif abs(r_conflict) > abs(r_density) * 0.9:
            status = "INCONCLUSIVE"
            print(f"  Conflict and density similar for mutation (r_c={r_conflict:.4f} vs r_d={r_density:.4f})")
        else:
            status = "FAILED"
            print(f"  Density beats conflict for mutation (r_d={r_density:.4f} vs r_c={r_conflict:.4f})")

        return MPRResult("conflict_mutation", status, metrics=metrics)
