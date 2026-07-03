from __future__ import annotations

from typing import Any

import numpy as np

from research.memory_physics.memory_validator import MemoryValidator, MPRResult
from research.temporal_reality.causality_analysis import AdaptiveTimeCausality


class ConflictAdaptiveTimeAnalysis:
    """RQ2: Compare memory_conflict vs memory_density as predictors of adaptive_time."""

    def __init__(self, validator: MemoryValidator, asset: str = "EURJPY"):
        self.validator = validator
        self.asset = asset
        self._max_lag = 200

    def run(self) -> MPRResult:
        data = self.validator.load_asset_data(self.asset)
        signals = self.validator.compute_signals(data)

        at = np.asarray(signals["adaptive_time"], dtype=np.float64)
        conflict = np.asarray(signals["memory_conflict"], dtype=np.float64)
        density = np.asarray(signals["memory_density"], dtype=np.float64)
        n = min(len(at), len(conflict), len(density))

        if n < self._max_lag * 2 + 1:
            return MPRResult("conflict_adaptive_time", "FAILED", metrics={"error": "insufficient data"})

        from research.memory_physics.memory_validator import _find_peak_lag
        lag_conflict, r_conflict = _find_peak_lag(conflict[:n], at[:n], self._max_lag)
        lag_density, r_density = _find_peak_lag(density[:n], at[:n], self._max_lag)

        flow_conflict_at = self.validator.information_flow("memory_conflict", "adaptive_time", signals)
        flow_density_at = self.validator.information_flow("memory_density", "adaptive_time", signals)

        conflict_leads_at = lag_conflict < 0
        density_leads_at = lag_density < 0

        if abs(r_density) > 0:
            relative_power = abs(r_conflict) / abs(r_density)
        else:
            relative_power = float('inf')

        if flow_density_at > 0:
            relative_info = flow_conflict_at / flow_density_at
        else:
            relative_info = float('inf')

        if abs(r_conflict) > abs(r_density):
            better_predictor = "memory_conflict"
        else:
            better_predictor = "memory_density"

        metrics = {
            "conflict_to_at_peak_lag": lag_conflict,
            "conflict_to_at_peak_corr": r_conflict,
            "density_to_at_peak_lag": lag_density,
            "density_to_at_peak_corr": r_density,
            "conflict_leads_at": conflict_leads_at,
            "density_leads_at": density_leads_at,
            "flow_conflict_to_at": flow_conflict_at,
            "flow_density_to_at": flow_density_at,
            "relative_corr_power": relative_power,
            "relative_info_flow_power": relative_info,
            "better_predictor": better_predictor,
        }

        if abs(r_conflict) > abs(r_density) * 1.1:
            status = "PASSED"
            print(f"  Conflict beats density for AT prediction (r_c={r_conflict:.4f} vs r_d={r_density:.4f})")
        elif abs(r_conflict) > abs(r_density) * 0.9:
            status = "INCONCLUSIVE"
            print(f"  Conflict and density similar for AT (r_c={r_conflict:.4f} vs r_d={r_density:.4f})")
        else:
            status = "FAILED"
            print(f"  Density beats conflict for AT (r_d={r_density:.4f} vs r_c={r_conflict:.4f})")

        return MPRResult("conflict_adaptive_time", status, metrics=metrics)
