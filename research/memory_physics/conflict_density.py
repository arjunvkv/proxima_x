from __future__ import annotations

from typing import Any

import numpy as np

from research.memory_physics.memory_validator import MemoryValidator, MPRResult


class ConflictDensityAnalysis:
    """RQ1: Conflict -> Density. Does memory_conflict consistently precede memory_density?"""

    def __init__(self, validator: MemoryValidator, asset: str = "EURJPY"):
        self.validator = validator
        self.asset = asset
        self._max_lag = 200

    def run(self) -> MPRResult:
        data = self.validator.load_asset_data(self.asset)
        signals = self.validator.compute_signals(data)

        conflict = np.asarray(signals["memory_conflict"], dtype=np.float64)
        density = np.asarray(signals["memory_density"], dtype=np.float64)
        n = min(len(conflict), len(density))

        if n < self._max_lag * 2 + 1:
            return MPRResult("conflict_density", "FAILED", metrics={"error": "insufficient data"})

        from research.memory_physics.memory_validator import _find_peak_lag
        lag, corr = _find_peak_lag(conflict[:n], density[:n], self._max_lag)

        info_flow_conflict_to_density = self.validator.lagged_information_flow(
            "memory_conflict", "memory_density", signals, min(lag, 0) if lag < 0 else 0)
        info_flow_density_to_conflict = self.validator.lagged_information_flow(
            "memory_density", "memory_conflict", signals, max(-lag, 0) if lag > 0 else 0)

        conflict_leads = lag < 0
        net_flow = info_flow_conflict_to_density - info_flow_density_to_conflict

        metrics = {
            "peak_lag": lag,
            "peak_corr": corr,
            "conflict_leads_density": conflict_leads,
            "info_flow_conflict_to_density": info_flow_conflict_to_density,
            "info_flow_density_to_conflict": info_flow_density_to_conflict,
            "net_information_flow": net_flow,
            "n_samples": n,
        }

        if conflict_leads and abs(corr) > 0.1:
            status = "PASSED"
            print(f"  Conflict consistently precedes density (lag={lag}, corr={corr:.4f})")
        elif abs(corr) > 0.1:
            status = "INCONCLUSIVE"
            print(f"  Conflict/density relationship exists but direction unclear (lag={lag}, corr={corr:.4f})")
        else:
            status = "FAILED"
            print(f"  No clear conflict->density relationship (lag={lag}, corr={corr:.4f})")

        return MPRResult("conflict_density", status, metrics=metrics)
