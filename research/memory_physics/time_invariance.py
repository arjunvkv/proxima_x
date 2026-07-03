from __future__ import annotations

from typing import Any

import numpy as np

from research.memory_physics.memory_validator import (
    MemoryValidator, MPRResult, TIME_WINDOWS,
)


class TimeInvariance:
    """RQ8: Does memory_conflict remain upstream across time windows?"""

    def __init__(self, validator: MemoryValidator, asset: str = "EURJPY"):
        self.validator = validator
        self.asset = asset
        self._max_lag = 200

    def run(self) -> MPRResult:
        from research.memory_physics.memory_validator import _find_peak_lag

        window_results = {}

        for start, end, label in TIME_WINDOWS:
            try:
                data = self.validator.load_data_window(self.asset, start, end)
                signals = self.validator.compute_signals(data)

                conflict = np.asarray(signals["memory_conflict"], dtype=np.float64)
                density = np.asarray(signals["memory_density"], dtype=np.float64)
                at = np.asarray(signals["adaptive_time"], dtype=np.float64)
                n = min(len(conflict), len(density), len(at))

                if n < self._max_lag * 2 + 1:
                    print(f"  [{label}] insufficient data ({n} pts)")
                    continue

                lag_cd, r_cd = _find_peak_lag(conflict[:n], density[:n], self._max_lag)
                lag_ca, r_ca = _find_peak_lag(conflict[:n], at[:n], self._max_lag)

                conflict_leads_density = lag_cd < 0

                window_results[label] = {
                    "n": n,
                    "conflict_to_density_lag": lag_cd,
                    "conflict_to_density_corr": r_cd,
                    "conflict_to_at_lag": lag_ca,
                    "conflict_to_at_corr": r_ca,
                    "conflict_leads_density": conflict_leads_density,
                }
                print(f"  [{label}] {n} pts: c->d lag={lag_cd} r={r_cd:.4f} | c->at lag={lag_ca} r={r_ca:.4f}")
            except Exception as e:
                print(f"  [{label}] FAILED: {e}")
                window_results[label] = {"error": str(e)}

        lead_signs = [r.get("conflict_leads_density") for r in window_results.values()
                      if isinstance(r, dict) and "conflict_leads_density" in r]
        lead_consistency = sum(lead_signs) / max(len(lead_signs), 1) if lead_signs else 0.0

        lags = [r["conflict_to_density_lag"] for r in window_results.values()
                if isinstance(r, dict) and "conflict_to_density_lag" in r]
        lag_std = float(np.std(lags)) if lags else 0.0

        corrs = [abs(r["conflict_to_density_corr"]) for r in window_results.values()
                 if isinstance(r, dict) and "conflict_to_density_corr" in r]
        avg_corr = float(np.mean(corrs)) if corrs else 0.0

        metrics = {
            "per_window": window_results,
            "lead_consistency": lead_consistency,
            "lag_std": lag_std,
            "avg_corr": avg_corr,
            "n_windows_with_conflict_leads": int(sum(lead_signs)),
        }

        if lead_consistency > 0.75 and avg_corr > 0.1:
            status = "PASSED"
            print(f"  Conflict->density persists across time (consistency={lead_consistency:.0%})")
        elif lead_consistency > 0.5:
            status = "INCONCLUSIVE"
        else:
            status = "FAILED"

        return MPRResult("time_invariance", status, metrics=metrics)
