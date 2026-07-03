from __future__ import annotations

from typing import Any

import numpy as np

from research.memory_physics.memory_validator import (
    MemoryValidator, MPRResult, TARGET_ASSETS,
)


class AssetInvariance:
    """RQ7: Does memory_conflict's relationship survive across all 5 assets?"""

    def __init__(self, validator: MemoryValidator):
        self.validator = validator
        self._max_lag = 200

    def run(self) -> MPRResult:
        from research.memory_physics.memory_validator import _find_peak_lag

        asset_results = {}

        for asset in TARGET_ASSETS:
            try:
                data = self.validator.load_asset_data(asset)
                signals = self.validator.compute_signals(data)

                conflict = np.asarray(signals["memory_conflict"], dtype=np.float64)
                density = np.asarray(signals["memory_density"], dtype=np.float64)
                at = np.asarray(signals["adaptive_time"], dtype=np.float64)
                smr = np.asarray(signals["state_mutation_rate"], dtype=np.float64)
                n = min(len(conflict), len(density), len(at))

                lag_cd, r_cd = _find_peak_lag(conflict[:n], density[:n], self._max_lag)
                lag_ca, r_ca = _find_peak_lag(conflict[:n], at[:n], self._max_lag)
                lag_cs, r_cs = _find_peak_lag(conflict[:n], smr[:n], self._max_lag)

                conflict_leads_density = lag_cd < 0

                asset_results[asset] = {
                    "conflict_to_density_lag": lag_cd,
                    "conflict_to_density_corr": r_cd,
                    "conflict_to_at_lag": lag_ca,
                    "conflict_to_at_corr": r_ca,
                    "conflict_to_mutation_lag": lag_cs,
                    "conflict_to_mutation_corr": r_cs,
                    "conflict_leads_density": conflict_leads_density,
                    "n": n,
                }
                print(f"  [{asset}] c->d lag={lag_cd} r={r_cd:.4f} | c->at lag={lag_ca} r={r_ca:.4f} | c->smr lag={lag_cs} r={r_cs:.4f}")
            except Exception as e:
                print(f"  [{asset}] FAILED: {e}")
                asset_results[asset] = {"error": str(e)}

        # Consistency metrics
        lead_signs = [r.get("conflict_leads_density") for r in asset_results.values()
                      if isinstance(r, dict) and "conflict_leads_density" in r]
        lead_consistency = sum(lead_signs) / max(len(lead_signs), 1) if lead_signs else 0.0

        lags_cd = [r["conflict_to_density_lag"] for r in asset_results.values()
                   if isinstance(r, dict) and "conflict_to_density_lag" in r]
        lag_std_cd = float(np.std(lags_cd)) if lags_cd else 0.0

        corrs_cd = [abs(r["conflict_to_density_corr"]) for r in asset_results.values()
                    if isinstance(r, dict) and "conflict_to_density_corr" in r]
        avg_corr_cd = float(np.mean(corrs_cd)) if corrs_cd else 0.0

        corrs_ca = [abs(r["conflict_to_at_corr"]) for r in asset_results.values()
                    if isinstance(r, dict) and "conflict_to_at_corr" in r]
        avg_corr_ca = float(np.mean(corrs_ca)) if corrs_ca else 0.0

        metrics = {
            "per_asset": asset_results,
            "lead_consistency": lead_consistency,
            "lag_std_conflict_to_density": lag_std_cd,
            "avg_corr_conflict_to_density": avg_corr_cd,
            "avg_corr_conflict_to_at": avg_corr_ca,
            "n_assets_with_conflict_leads": int(sum(lead_signs)),
            "n_assets_total": len(TARGET_ASSETS),
        }

        if lead_consistency > 0.8 and avg_corr_cd > 0.1:
            status = "PASSED"
            print(f"  Conflict->density relationship survives across assets (consistency={lead_consistency:.0%})")
        elif lead_consistency > 0.5:
            status = "INCONCLUSIVE"
            print(f"  Partial asset invariance (consistency={lead_consistency:.0%})")
        else:
            status = "FAILED"
            print(f"  Conflict->density NOT invariant across assets (consistency={lead_consistency:.0%})")

        return MPRResult("asset_invariance", status, metrics=metrics)
