from __future__ import annotations

import numpy as np
from research.residual_energy.residual_validator import ResidualEnergyValidator, REPResult, RESIDUAL_TYPES


class ResidualConstructor:
    def __init__(self, validator: ResidualEnergyValidator, asset: str = "EURJPY"):
        self.validator = validator
        self.asset = asset

    def run(self) -> REPResult:
        self.validator.load(self.asset)
        self.validator.build_residuals()

        es_a = self.validator.es_alpha(2)

        residual_results = {}
        best_residual_type = None
        best_sharpe = -np.inf

        print(f"\n{'='*72}")
        print(f"  REP-1: Residual Construction — {self.asset}")
        print(f"{'='*72}")
        print(f"\n  ES (H20): mean={es_a['mean']:.6f}  pp={es_a['pp']:.4f}  sharpe={es_a['sharpe']:.4f}  n={es_a['n']}")
        print(f"\n  {'Residual Type':<18} {'R²':>8} {'Res PP':>8} {'Res Sharpe':>12} {'ES PP':>8}  {'Beats ES':>9}")
        print(f"  {'-'*18} {'-'*8} {'-'*8} {'-'*12} {'-'*8}  {'-'*9}")

        for rt in RESIDUAL_TYPES:
            r2 = self.validator.models[rt]["r2"]
            n_valid = self.validator.models[rt]["n_valid"]
            res_a = self.validator.residual_alpha(rt, 2)
            mh = self.validator.multi_horizon_alpha(self.validator.get_residual(rt))

            beats_es = res_a["pp"] >= es_a["pp"]

            residual_results[rt] = {
                "r2": r2,
                "n_valid": n_valid,
                "alpha_h20": res_a,
                "multi_horizon": mh,
                "beats_es": beats_es,
            }

            print(f"  {rt:<18} {r2:>8.4f} {res_a['pp']:>8.4f} {res_a['sharpe']:>12.4f} {es_a['pp']:>8.4f}  {str(beats_es):>9}")

            if res_a["sharpe"] > best_sharpe:
                best_sharpe = res_a["sharpe"]
                best_residual_type = rt

        best_beats = residual_results[best_residual_type]["beats_es"]
        best_mh = residual_results[best_residual_type]["multi_horizon"]

        print(f"\n  Best residual type: {best_residual_type} (sharpe={best_sharpe:.4f})")
        print(f"  Best residual beats ES: {best_beats}")

        print(f"\n  Multi-horizon alpha — {best_residual_type}")
        print(f"  {'Horizon':<10} {'Mean':>10} {'PP':>8} {'Sharpe':>10} {'Std':>10} {'N':>6}")
        print(f"  {'-'*10} {'-'*10} {'-'*8} {'-'*10} {'-'*10} {'-'*6}")
        for h_str, alpha_dict in best_mh.items():
            print(f"  {h_str:<10} {alpha_dict['mean']:>10.6f} {alpha_dict['pp']:>8.4f} {alpha_dict['sharpe']:>10.4f} {alpha_dict['std']:>10.6f} {alpha_dict['n']:>6}")
        print()

        metrics = {
            "es_alpha": es_a,
            "residual_results": residual_results,
            "best_residual_type": best_residual_type,
            "best_residual_beats_es": best_beats,
        }

        return REPResult(rq_name="REP-1", status="COMPLETE", metrics=metrics)
