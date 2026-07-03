from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from research.residual_energy.residual_validator import ResidualEnergyValidator, REPResult
from research.adaptive_alpha_engine.aae_validator import HORIZONS


class ResidualAlpha:
    def __init__(self, validator: ResidualEnergyValidator, asset: str = "EURJPY"):
        self.validator = validator
        self.asset = asset

    def run(self) -> REPResult:
        self.validator.load(self.asset)
        self.validator.build_residuals()

        res = self.validator.get_residual("xgboost")
        es = self.validator.es
        vol_metrics = self.validator.energy.vol_metrics
        fut_ret = self.validator.fut_ret
        target = fut_ret[:, 2]

        vol_names = list(vol_metrics.keys())
        X_vol = np.column_stack([vol_metrics[k] for k in vol_names])

        valid = ~np.isnan(target)
        valid = valid & ~np.isnan(es)
        for k in vol_names:
            valid = valid & ~np.isnan(vol_metrics[k])
        valid = valid & ~np.isnan(res)

        X_v = X_vol[valid]
        es_v = es[valid]
        res_v = res[valid]
        y_v = target[valid]

        X_b = np.column_stack([X_v, es_v])
        X_c = np.column_stack([X_v, res_v])

        model_types = ["linear", "random_forest"]
        models = {
            "linear": LinearRegression(),
            "random_forest": RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1),
        }

        model_results = {}
        for name in model_types:
            m = models[name]

            m.fit(X_v, y_v)
            p_a = m.predict(X_v)
            r2_a = r2_score(y_v, p_a)
            mi_a = self.validator.mutual_info(p_a, y_v)

            m.fit(X_b, y_v)
            p_b = m.predict(X_b)
            r2_b = r2_score(y_v, p_b)
            mi_b = self.validator.mutual_info(p_b, y_v)

            m.fit(X_c, y_v)
            p_c = m.predict(X_c)
            r2_c = r2_score(y_v, p_c)
            mi_c = self.validator.mutual_info(p_c, y_v)

            gain_b = r2_b - r2_a
            gain_c = r2_c - r2_a

            model_results[name] = {
                "r2_a": r2_a, "r2_b": r2_b, "r2_c": r2_c,
                "gain_b": gain_b, "gain_c": gain_c,
                "mi_a": mi_a, "mi_b": mi_b, "mi_c": mi_c,
            }

        residual_adds_more_info = model_results["random_forest"]["gain_c"] > model_results["random_forest"]["gain_b"]

        print(f"\n{'='*72}")
        print(f"  REP-4: Orthogonality Test — {self.asset}")
        print(f"{'='*72}")
        print(f"\n  Target: H20 ({HORIZONS[2]} bars ahead)")
        print(f"  Valid observations: {len(y_v)}")
        print()
        for name in model_types:
            r = model_results[name]
            print(f"  [{name.upper()}]")
            print(f"    Model A (vol only):          R\u00b2 = {r['r2_a']:.6f}  MI = {r['mi_a']:.6f}")
            print(f"    Model B (vol + ES):          R\u00b2 = {r['r2_b']:.6f}  MI = {r['mi_b']:.6f}")
            print(f"    Model C (vol + Residual):    R\u00b2 = {r['r2_c']:.6f}  MI = {r['mi_c']:.6f}")
            print(f"    Gain from ES:      DeltaR2 = {r['gain_b']:+.6f}")
            print(f"    Gain from Residual: DeltaR2 = {r['gain_c']:+.6f}")
            print()

        print(f"  Cross-prediction: Residual {'ADDS MORE' if residual_adds_more_info else 'ADDS LESS OR EQUAL'} unique information vs ES")
        print()

        summary = (
            f"Residual adds {'more' if residual_adds_more_info else 'less or equal'} unique info than ES "
            f"(RF gain_c={model_results['random_forest']['gain_c']:.6f} vs gain_b={model_results['random_forest']['gain_b']:.6f})"
        )

        metrics = {
            "model_results": model_results,
            "residual_adds_more_info": residual_adds_more_info,
            "summary": summary,
            "n_valid": int(np.sum(valid)),
        }

        return REPResult(rq_name="REP-4", status="COMPLETE", metrics=metrics)
