from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

from research.energy_reality.energy_validator import EnergyValidator, ERLResult, VOL_METRICS


class ResidualAlpha:
    def __init__(self, validator: EnergyValidator, asset: str = "EURJPY"):
        self.validator = validator
        self.asset = asset

    def run(self) -> ERLResult:
        self.validator.load(self.asset)

        es = self.validator.es_signal()

        X = np.column_stack([self.validator.vol_metrics[m] for m in VOL_METRICS])
        y = es

        valid = np.all(~np.isnan(X), axis=1) & ~np.isnan(y)
        X_valid = X[valid]
        y_valid = y[valid]

        models = {
            "linear": LinearRegression(),
            "random_forest": RandomForestRegressor(
                n_estimators=100, max_depth=5, random_state=42, n_jobs=-1
            ),
            "xgboost": XGBRegressor(
                n_estimators=100, max_depth=5, random_state=42, n_jobs=-1, verbosity=0
            ),
        }

        model_results = {}
        for name, model in models.items():
            model.fit(X_valid, y_valid)
            y_pred = model.predict(X_valid)
            r2 = r2_score(y_valid, y_pred)
            residual = y_valid - y_pred
            full_residual = np.full(len(es), np.nan, dtype=np.float64)
            full_residual[valid] = residual
            residual_alpha = self.validator.eval_alpha(full_residual, 2)
            model_results[name] = {
                "r2_score": r2,
                "residual_alpha": residual_alpha,
                "retention": 0.0,
            }

        es_alpha = self.validator.eval_alpha(es, 2)

        for name in model_results:
            res_pp = model_results[name]["residual_alpha"]["pp"]
            model_results[name]["retention"] = res_pp / es_alpha["pp"] if es_alpha["pp"] > 0 else 0.0

        passes = any(
            model_results[m]["retention"] > 0.50 for m in model_results
        )
        verdict = "PASS" if passes else "FAIL"

        best_model = max(model_results, key=lambda m: model_results[m]["retention"])

        header = f"{'Model':<15} {'R²':>8} {'Resid PP':>10} {'Resid Sharpe':>14} {'Retention':>10}"
        sep = "-" * len(header)
        lines = [header, sep]
        for name in ["linear", "random_forest", "xgboost"]:
            r = model_results[name]
            ra = r["residual_alpha"]
            lines.append(
                f"{name:<15} {r['r2_score']:>8.4f} {ra['pp']:>10.4f} {ra['sharpe']:>14.4f} {r['retention']:>9.2%}"
            )
        lines.append("")
        lines.append(f"Benchmark ES PP: {es_alpha['pp']:.4f} | Sharpe: {es_alpha['sharpe']:.4f}")
        lines.append(f"Best model: {best_model} (retention {model_results[best_model]['retention']:.2%})")
        lines.append(f"Verdict: {verdict}")
        print("\n".join(lines))

        metrics = {
            "benchmark_es_alpha": es_alpha,
            "model_results": model_results,
            "passes": passes,
            "verdict": verdict,
            "best_model": best_model,
        }

        return ERLResult(rq_name="ERL-2 Residual Alpha Test", status=verdict, metrics=metrics)
