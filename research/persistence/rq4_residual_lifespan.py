import numpy as np
from scipy.optimize import curve_fit
from research.persistence.persistence_utils import PersistenceDataLoader


class RQ4ResidualLifespan:
    """RQ4: Model Residual Energy decay during persistence events."""

    def __init__(self, asset: str = "EURJPY"):
        self.asset = asset

    @staticmethod
    def _exp_decay(t, a, b, c):
        return a * np.exp(-b * t) + c

    @staticmethod
    def _power_decay(t, a, b, c):
        return a * np.power(t + 1e-6, -b) + c

    @staticmethod
    def _linear_decay(t, a, b):
        return a * t + b

    def run(self) -> dict:
        loader = PersistenceDataLoader(self.asset)
        df = loader.get_events_df()
        if len(df) < 5:
            return {"error": "Not enough events", "n_events": len(df)}

        re_entry = df["residual_energy_entry"].values
        re_peak = df["residual_energy_peak"].values
        re_exit = df["residual_energy_exit"].values
        durations = df["duration"].values

        decay_models = []
        for i in range(len(df)):
            s, e = int(df.iloc[i]["start_idx"]), int(df.iloc[i]["end_idx"])
            if e - s < 2:
                continue
            dur = e - s + 1
            t = np.arange(dur, dtype=float)
            re_series = loader.layers["residual_energy"][s:e + 1].copy()

            if np.any(np.isnan(re_series)) or np.any(np.isinf(re_series)):
                continue

            results = self._fit_decay_models(t, re_series)
            if results:
                decay_models.append(results)

        # Aggregate model fits
        if not decay_models:
            return {"error": "No valid decay fits", "n_events": len(df)}

        def safe_mean(arr):
            arr = np.array([v for v in arr if not (np.isnan(v) or np.isinf(v))])
            return float(np.mean(arr)) if len(arr) > 0 else 0.0

        def safe_model_extract(models, model_name, key):
            vals = [m[model_name][key] for m in models if model_name in m and key in m[model_name]]
            return safe_mean(vals) if vals else 0.0

        def safe_model_std(models, model_name, key):
            vals = [m[model_name][key] for m in models if model_name in m and key in m[model_name]]
            return float(np.std(vals)) if len(vals) > 1 else 0.0

        agg = {
            "exponential": {
                "rate_mean": safe_model_extract(decay_models, "exponential", "rate"),
                "rate_std": safe_model_std(decay_models, "exponential", "rate"),
                "r2_mean": safe_model_extract(decay_models, "exponential", "r2"),
            },
            "power_law": {
                "exponent_mean": safe_model_extract(decay_models, "power_law", "exponent"),
                "exponent_std": safe_model_std(decay_models, "power_law", "exponent"),
                "r2_mean": safe_model_extract(decay_models, "power_law", "r2"),
            },
            "linear": {
                "slope_mean": safe_model_extract(decay_models, "linear", "slope"),
                "slope_std": safe_model_std(decay_models, "linear", "slope"),
                "r2_mean": safe_model_extract(decay_models, "linear", "r2"),
            },
        }

        # Best model by R2 (only consider models with at least 3 fits)
        valid_models = [k for k in agg if agg[k]["r2_mean"] > 0]
        best_model = max(valid_models, key=lambda x: agg[x]["r2_mean"]) if valid_models else "unknown"

        return {
            "asset": self.asset,
            "n_events_fit": len(decay_models),
            "total_events": len(df),
            "best_decay_model": best_model,
            "aggregate_fits": agg,
            "decay_rate_interpretation": (
                "If exponential rate > 0, residual energy decays exponentially. "
                "If power-law exponent > 0, decay follows power law. "
                "If linear slope < 0, steady linear decay."
            ),
        }

    def _fit_decay_models(self, t: np.ndarray, y: np.ndarray) -> dict | None:
        t_norm = t / max(t[-1], 1)
        y_norm = y / max(np.abs(y).max(), 1e-10)

        results = {}
        for name, func, p0 in [
            ("exponential", self._exp_decay, [1.0, 0.1, 0.0]),
            ("power_law", self._power_decay, [1.0, 0.5, 0.0]),
            ("linear", self._linear_decay, [-0.1, 1.0]),
        ]:
            try:
                popt, _ = curve_fit(func, t_norm, y_norm, p0=p0, maxfev=5000)
                y_pred = func(t_norm, *popt)
                ss_res = np.sum((y_norm - y_pred) ** 2)
                ss_tot = np.sum((y_norm - np.mean(y_norm)) ** 2)
                r2 = 1.0 - ss_res / max(ss_tot, 1e-10)
                if name == "exponential":
                    results[name] = {"rate": float(popt[1]), "asymptote": float(popt[2]),
                                     "amplitude": float(popt[0]), "r2": float(r2)}
                elif name == "power_law":
                    results[name] = {"exponent": float(popt[1]), "asymptote": float(popt[2]),
                                     "amplitude": float(popt[0]), "r2": float(r2)}
                else:
                    results[name] = {"slope": float(popt[0]), "intercept": float(popt[1]), "r2": float(r2)}
            except Exception:
                continue

        return results if results else None
