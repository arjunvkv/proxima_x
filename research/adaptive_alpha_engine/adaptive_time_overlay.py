from __future__ import annotations
import numpy as np
from research.adaptive_alpha_engine.aae_validator import AAEValidator, AAEResult, HORIZONS, _future_returns


def _max_drawdown(returns: np.ndarray) -> float:
    cum = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / np.maximum(peak, 1e-12)
    return float(np.min(dd))


SIZING = {0: 0.10, 1: 0.25, 2: 0.50, 3: 0.75, 4: 1.0}


class AdaptiveTimeOverlay:
    def __init__(self, validator: AAEValidator, asset: str = "EURJPY"):
        self.validator = validator
        self.asset = asset

    def run(self) -> AAEResult:
        data = self.validator.load_asset_data(self.asset)
        price = data["price"]
        signals = self.validator.compute_signals(data)
        es = np.asarray(signals["energy_storage"], dtype=np.float64)
        at = np.asarray(signals["adaptive_time"], dtype=np.float64)
        fr_all = _future_returns(price, np.array(HORIZONS, dtype=np.int32))
        n = min(len(es), len(at), fr_all.shape[0])
        es, at = es[:n], at[:n]
        fr_all = fr_all[:n]

        es_threshold = np.nanpercentile(es, 90)
        es_top_mask = es >= es_threshold

        at_bounds = [
            np.nanpercentile(at, 20),
            np.nanpercentile(at, 40),
            np.nanpercentile(at, 60),
            np.nanpercentile(at, 80),
        ]

        at_quintile = np.zeros(n, dtype=np.int32)
        at_quintile[at <= at_bounds[0]] = 0
        at_quintile[(at > at_bounds[0]) & (at <= at_bounds[1])] = 1
        at_quintile[(at > at_bounds[1]) & (at <= at_bounds[2])] = 2
        at_quintile[(at > at_bounds[2]) & (at <= at_bounds[3])] = 3
        at_quintile[at > at_bounds[3]] = 4

        size_weights = np.array([SIZING[q] for q in at_quintile], dtype=np.float64)

        es_only_results = {}
        es_at_overlay_results = {}

        for hi, h in enumerate(HORIZONS):
            fwd = fr_all[:, hi]
            mask = es_top_mask & ~np.isnan(fwd)

            if np.sum(mask) < 5:
                es_only_results[f"H{h}"] = {
                    "mean": 0.0, "pp": 0.5, "sharpe": 0.0, "max_dd": 0.0, "n": 0,
                }
                es_at_overlay_results[f"H{h}"] = {
                    "mean": 0.0, "pp": 0.5, "sharpe": 0.0, "max_dd": 0.0, "n": 0,
                }
                continue

            es_returns = fwd[mask]
            es_mean = float(np.nanmean(es_returns))
            es_std = float(np.nanstd(es_returns))
            es_sharpe = es_mean / max(es_std, 1e-12)
            es_pp = float(np.mean(es_returns > 0))
            es_dd = _max_drawdown(es_returns)

            es_only_results[f"H{h}"] = {
                "mean": es_mean,
                "pp": es_pp,
                "sharpe": es_sharpe,
                "max_dd": es_dd,
                "n": int(np.sum(mask)),
            }

            w = size_weights[mask]
            overlay_returns = es_returns * w
            ov_mean = float(np.nanmean(overlay_returns))
            ov_std = float(np.nanstd(overlay_returns))
            ov_sharpe = ov_mean / max(ov_std, 1e-12)
            ov_pp = float(np.mean(overlay_returns > 0))
            ov_dd = _max_drawdown(overlay_returns)

            es_at_overlay_results[f"H{h}"] = {
                "mean": ov_mean,
                "pp": ov_pp,
                "sharpe": ov_sharpe,
                "max_dd": ov_dd,
                "n": int(np.sum(mask)),
            }

        h20_es = es_only_results.get("H20", {})
        h20_ov = es_at_overlay_results.get("H20", {})

        print(f"  Adaptive Time Overlay @ H20 ({self.asset}):")
        print(f"    {'Metric':<12s} {'ES Only':>10s} {'ES+AT':>10s} {'Delta':>10s}")
        print(f"    {'-'*42}")
        for k in ["mean", "pp", "sharpe", "max_dd"]:
            ev = h20_es.get(k, 0)
            ov = h20_ov.get(k, 0)
            print(f"    {k:<12s} {ev:>10.4f} {ov:>10.4f} {ov - ev:>+10.4f}")

        better_sharpe = h20_ov.get("sharpe", 0) > h20_es.get("sharpe", 0)
        lower_dd = h20_ov.get("max_dd", 0) > h20_es.get("max_dd", 0)
        es_sh = h20_es.get("sharpe", 1e-12)
        improvement_ratio = (h20_ov.get("sharpe", 0) - es_sh) / abs(es_sh) if abs(es_sh) > 1e-12 else 0

        print(f"    Higher Sharpe:     {'YES' if better_sharpe else 'NO'}")
        print(f"    Lower Drawdown:    {'YES' if lower_dd else 'NO'}")
        print(f"    Improvement Ratio: {improvement_ratio:+.4f}")

        return AAEResult("adaptive_time_overlay", "COMPLETE", metrics={
            "es_only": es_only_results,
            "es_at_overlay": es_at_overlay_results,
            "comparison": {
                "better_sharpe": better_sharpe,
                "lower_drawdown": lower_dd,
                "improvement_ratio": improvement_ratio,
            },
        })
