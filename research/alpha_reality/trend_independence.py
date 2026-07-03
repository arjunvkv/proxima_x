"""RQ1: Does alpha survive after removing trend-related information?"""

from __future__ import annotations

import numpy as np

from research.alpha_reality.arl_validator import ARLValidator, ARLResult, HORIZONS, _future_returns


class TrendIndependence:
    def __init__(self, validator: ARLValidator, asset: str = "EURJPY"):
        self.validator = validator
        self.asset = asset

    def run(self) -> ARLResult:
        data = self.validator.load_asset_data(self.asset)
        price = data["price"]

        orig_signals = self.validator.compute_signals(data)
        det_signals = self.validator.compute_detrended_signals(data, trend_period=200)

        fr_all = _future_returns(price, np.array(HORIZONS, dtype=np.int32))

        orig_alpha = self.validator.alpha_signal(orig_signals)
        det_alpha = self.validator.alpha_signal(det_signals)

        results = {}
        for hi, h in enumerate(HORIZONS):
            orig_eval = self.validator.eval_alpha(orig_alpha, fr_all, hi)
            det_eval = self.validator.eval_alpha(det_alpha, fr_all, hi)
            mean_decay = (orig_eval["mean"] - det_eval["mean"]) / max(abs(orig_eval["mean"]), 1e-12)
            pp_decay = orig_eval["pp"] - det_eval["pp"]
            results[f"H{h}"] = {
                "original": orig_eval,
                "detrended": det_eval,
                "mean_decay": mean_decay,
                "pp_decay": pp_decay,
                "alpha_survives": det_eval["pp"] > 0.55,
            }

        h20 = results.get("H20", {})
        surv = h20.get("alpha_survives", False)
        print(f"  Trend Independence @ H20:")
        print(f"    Original: mean={h20.get('original', {}).get('mean', 0):.6f}, pp={h20.get('original', {}).get('pp', 0):.3f}")
        print(f"    Detrend:  mean={h20.get('detrended', {}).get('mean', 0):.6f}, pp={h20.get('detrended', {}).get('pp', 0):.3f}")
        print(f"    Mean decay: {h20.get('mean_decay', 0):.2f}")
        print(f"    Alpha survives detrending: {'YES' if surv else 'NO'}")

        status = "PASSED" if surv else "FAILED"
        return ARLResult("trend_independence", status, metrics=results)
