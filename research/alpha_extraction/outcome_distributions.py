"""RQ1: Do future outcome distributions materially differ across variable deciles?"""

from __future__ import annotations

from typing import Any

import numpy as np

from research.alpha_extraction.alpha_validator import (
    AlphaValidator, AELResult, PRIMARY_VARIABLES, HORIZONS,
    _future_returns, _future_drawdown, _future_runup,
)


class OutcomeDistributions:
    def __init__(self, validator: AlphaValidator, asset: str = "EURJPY"):
        self.validator = validator
        self.asset = asset

    def run(self) -> AELResult:
        data = self.validator.load_asset_data(self.asset)
        signals = self.validator.compute_signals(data)
        price = data["price"]

        fr_all = _future_returns(price, np.array(HORIZONS, dtype=np.int32))
        horizons_arr = np.array(HORIZONS, dtype=np.int32)

        all_results = {}
        for var in PRIMARY_VARIABLES:
            sig = np.asarray(signals[var], dtype=np.float64)
            n = min(len(sig), fr_all.shape[0])
            sig, fr = sig[:n], fr_all[:n]

            boundaries, masks = self.validator.decile_bins(sig)
            var_results = {}
            for di in range(10):
                mask = masks[di]
                if np.sum(mask) < 5:
                    continue

                decile_stats = {}
                for hi, h in enumerate(HORIZONS):
                    fwd = fr[:, hi]
                    stats = self.validator.bucket_statistics(fwd, mask)
                    dd = _future_drawdown(price[:len(fwd)], h)
                    ru = _future_runup(price[:len(fwd)], h)
                    stats["drawdown"] = float(np.nanmean(dd[mask])) if np.sum(mask) > 0 else 0.0
                    stats["runup"] = float(np.nanmean(ru[mask])) if np.sum(mask) > 0 else 0.0
                    decile_stats[f"H{h}"] = stats

                var_results[f"decile_{di + 1}"] = decile_stats

            var_results["decile_boundaries"] = boundaries.tolist()
            all_results[var] = var_results

        sep_metrics = self._separation_metrics(all_results, signals)
        print("  Outcome Distribution Separation:")
        for var in PRIMARY_VARIABLES:
            vr = all_results[var]
            d1 = vr.get("decile_1", {}).get("H20", {})
            d10 = vr.get("decile_10", {}).get("H20", {})
            spread = d10.get("mean", 0) - d1.get("mean", 0)
            print(f"    {var:25s}: D10-D1 mean@H20={spread:.6f}, "
                  f"separation={sep_metrics[var]['max_separation_horizon']}")

        return AELResult("outcome_distributions", "COMPLETE",
                         metrics={"variables": all_results, "separation": sep_metrics})

    @staticmethod
    def _separation_metrics(results: dict, signals: dict) -> dict:
        sep = {}
        for var in PRIMARY_VARIABLES:
            vr = results.get(var, {})
            max_f_stat = 0.0
            best_h = None
            for hi_str in ["H1", "H5", "H20", "H50", "H100", "H500"]:
                means = []
                for di in range(10):
                    d = vr.get(f"decile_{di + 1}", {}).get(hi_str, {})
                    means.append(d.get("mean", 0))
                if means:
                    arr = np.array(means)
                    f_stat = float(np.std(arr) / (np.abs(np.mean(arr)) + 1e-12))
                    if f_stat > max_f_stat:
                        max_f_stat = f_stat
                        best_h = hi_str
            sep[var] = {"max_separation_horizon": best_h, "f_stat": max_f_stat}
        return sep
