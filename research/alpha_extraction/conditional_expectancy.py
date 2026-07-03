"""RQ4: Does conditioning on variable states improve expectancy over unconditional?"""

from __future__ import annotations

from typing import Any

import numpy as np

from research.alpha_extraction.alpha_validator import (
    AlphaValidator, AELResult, PRIMARY_VARIABLES, HORIZONS,
    _future_returns, _future_drawdown, _future_runup,
)


class ConditionalExpectancy:
    def __init__(self, validator: AlphaValidator, asset: str = "EURJPY"):
        self.validator = validator
        self.asset = asset

    def run(self) -> AELResult:
        data = self.validator.load_asset_data(self.asset)
        signals = self.validator.compute_signals(data)
        price = data["price"]

        horizons_arr = np.array(HORIZONS, dtype=np.int32)
        fr_all = _future_returns(price, horizons_arr)

        # Unconditional expectancy
        unconditional = {}
        for hi, h in enumerate(HORIZONS):
            fwd = fr_all[:, hi]
            dd = _future_drawdown(price, h)
            ru = _future_runup(price, h)
            valid = ~np.isnan(fwd)
            unconditional[f"H{h}"] = {
                "mean": float(np.nanmean(fwd)),
                "median": float(np.nanmedian(fwd)),
                "std": float(np.nanstd(fwd)),
                "drawdown": float(np.nanmean(dd[valid])),
                "runup": float(np.nanmean(ru[valid])),
                "profit_probability": float(np.mean(fwd[valid] > 0)),
            }

        # Conditional: high states (top decile)
        conditions = {}
        for var in PRIMARY_VARIABLES:
            sig = np.asarray(signals[var], dtype=np.float64)
            n = min(len(sig), fr_all.shape[0])
            sig, fr = sig[:n], fr_all[:n]

            _, masks = self.validator.decile_bins(sig)
            high_mask = masks[-1]
            low_mask = masks[0]

            cond_high = {}
            cond_low = {}
            for hi, h in enumerate(HORIZONS):
                fwd = fr[:, hi]
                dd = _future_drawdown(price[:len(fwd)], h)
                ru = _future_runup(price[:len(fwd)], h)

                hm = high_mask[:len(fwd)]
                lm = low_mask[:len(fwd)]
                if np.sum(hm) > 5:
                    cond_high[f"H{h}"] = {
                        "mean": float(np.nanmean(fwd[hm])),
                        "std": float(np.nanstd(fwd[hm])),
                        "drawdown": float(np.nanmean(dd[hm])),
                        "runup": float(np.nanmean(ru[hm])),
                        "profit_probability": float(np.mean(fwd[hm] > 0)),
                        "n": int(np.sum(hm)),
                    }
                if np.sum(lm) > 5:
                    cond_low[f"H{h}"] = {
                        "mean": float(np.nanmean(fwd[lm])),
                        "std": float(np.nanstd(fwd[lm])),
                        "drawdown": float(np.nanmean(dd[lm])),
                        "runup": float(np.nanmean(ru[lm])),
                        "profit_probability": float(np.mean(fwd[lm] > 0)),
                        "n": int(np.sum(lm)),
                    }

            improvement = {}
            for h_str in cond_high:
                uncond = unconditional.get(h_str, {})
                high = cond_high.get(h_str, {})
                improvement[h_str] = {
                    "unconditional_mean": uncond.get("mean", 0),
                    "conditional_high_mean": high.get("mean", 0),
                    "delta": high.get("mean", 0) - uncond.get("mean", 0),
                    "conditional_high_pp": high.get("profit_probability", 0),
                    "unconditional_pp": uncond.get("profit_probability", 0),
                }

            conditions[var] = {
                "high": cond_high,
                "low": cond_low,
                "improvement": improvement,
            }

            h20 = improvement.get("H20", {})
            print(f"    {var:25s}: unconditional_mean={h20.get('unconditional_mean', 0):.6f}, "
                  f"high_mean={h20.get('conditional_high_mean', 0):.6f}, "
                  f"delta={h20.get('delta', 0):.6f}")

        return AELResult("conditional_expectancy", "COMPLETE",
                         metrics={"unconditional": unconditional, "conditional": conditions})
