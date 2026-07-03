"""RQ2: Map the expectancy surface — state-to-outcome across variables and horizons."""

from __future__ import annotations

from typing import Any

import numpy as np

from research.alpha_extraction.alpha_validator import (
    AlphaValidator, AELResult, PRIMARY_VARIABLES, HORIZONS,
    _future_returns, _future_drawdown, _future_runup,
)


class ExpectancySurface:
    def __init__(self, validator: AlphaValidator, asset: str = "EURJPY"):
        self.validator = validator
        self.asset = asset

    def run(self) -> AELResult:
        data = self.validator.load_asset_data(self.asset)
        signals = self.validator.compute_signals(data)
        price = data["price"]

        horizons_arr = np.array(HORIZONS, dtype=np.int32)
        fr_all = _future_returns(price, horizons_arr)

        surface = {}
        for var in PRIMARY_VARIABLES:
            sig = np.asarray(signals[var], dtype=np.float64)
            n = min(len(sig), fr_all.shape[0])
            sig, fr = sig[:n], fr_all[:n]

            boundaries, masks = self.validator.decile_bins(sig)
            var_surface = {}
            for di in range(10):
                mask = masks[di]
                if np.sum(mask) < 5:
                    continue
                for hi, h in enumerate(HORIZONS):
                    fwd = fr[:, hi]
                    stats = self.validator.bucket_statistics(fwd, mask)
                    dd = _future_drawdown(price[:len(fwd)], h)
                    ru = _future_runup(price[:len(fwd)], h)
                    stats["expected_drawdown"] = float(np.nanmean(dd[mask])) if np.sum(mask) > 0 else 0.0
                    stats["expected_runup"] = float(np.nanmean(ru[mask])) if np.sum(mask) > 0 else 0.0
                    stats["profit_probability"] = float(np.mean(fwd[mask] > 0)) if np.sum(mask) > 0 else 0.5
                    key = f"D{di + 1}_H{h}"
                    var_surface[key] = stats
            surface[var] = var_surface

        print("  Expectancy Surface:")
        for var in PRIMARY_VARIABLES:
            vs = surface[var]
            d1_h20 = vs.get("D1_H20", {})
            d10_h20 = vs.get("D10_H20", {})
            print(f"    {var:25s}: D1 mean={d1_h20.get('mean', 0):.6f}, "
                  f"D10 mean={d10_h20.get('mean', 0):.6f}, "
                  f"D10 pp={d10_h20.get('profit_probability', 0):.3f}")

        return AELResult("expectancy_surface", "COMPLETE", metrics={"surface": surface})
