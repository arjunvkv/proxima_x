"""RQ6: Which state combinations create the strongest outcome asymmetry?"""

from __future__ import annotations

from typing import Any

import numpy as np

from research.alpha_extraction.alpha_validator import (
    AlphaValidator, AELResult, PRIMARY_VARIABLES, HORIZONS,
    _future_returns, _future_drawdown, _future_runup,
)


COMBINATIONS = [
    ["energy_storage", "memory_density"],
    ["energy_storage", "adaptive_time"],
    ["memory_density", "adaptive_time"],
    ["energy_storage", "memory_density", "adaptive_time"],
]
TOP_PCTS = [10, 20, 30]


class StateCombinations:
    def __init__(self, validator: AlphaValidator, asset: str = "EURJPY"):
        self.validator = validator
        self.asset = asset

    def run(self) -> AELResult:
        data = self.validator.load_asset_data(self.asset)
        signals = self.validator.compute_signals(data)
        price = data["price"]

        horizons_arr = np.array(HORIZONS, dtype=np.int32)
        fr_all = _future_returns(price, horizons_arr)

        def _top_mask(sig: np.ndarray, pct: float) -> np.ndarray:
            thresh = np.nanpercentile(sig, 100 - pct)
            return sig > thresh

        results = {}
        for combo in COMBINATIONS:
            signals_list = [np.asarray(signals[v], dtype=np.float64) for v in combo]
            n = min(len(signals_list[0]), fr_all.shape[0])

            combo_key = "_".join(combo)
            combo_results = {}
            for pct in TOP_PCTS:
                masks = [_top_mask(s[:n], pct) for s in signals_list]
                combined = np.all(masks, axis=0)

                if np.sum(combined) < 5:
                    continue

                fr = fr_all[:n]
                for hi, h in enumerate(HORIZONS):
                    fwd = fr[:, hi]
                    stats = self.validator.bucket_statistics(fwd, combined)
                    dd = _future_drawdown(price[:len(fwd)], h)
                    ru = _future_runup(price[:len(fwd)], h)
                    stats["drawdown"] = float(np.nanmean(dd[combined]))
                    stats["runup"] = float(np.nanmean(ru[combined]))
                    stats["profit_probability"] = float(np.mean(fwd[combined] > 0))
                    stats["n"] = int(np.sum(combined))
                    combo_results[f"top{pct}_H{h}"] = stats

            results[combo_key] = combo_results

            h20_keys = [k for k in combo_results if k.endswith("_H20")]
            if h20_keys:
                print(f"    {combo_key:50s}: top20 mean={combo_results.get('top20_H20', {}).get('mean', 0):.6f}, "
                      f"pp={combo_results.get('top20_H20', {}).get('profit_probability', 0):.3f}")

        return AELResult("state_combinations", "COMPLETE", metrics={"combinations": results})
