"""RQ8: Test all AT/ES/MD combinations — determine minimum viable alpha structure."""

from __future__ import annotations

import numpy as np
import numba
from numpy.typing import NDArray

from research.alpha_reality.arl_validator import (
    ARLValidator, ARLResult, HORIZONS, _future_returns, _zscore,
)


COMBINATIONS = {
    "AT only": ["adaptive_time"],
    "ES only": ["energy_storage"],
    "MD only": ["memory_density"],
    "ES+MD": ["energy_storage", "memory_density"],
    "ES+AT": ["energy_storage", "adaptive_time"],
    "MD+AT": ["memory_density", "adaptive_time"],
    "ES+MD+AT": ["energy_storage", "memory_density", "adaptive_time"],
}


class InteractionNecessity:
    def __init__(self, validator: ARLValidator, asset: str = "EURJPY"):
        self.validator = validator
        self.asset = asset

    def _combo_alpha(self, signals: dict, vars: list[str]) -> NDArray[np.float64]:
        prod = None
        for v in vars:
            sig = np.asarray(signals[v], dtype=np.float64)
            z = _zscore(sig)
            if prod is None:
                prod = z
            else:
                prod = prod * z
        return prod if prod is not None else np.zeros(len(signals.get("price", [0])))

    def run(self) -> ARLResult:
        data = self.validator.load_asset_data(self.asset)
        price = data["price"]
        signals = self.validator.compute_signals(data)
        fr_all = _future_returns(price, np.array(HORIZONS, dtype=np.int32))

        results = {}
        for label, vars in COMBINATIONS.items():
            alpha = self._combo_alpha(signals, vars)
            combo_results = {}
            for hi, h in enumerate(HORIZONS):
                eval_r = self.validator.eval_alpha(alpha, fr_all, hi)
                combo_results[f"H{h}"] = eval_r
            results[label] = combo_results

        print("  Interaction Necessity @ H20:")
        sorted_results = []
        for label, cr in results.items():
            h20 = cr.get("H20", {})
            sorted_results.append((label, h20.get("mean", 0), h20.get("pp", 0)))
        sorted_results.sort(key=lambda x: x[1], reverse=True)

        for label, mean, pp in sorted_results:
            print(f"    {label:15s}: mean={mean:.6f}, pp={pp:.3f}")

        best = sorted_results[0][0] if sorted_results else "NONE"
        print(f"    Minimum viable structure: {best}")

        status = "COMPLETE"
        return ARLResult("interaction_necessity", status,
                         metrics={"combinations": results, "best": best})
