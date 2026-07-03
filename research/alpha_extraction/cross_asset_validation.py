"""RQ8: Do expectancy surfaces transfer across assets?"""

from __future__ import annotations

from typing import Any

import numpy as np

from research.alpha_extraction.alpha_validator import (
    AlphaValidator, AELResult, PRIMARY_VARIABLES, HORIZONS, TARGET_ASSETS,
    _future_returns,
)


class CrossAssetValidation:
    def __init__(self, validator: AlphaValidator):
        self.validator = validator

    def _expectancy_profile(self, asset: str) -> dict:
        data = self.validator.load_asset_data(asset)
        signals = self.validator.compute_signals(data)
        price = data["price"]
        fr_all = _future_returns(price, np.array(HORIZONS, dtype=np.int32))
        profile = {}
        for var in PRIMARY_VARIABLES:
            sig = np.asarray(signals[var], dtype=np.float64)
            n = min(len(sig), fr_all.shape[0])
            sig, fr = sig[:n], fr_all[:n]
            _, masks = self.validator.decile_bins(sig)
            top_mean = float(np.nanmean(fr[:, 2][masks[-1]])) if np.sum(masks[-1]) > 5 else 0.0
            bot_mean = float(np.nanmean(fr[:, 2][masks[0]])) if np.sum(masks[0]) > 5 else 0.0
            profile[var] = {"top_decile_mean": top_mean, "bottom_decile_mean": bot_mean,
                            "spread": top_mean - bot_mean}
        return profile

    def run(self) -> AELResult:
        profiles = {}
        for asset in TARGET_ASSETS:
            try:
                profiles[asset] = self._expectancy_profile(asset)
            except Exception as e:
                profiles[asset] = {"error": str(e)}

        valid = [a for a in TARGET_ASSETS if "error" not in profiles.get(a, {})]

        similarity = {}
        if len(valid) >= 2:
            ref = profiles[valid[0]]
            for asset in valid[1:]:
                sim = {}
                for var in PRIMARY_VARIABLES:
                    r_spread = ref.get(var, {}).get("spread", 0)
                    a_spread = profiles[asset].get(var, {}).get("spread", 0)
                    r_top = ref.get(var, {}).get("top_decile_mean", 0)
                    a_top = profiles[asset].get(var, {}).get("top_decile_mean", 0)
                    sim[var] = {
                        "spread_diff": a_spread - r_spread,
                        "spread_ratio": a_spread / max(abs(r_spread), 1e-12),
                        "top_diff": a_top - r_top,
                    }
                similarity[f"{valid[0]}_vs_{asset}"] = sim

        print("  Cross-Asset Expectancy (H20 top-decile spread):")
        for asset in valid:
            for var in PRIMARY_VARIABLES:
                p = profiles[asset].get(var, {})
                print(f"    {asset:8s} {var:20s}: spread={p.get('spread', 0):.6f}")

        transferability = all(profiles[valid[0]].get(v, {}).get("spread", 0) *
                              profiles[a].get(v, {}).get("spread", 0) > 0
                              for v in PRIMARY_VARIABLES for a in valid[1:]) if len(valid) >= 2 else False

        print(f"    Sign consistency: {'YES' if transferability else 'NO'}")

        return AELResult("cross_asset_validation", "COMPLETE" if transferability else "INCONCLUSIVE",
                         metrics={"profiles": profiles, "similarity": similarity,
                                  "transferability": transferability})
