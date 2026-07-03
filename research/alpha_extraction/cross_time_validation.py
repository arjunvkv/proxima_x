"""RQ9: Do expectancy relationships survive across time?"""

from __future__ import annotations

from typing import Any

import numpy as np

from research.alpha_extraction.alpha_validator import (
    AlphaValidator, AELResult, PRIMARY_VARIABLES, HORIZONS, TIME_WINDOWS,
    _future_returns,
)


class CrossTimeValidation:
    def __init__(self, validator: AlphaValidator, asset: str = "EURJPY"):
        self.validator = validator
        self.asset = asset

    def _expectancy_profile(self, start: str, end: str) -> dict:
        data = self.validator.load_data_window(self.asset, start, end)
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
        for start, end, label in TIME_WINDOWS:
            try:
                profiles[label] = self._expectancy_profile(start, end)
            except Exception as e:
                profiles[label] = {"error": str(e)}

        valid = [l for l, v in profiles.items() if "error" not in v]

        print("  Cross-Time Expectancy (H20 top-decile spread):")
        for label in valid:
            for var in PRIMARY_VARIABLES:
                p = profiles[label].get(var, {})
                print(f"    {label:12s} {var:20s}: spread={p.get('spread', 0):.6f}")

        survival = True
        if len(valid) >= 3:
            for var in PRIMARY_VARIABLES:
                spreads = [profiles[l].get(var, {}).get("spread", 0) for l in valid]
                sign_consistent = all(s * spreads[0] >= 0 for s in spreads)
                if not sign_consistent:
                    survival = False
                    break

        print(f"    Sign consistency across time: {'YES' if survival else 'NO'}")

        return AELResult("cross_time_validation", "PASSED" if survival else "FAILED",
                         metrics={"profiles": profiles, "survival": survival})
