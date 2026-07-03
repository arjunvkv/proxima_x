"""RQ9: How many opportunities per year survive filters?"""

from __future__ import annotations

import numpy as np

from research.alpha_reality.arl_validator import ARLValidator, ARLResult


class CapacityAnalysis:
    def __init__(self, validator: ARLValidator, asset: str = "EURJPY"):
        self.validator = validator
        self.asset = asset

    def run(self) -> ARLResult:
        data = self.validator.load_asset_data(self.asset)
        signals = self.validator.compute_signals(data)
        price = data["price"]
        n_total = len(price)

        years = max(n_total / 50000, 1.0)

        alpha = self.validator.alpha_signal(signals)
        n = min(len(alpha), n_total)

        results = {}
        for pct in [20, 10, 5, 2, 1]:
            thresh = float(np.nanpercentile(alpha, 100 - pct))
            mask = alpha > thresh
            n_signals = int(np.sum(mask))
            signals_per_year = n_signals / years
            results[f"top{pct}%"] = {
                "n_total": n_signals,
                "signals_per_year": signals_per_year,
                "signals_per_month": signals_per_year / 12,
                "signals_per_week": signals_per_year / 52,
                "pct_of_data": pct,
            }

        print("  Capacity Analysis:")
        for pct in [20, 10, 5, 2, 1]:
            r = results.get(f"top{pct}%", {})
            spy = r.get("signals_per_year", 0)
            spm = r.get("signals_per_month", 0)
            print(f"    top{pct:2d}%: {r.get('n_total', 0):5d} total, {spy:.0f}/year, {spm:.1f}/month")

        top20 = results.get("top20%", {})
        viable = top20.get("signals_per_month", 0) >= 5

        status = "PASSED" if viable else "FAILED"
        return ARLResult("capacity_analysis", status, metrics=results)
