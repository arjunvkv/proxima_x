"""RQ5: Is compression's origin role universal across assets?"""

from __future__ import annotations

from typing import Any

import numpy as np

from research.compression_physics.compression_validator import CompressionValidator, CPIResult


ASSETS = ["EURJPY", "USDJPY", "GBPJPY", "XAUUSD"]


class AssetUniversality:
    def __init__(self, validator: CompressionValidator):
        self.validator = validator
        self._max_lag = 200

    def run(self) -> CPIResult:
        results = []
        for asset in ASSETS:
            try:
                data = self.validator.load_asset_data(asset)
                signals = self.validator.compute_signals(data)

                compression = np.asarray(signals["compression"], dtype=np.float64)
                es = np.asarray(signals["energy_storage"], dtype=np.float64)

                from research.compression_physics.compression_validator import _find_peak_lag
                lag, corr = _find_peak_lag(compression, es, self._max_lag)

                flow = self.validator.information_flow("compression", "energy_storage", signals)

                results.append({
                    "asset": asset,
                    "peak_lag": lag,
                    "peak_corr": corr,
                    "information_flow": flow,
                    "n_obs": len(compression),
                })
            except Exception as e:
                results.append({"asset": asset, "error": str(e)})

        valid = [r for r in results if "error" not in r]
        if not valid:
            return CPIResult("asset_universality", "FAILED", metrics={"results": results})

        # Does compression consistently precede energy_storage?
        lead_signs = [r["peak_lag"] < 0 for r in valid if r["peak_lag"] != 0]
        precession_consistency = sum(lead_signs) / max(len(lead_signs), 1)

        lags = [r["peak_lag"] for r in valid]
        lag_std = float(np.std(lags))
        corrs = [r["peak_corr"] for r in valid]

        metrics = {
            "results": results,
            "n_valid": len(valid),
            "precession_consistency": precession_consistency,
            "lag_mean": float(np.mean(lags)),
            "lag_std": lag_std,
            "lag_cv": lag_std / max(abs(np.mean(lags)), 1e-12),
            "corr_mean": float(np.mean(corrs)),
            "corr_std": float(np.std(corrs)),
            "min_corr": float(min(corrs)),
            "max_corr": float(max(corrs)),
        }

        print(f"  Cross-asset compression->energy correlation:")
        for r in valid:
            lead = "<" if r["peak_lag"] < 0 else ">" if r["peak_lag"] > 0 else "="
            print(f"    {r['asset']:8s}: lag={r['peak_lag']:+4d} {lead}, corr={r['peak_corr']:.4f}, flow={r['information_flow']:.6f}")

        print(f"  Lag CV: {metrics['lag_cv']:.2f}, corr range: {metrics['min_corr']:.4f} to {metrics['max_corr']:.4f}")

        if precession_consistency > 0.8 and abs(metrics["corr_mean"]) > 0.1:
            status = "PASSED"
            print(f"  Compression universally precedes energy_storage (consistency={precession_consistency:.2f})")
        elif precession_consistency > 0.5:
            status = "INCONCLUSIVE"
        else:
            status = "FAILED"

        return CPIResult("asset_universality", status, metrics=metrics)
