"""RQ5: Train on EURJPY, evaluate on USDJPY, GBPJPY, XAUUSD."""

from __future__ import annotations

import numpy as np

from research.alpha_reality.arl_validator import (
    ARLValidator, ARLResult, HORIZONS, TARGET_ASSETS,
    _future_returns,
)


class CrossAssetTransfer:
    def __init__(self, validator: ARLValidator):
        self.validator = validator

    def run(self) -> ARLResult:
        train_asset = "EURJPY"
        results = {}

        train_data = self.validator.load_asset_data(train_asset)
        train_signals = self.validator.compute_signals(train_data)
        train_alpha = self.validator.alpha_signal(train_signals)

        # Train threshold on EURJPY combined alpha
        train_threshold = float(np.nanpercentile(train_alpha, 90))
        results["train_asset"] = train_asset
        results["threshold"] = train_threshold

        for asset in TARGET_ASSETS:
            try:
                data = self.validator.load_asset_data(asset)
                signals = self.validator.compute_signals(data)
                price = data["price"]
                fr_all = _future_returns(price, np.array(HORIZONS, dtype=np.int32))

                test_alpha = self.validator.alpha_signal(signals)
                n = min(len(test_alpha), fr_all.shape[0])
                mask = test_alpha[:n] > train_threshold

                asset_results = {}
                for hi, h in enumerate(HORIZONS):
                    fwd = fr_all[:n, hi]
                    sig_mask = mask[:len(fwd)]
                    if np.sum(sig_mask) < 5:
                        asset_results[f"H{h}"] = {"mean": 0, "pp": 0.5, "n": 0, "survives": False}
                        continue
                    vals = fwd[sig_mask]
                    m = float(np.nanmean(vals))
                    s = float(np.nanstd(vals))
                    asset_results[f"H{h}"] = {
                        "mean": m,
                        "pp": float(np.mean(vals > 0)),
                        "std": s,
                        "sharpe": m / max(s, 1e-12),
                        "n": int(np.sum(sig_mask)),
                        "survives": float(np.mean(vals > 0)) > 0.52,
                    }
                results[asset] = asset_results
            except Exception as e:
                results[asset] = {"error": str(e)}

        print("  Cross-Asset Transfer @ H20 (EURJPY combined alpha threshold):")
        for asset in TARGET_ASSETS:
            r = results.get(asset, {})
            h20 = r.get("H20", {})
            print(f"    {asset:8s}: mean={h20.get('mean', 0):.6f}, pp={h20.get('pp', 0):.3f}, "
                  f"n={h20.get('n', 0)}, survives={h20.get('survives', False)}")

        n_survive = sum(1 for a in TARGET_ASSETS
                        if results.get(a, {}).get("H20", {}).get("survives", False))
        status = "PASSED" if n_survive >= 2 else "FAILED"
        print(f"    Assets where alpha survives: {n_survive}/{len(TARGET_ASSETS)}")

        return ARLResult("cross_asset_transfer", status, metrics=results)
