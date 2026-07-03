"""RQ7: Test alpha at top 20%, 10%, 5%, 2%, 1% — measure decay."""

from __future__ import annotations

import numpy as np

from research.alpha_reality.arl_validator import ARLValidator, ARLResult, HORIZONS, _future_returns


THRESHOLDS = [20, 10, 5, 2, 1]


class ThresholdStability:
    def __init__(self, validator: ARLValidator, asset: str = "EURJPY"):
        self.validator = validator
        self.asset = asset

    def run(self) -> ARLResult:
        data = self.validator.load_asset_data(self.asset)
        price = data["price"]

        signals = self.validator.compute_signals(data)
        alpha = self.validator.alpha_signal(signals)
        fr_all = _future_returns(price, np.array(HORIZONS, dtype=np.int32))
        n = min(len(alpha), fr_all.shape[0])
        alpha, fr = alpha[:n], fr_all[:n]

        results = {}
        for pct in THRESHOLDS:
            thresh = float(np.nanpercentile(alpha, 100 - pct))
            mask = alpha > thresh
            pct_results = {}
            for hi, h in enumerate(HORIZONS):
                fwd = fr[:, hi]
                sig_mask = mask[:len(fwd)]
                if np.sum(sig_mask) < 5:
                    pct_results[f"H{h}"] = {"mean": 0, "pp": 0.5, "n": 0}
                    continue
                vals = fwd[sig_mask]
                m = float(np.nanmean(vals))
                s = float(np.nanstd(vals))
                pct_results[f"H{h}"] = {
                    "mean": m,
                    "pp": float(np.mean(vals > 0)),
                    "std": s,
                    "sharpe": m / max(s, 1e-12),
                    "n": int(np.sum(sig_mask)),
                }
            results[f"top{pct}"] = pct_results

        decay = {}
        for h in HORIZONS:
            means = [results.get(f"top{pct}", {}).get(f"H{h}", {}).get("mean", 0) for pct in THRESHOLDS]
            decay[f"H{h}"] = {
                "means": means,
                "monotonic": all(means[i] <= means[i + 1] + 1e-8 for i in range(len(means) - 1)),
                "max_at_thinnest": means[-1] >= max(means) - 1e-8 if means else False,
            }

        print("  Threshold Stability @ H20:")
        for pct in THRESHOLDS:
            r = results.get(f"top{pct}", {}).get("H20", {})
            print(f"    top{pct:2d}%: mean={r.get('mean', 0):.6f}, pp={r.get('pp', 0):.3f}, "
                  f"sharpe={r.get('sharpe', 0):.3f}, n={r.get('n', 0)}")

        h20_decay = decay.get("H20", {})
        monotonic = h20_decay.get("monotonic", False)
        strengthens = h20_decay.get("max_at_thinnest", False)
        print(f"    Monotonic alpha growth: {'YES' if monotonic else 'NO'}")
        print(f"    Strongest at thinnest slice: {'YES' if strengthens else 'NO'}")

        status = "PASSED" if (monotonic or strengthens) else "FAILED"
        return ARLResult("threshold_stability", status, metrics={"deciles": results, "decay": decay})
