"""RQ6: Train on one period, evaluate on unseen future periods."""

from __future__ import annotations

import numpy as np

from research.alpha_reality.arl_validator import (
    ARLValidator, ARLResult, HORIZONS, TIME_WINDOWS,
    _future_returns, _zscore,
)


class CrossTimeTransfer:
    def __init__(self, validator: ARLValidator, asset: str = "EURJPY"):
        self.validator = validator
        self.asset = asset

    def run(self) -> ARLResult:
        results = {}

        # Train on first window: find alpha threshold
        train_start, train_end, train_label = TIME_WINDOWS[0]
        train_data = self.validator.load_data_window(self.asset, train_start, train_end)
        train_signals = self.validator.compute_signals(train_data)
        train_alpha = self.validator.alpha_signal(train_signals)
        train_threshold = float(np.nanpercentile(train_alpha, 90))

        results["train_window"] = train_label
        results["threshold"] = train_threshold

        for start, end, label in TIME_WINDOWS[1:]:
            try:
                data = self.validator.load_data_window(self.asset, start, end)
                signals = self.validator.compute_signals(data)
                price = data["price"]
                fr_all = _future_returns(price, np.array(HORIZONS, dtype=np.int32))

                test_alpha = self.validator.alpha_signal(signals)
                n = min(len(test_alpha), fr_all.shape[0])
                mask = test_alpha[:n] > train_threshold

                window_results = {}
                for hi, h in enumerate(HORIZONS):
                    fwd = fr_all[:n, hi]
                    sig_mask = mask[:len(fwd)]
                    if np.sum(sig_mask) < 5:
                        window_results[f"H{h}"] = {"mean": 0, "pp": 0.5, "n": 0, "survives": False}
                        continue
                    vals = fwd[sig_mask]
                    m = float(np.nanmean(vals))
                    s = float(np.nanstd(vals))
                    window_results[f"H{h}"] = {
                        "mean": m,
                        "pp": float(np.mean(vals > 0)),
                        "std": s,
                        "sharpe": m / max(s, 1e-12),
                        "n": int(np.sum(sig_mask)),
                        "survives": float(np.mean(vals > 0)) > 0.52,
                    }
                results[label] = window_results
            except Exception as e:
                results[label] = {"error": str(e)}

        print("  Cross-Time Transfer @ H20 (trained on 2018-2020):")
        for _, _, label in TIME_WINDOWS[1:]:
            r = results.get(label, {})
            h20 = r.get("H20", {})
            print(f"    {label:12s}: mean={h20.get('mean', 0):.6f}, pp={h20.get('pp', 0):.3f}, "
                  f"n={h20.get('n', 0)}, survives={h20.get('survives', False)}")

        n_survive = sum(1 for _, _, l in TIME_WINDOWS[1:]
                        if results.get(l, {}).get("H20", {}).get("survives", False))
        status = "PASSED" if n_survive >= 2 else "FAILED"
        print(f"    Future windows where alpha survives: {n_survive}/{len(TIME_WINDOWS) - 1}")

        return ARLResult("cross_time_transfer", status, metrics=results)
