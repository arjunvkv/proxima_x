from __future__ import annotations

import numpy as np
from research.adaptive_alpha_engine.aae_validator import AAEValidator, AAEResult, HORIZONS, _future_returns


class WalkForward:
    def __init__(self, validator: AAEValidator, asset: str = "EURJPY"):
        self.validator = validator
        self.asset = asset

    def run(self) -> AAEResult:
        data = self.validator.load_asset_data(self.asset)
        price = data["price"]

        signals = self.validator.compute_signals(data)
        energy_storage = signals["energy_storage"]

        horizons_arr = np.array(HORIZONS, dtype=np.int32)
        fut_ret = _future_returns(price, horizons_arr)

        n = len(energy_storage)
        train_frac = 0.5
        test_frac = 0.2
        train_bars = int(n * train_frac)
        test_bars = int(n * test_frac)
        step = test_bars

        window_results = []
        survival_count = 0
        total_windows = 0

        train_start = 0
        while train_start + train_bars + test_bars <= n:
            train_end = train_start + train_bars
            test_start = train_end
            test_end = min(test_start + test_bars, n)

            train_es = energy_storage[train_start:train_end]
            train_threshold = float(np.nanpercentile(train_es, 90))

            test_es = energy_storage[test_start:test_end]
            mask = test_es > train_threshold

            window_metrics = {}
            for h_idx, horizon in enumerate(HORIZONS):
                fut = fut_ret[test_start:test_end, h_idx]
                n_mask = int(np.sum(mask))
                vals = fut[mask] if n_mask > 0 else np.array([])

                if n_mask < 5:
                    pp = 0.5
                    mean_val = 0.0
                    std_val = 0.0
                    sharpe = 0.0
                else:
                    pp = float(np.mean(vals > 0))
                    mean_val = float(np.nanmean(vals))
                    std_val = float(np.nanstd(vals))
                    sharpe = mean_val / max(std_val, 1e-12)

                window_metrics[f"H{horizon}"] = {
                    "pp": pp, "mean": mean_val, "sharpe": sharpe, "n": n_mask,
                }

            h20 = window_metrics.get("H20", {})
            if h20.get("pp", 0) > 0.55 and h20.get("mean", 0) > 0:
                survival_count += 1
            total_windows += 1

            print(
                f"  Window {total_windows}: train=[{train_start}:{train_end}] "
                f"test=[{test_start}:{test_end}] threshold={train_threshold:.6f} "
                f"H20_pp={h20.get('pp', 0):.3f} H20_mean={h20.get('mean', 0):.6f} "
                f"H20_sharpe={h20.get('sharpe', 0):.3f}"
            )

            window_results.append(window_metrics)
            train_start += step

        survival_rate = survival_count / max(total_windows, 1)
        print(f"\nWalk-Forward Results for {self.asset}:")
        print(f"  Total Windows: {total_windows}")
        print(f"  Survival Count (pp>0.55 & mean>0): {survival_count}")
        print(f"  Survival Rate: {survival_rate:.2%}")

        return AAEResult(
            rq_name="walk_forward",
            status="completed",
            metrics={
                "asset": self.asset,
                "total_windows": total_windows,
                "survival_count": survival_count,
                "survival_rate": survival_rate,
                "window_results": window_results,
            },
        )
