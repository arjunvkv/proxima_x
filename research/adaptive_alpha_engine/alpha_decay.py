from __future__ import annotations

import numpy as np
from research.adaptive_alpha_engine.aae_validator import AAEValidator, AAEResult, HORIZONS, _future_returns


class AlphaDecay:
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
        window_size = max(int(n * 0.5), 100)
        step_size = max(int(n * 0.1), 50)
        h20_idx = 2

        rolling_pp: list[float] = []
        rolling_mean: list[float] = []
        rolling_sharpe: list[float] = []

        start = 0
        while start + window_size <= n:
            end = start + window_size

            es_slice = energy_storage[start:end]
            threshold = float(np.nanpercentile(es_slice, 90))
            mask = es_slice > threshold

            fut = fut_ret[start:end, h20_idx]
            vals = fut[mask]

            if np.sum(mask) >= 5:
                pp = float(np.mean(vals > 0))
                mean_val = float(np.nanmean(vals))
                std_val = float(np.nanstd(vals))
                sharpe = mean_val / max(std_val, 1e-12)
            else:
                pp = 0.5
                mean_val = 0.0
                sharpe = 0.0

            rolling_pp.append(pp)
            rolling_mean.append(mean_val)
            rolling_sharpe.append(sharpe)

            start += step_size

        rp = np.array(rolling_pp, dtype=np.float64)
        rm = np.array(rolling_mean, dtype=np.float64)
        rs = np.array(rolling_sharpe, dtype=np.float64)

        mean_pp = float(np.mean(rp))
        std_pp = float(np.std(rp))
        mean_rm = float(np.mean(rm))
        std_rm = float(np.std(rm))
        mean_rs = float(np.mean(rs))
        std_rs = float(np.std(rs))

        x = np.arange(len(rs), dtype=np.float64)
        slope, intercept = np.polyfit(x, rs, 1)

        if slope < -0.01:
            classification = "DECAYING"
        elif std_rs > 0.5:
            classification = "CYCLICAL"
        elif mean_rs > 0.3 and abs(slope) < 0.01:
            classification = "STABLE"
        else:
            classification = "REGIME_DEPENDENT"

        print(f"\nAlpha Decay Analysis for {self.asset}:")
        print(f"  Windows: {len(rp)}")
        print(f"  Rolling PP - Mean: {mean_pp:.3f}, Std: {std_pp:.3f}")
        print(f"  Rolling Mean - Mean: {mean_rm:.6f}, Std: {std_rm:.6f}")
        print(f"  Rolling Sharpe - Mean: {mean_rs:.3f}, Std: {std_rs:.3f}")
        print(f"  Trend Slope: {slope:.4f}")
        print(f"  Classification: {classification}")

        return AAEResult(
            rq_name="alpha_decay",
            status="completed",
            metrics={
                "asset": self.asset,
                "n_windows": len(rp),
                "rolling_pp": rolling_pp,
                "rolling_mean": rolling_mean,
                "rolling_sharpe": rolling_sharpe,
                "mean_pp": mean_pp,
                "std_pp": std_pp,
                "mean_return": mean_rm,
                "std_return": std_rm,
                "mean_sharpe": mean_rs,
                "std_sharpe": std_rs,
                "trend_slope": float(slope),
                "trend_intercept": float(intercept),
                "classification": classification,
            },
        )
