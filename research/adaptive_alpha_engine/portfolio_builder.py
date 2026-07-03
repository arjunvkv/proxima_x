from __future__ import annotations

import numpy as np
from research.adaptive_alpha_engine.aae_validator import (
    AAEValidator, AAEResult, HORIZONS, TARGET_ASSETS, _future_returns,
)


class PortfolioBuilder:
    def __init__(self, validator: AAEValidator):
        self.validator = validator

    def run(self) -> AAEResult:
        h20_idx = 2
        horizons_arr = np.array(HORIZONS, dtype=np.int32)

        all_returns: list[np.ndarray] = []
        asset_names: list[str] = []
        asset_signal_counts: list[int] = []
        asset_standalone: dict[str, dict] = {}

        for asset in TARGET_ASSETS:
            data = self.validator.load_asset_data(asset)
            price = data["price"]

            signals = self.validator.compute_signals(data)
            energy_storage = signals["energy_storage"]

            es_z = (energy_storage - np.nanmean(energy_storage)) / max(
                np.nanstd(energy_storage), 1e-12
            )

            fut_ret = _future_returns(price, horizons_arr)
            fwd = fut_ret[:, h20_idx]

            top_decile = energy_storage > float(np.nanpercentile(energy_storage, 90))
            vals = fwd[top_decile]

            signal_count = int(np.sum(top_decile))

            if len(vals) >= 5:
                sa_mean = float(np.nanmean(vals))
                sa_std = float(np.nanstd(vals))
                sa_sharpe = sa_mean / max(sa_std, 1e-12)
                sa_pp = float(np.mean(vals > 0))

                all_returns.append(vals)
                asset_names.append(asset)
                asset_signal_counts.append(signal_count)
                asset_standalone[asset] = {
                    "mean": sa_mean, "std": sa_std,
                    "sharpe": sa_sharpe, "pp": sa_pp,
                    "n_signals": signal_count, "n": len(vals),
                }

                print(
                    f"  {asset}: top-decile mean={sa_mean:.6f} "
                    f"sharpe={sa_sharpe:.3f} pp={sa_pp:.3f} n_signals={signal_count}"
                )
            else:
                print(f"  {asset}: insufficient data, skipping")

        if len(all_returns) < 2:
            return AAEResult(
                rq_name="portfolio_builder", status="failed",
                metrics={"error": "Need at least 2 assets with sufficient data"},
            )

        min_len = min(len(r) for r in all_returns)
        aligned = np.array([r[:min_len] for r in all_returns], dtype=np.float64)

        n_assets = len(aligned)
        portfolio_metrics: dict[str, dict] = {}

        ew_returns = np.mean(aligned, axis=0)
        ew_mean = float(np.nanmean(ew_returns))
        ew_std = float(np.nanstd(ew_returns))
        ew_sharpe = ew_mean / max(ew_std, 1e-12)
        ew_pp = float(np.mean(ew_returns > 0))
        portfolio_metrics["equal_weight"] = {
            "mean": ew_mean, "std": ew_std, "sharpe": ew_sharpe, "pp": ew_pp,
        }

        asset_vols = np.array([float(np.nanstd(r)) for r in all_returns], dtype=np.float64)
        inv_vol = 1.0 / np.maximum(asset_vols, 1e-12)
        vol_weights = inv_vol / np.sum(inv_vol)
        vw_returns = np.dot(vol_weights, aligned)
        vw_mean = float(np.nanmean(vw_returns))
        vw_std = float(np.nanstd(vw_returns))
        vw_sharpe = vw_mean / max(vw_std, 1e-12)
        vw_pp = float(np.mean(vw_returns > 0))
        portfolio_metrics["volatility_weight"] = {
            "mean": vw_mean, "std": vw_std, "sharpe": vw_sharpe, "pp": vw_pp,
            "weights": vol_weights.tolist(),
        }

        sc = np.array(asset_signal_counts, dtype=np.float64)
        sig_weights = sc / np.sum(sc)
        sw_returns = np.dot(sig_weights, aligned)
        sw_mean = float(np.nanmean(sw_returns))
        sw_std = float(np.nanstd(sw_returns))
        sw_sharpe = sw_mean / max(sw_std, 1e-12)
        sw_pp = float(np.mean(sw_returns > 0))
        portfolio_metrics["signal_weight"] = {
            "mean": sw_mean, "std": sw_std, "sharpe": sw_sharpe, "pp": sw_pp,
            "weights": sig_weights.tolist(),
        }

        print(f"\nPortfolio Comparison:")
        for name, pm in portfolio_metrics.items():
            print(
                f"  {name:20s}  mean={pm['mean']:.6f}  std={pm['std']:.6f}  "
                f"sharpe={pm['sharpe']:.3f}  pp={pm['pp']:.3f}"
            )

        print(f"\nSingle Asset Comparison (ES top-decile H20):")
        for asset, sa in asset_standalone.items():
            print(
                f"  {asset:10s}  mean={sa['mean']:.6f}  sharpe={sa['sharpe']:.3f}  "
                f"pp={sa['pp']:.3f}  n_signals={sa['n_signals']}"
            )

        return AAEResult(
            rq_name="portfolio_builder",
            status="completed",
            metrics={
                "assets": asset_names,
                "asset_standalone": asset_standalone,
                "portfolio_metrics": portfolio_metrics,
                "n_assets": n_assets,
                "aligned_length": min_len,
            },
        )
