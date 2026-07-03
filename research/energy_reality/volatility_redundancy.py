from __future__ import annotations
import numpy as np
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from research.energy_reality.energy_validator import EnergyValidator, ERLResult, VOL_METRICS, TARGET_ASSETS


class VolatilityRedundancy:
    def __init__(self, validator: EnergyValidator):
        self.validator = validator

    def run(self) -> ERLResult:
        per_asset = {}
        max_r2_overall = 0.0
        max_r2_metric = ""
        max_r2_asset = ""

        for asset in TARGET_ASSETS:
            self.validator.load(asset)
            es = self.validator.es_signal()
            vol_metrics = self.validator.vol_metrics

            asset_results = {}
            asset_max_r2 = 0.0
            asset_max_metric = ""

            for metric_name in VOL_METRICS:
                metric = vol_metrics[metric_name]
                valid = ~(np.isnan(es) | np.isnan(metric))
                es_v = es[valid]
                metric_v = metric[valid]

                if len(es_v) < 10:
                    continue

                corr, _ = pearsonr(es_v, metric_v)
                mi = mutual_info_regression(metric_v.reshape(-1, 1), es_v, n_neighbors=5)[0]
                lr = LinearRegression()
                lr.fit(metric_v.reshape(-1, 1), es_v)
                r2 = lr.score(metric_v.reshape(-1, 1), es_v)

                asset_results[metric_name] = {
                    "corr": corr,
                    "mi": mi,
                    "r2": r2,
                }

                if r2 > asset_max_r2:
                    asset_max_r2 = r2
                    asset_max_metric = metric_name

            per_asset[asset] = asset_results

            if asset_max_r2 > max_r2_overall:
                max_r2_overall = asset_max_r2
                max_r2_metric = asset_max_metric
                max_r2_asset = asset

        passes = max_r2_overall < 0.80

        metric_summary = {}
        for metric_name in VOL_METRICS:
            corrs = []
            r2s = []
            for asset in TARGET_ASSETS:
                if metric_name in per_asset[asset]:
                    corrs.append(per_asset[asset][metric_name]["corr"])
                    r2s.append(per_asset[asset][metric_name]["r2"])
            if corrs:
                metric_summary[metric_name] = {
                    "mean_corr": float(np.mean(corrs)),
                    "max_corr": float(np.max(corrs)),
                    "mean_r2": float(np.mean(r2s)),
                    "max_r2": float(np.max(r2s)),
                }

        self._print_results(per_asset, metric_summary)

        if passes:
            verdict = "PASS: No volatility metric explains >80% of ES variance"
        else:
            verdict = f"FAIL: {max_r2_metric} explains {max_r2_overall:.2%} of ES variance on {max_r2_asset}"

        metrics = {
            "per_asset": per_asset,
            "max_r2_overall": max_r2_overall,
            "max_r2_metric": max_r2_metric,
            "max_r2_asset": max_r2_asset,
            "passes": passes,
            "verdict": verdict,
        }

        return ERLResult(
            rq_name="ERL-1: Volatility Redundancy Test",
            status="PASS" if passes else "FAIL",
            metrics=metrics,
        )

    def _print_results(self, per_asset: dict, metric_summary: dict) -> None:
        r2s = "R" + chr(0xb2)
        header = f"{'Metric':<25} {'Corr':>8} {'MI':>8} {r2s:>8}"
        sep = "-" * 50
        for asset in TARGET_ASSETS:
            print(f"\n=== {asset} ===")
            print(header)
            print(sep)
            for metric_name in VOL_METRICS:
                if metric_name in per_asset[asset]:
                    r = per_asset[asset][metric_name]
                    print(f"{metric_name:<25} {r['corr']:>8.4f} {r['mi']:>8.4f} {r['r2']:>8.4f}")
        print("\n=== Metric Summary (across assets) ===")
        sum_header = f"{'Metric':<25} {'Mean Corr':>10} {'Max Corr':>10} {'Mean '+r2s:>10} {'Max '+r2s:>10}"
        print(sum_header)
        print("-" * 65)
        for metric_name in VOL_METRICS:
            if metric_name in metric_summary:
                s = metric_summary[metric_name]
                print(f"{metric_name:<25} {s['mean_corr']:>10.4f} {s['max_corr']:>10.4f} {s['mean_r2']:>10.4f} {s['max_r2']:>10.4f}")
