from __future__ import annotations

import numpy as np

from research.adaptive_alpha_engine.aae_validator import (
    AAEValidator, AAEResult, TIME_WINDOWS, HORIZONS,
    _numba_skew, _numba_kurtosis, _numba_rolling_percentile,
)


class ThresholdDrift:
    def __init__(self, validator: AAEValidator, asset: str = "EURJPY"):
        self.validator = validator
        self.asset = asset

    def run(self) -> AAEResult:
        all_energy = []
        window_labels = []

        for start, end, label in TIME_WINDOWS:
            data = self.validator.load_data_window(self.asset, start, end)
            signals = self.validator.compute_signals(data)
            es = signals["energy_storage"]
            all_energy.append(es)
            window_labels.append(label)

        original_90th = np.nanpercentile(all_energy[0], 90)

        metrics = {}
        for idx, (label, es) in enumerate(zip(window_labels, all_energy)):
            m = float(np.nanmean(es))
            s = float(np.nanstd(es))
            sk = float(_numba_skew(es.astype(np.float64)))
            ku = float(_numba_kurtosis(es.astype(np.float64)))
            p50 = float(np.nanpercentile(es, 50))
            p75 = float(np.nanpercentile(es, 75))
            p90 = float(np.nanpercentile(es, 90))
            p95 = float(np.nanpercentile(es, 95))
            p99 = float(np.nanpercentile(es, 99))
            n_above_orig = int(np.sum(es > original_90th))

            metrics[label] = {
                "mean": m, "std": s, "skew": sk, "kurtosis": ku,
                "p50": p50, "p75": p75, "p90": p90, "p95": p95, "p99": p99,
                "n_above_original_90th": n_above_orig,
                "total_n": len(es),
            }

        threshold_shifts = {}
        quantile_shifts = {}
        for idx in range(1, len(window_labels)):
            prev_label = window_labels[idx - 1]
            curr_label = window_labels[idx]
            prev_p90 = metrics[prev_label]["p90"]
            curr_p90 = metrics[curr_label]["p90"]
            threshold_shifts[f"{prev_label}_to_{curr_label}"] = curr_p90 - prev_p90

            curr_es = all_energy[idx]
            orig_quantile = float(np.mean(curr_es < original_90th) * 100)
            quantile_shifts[curr_label] = orig_quantile

        print("=" * 72)
        print("THRESHOLD DRIFT ANALYSIS — Cross-Time Alpha Failure Diagnosis")
        print(f"Asset: {self.asset}")
        print("=" * 72)

        header = f"{'Window':<14} {'Mean':>8} {'Std':>8} {'Skew':>7} {'Kurt':>7} {'p50':>8} {'p75':>8} {'p90':>8} {'p95':>8} {'p99':>8} {'#AboveOrig':>10}"
        print(header)
        print("-" * 96)
        for label in window_labels:
            m = metrics[label]
            print(
                f"{label:<14} {m['mean']:>8.4f} {m['std']:>8.4f} {m['skew']:>7.3f} {m['kurtosis']:>7.3f} "
                f"{m['p50']:>8.4f} {m['p75']:>8.4f} {m['p90']:>8.4f} {m['p95']:>8.4f} {m['p99']:>8.4f} "
                f"{m['n_above_original_90th']:>10d}"
            )

        print("\nThreshold Shifts (90th percentile movement):")
        for k, v in threshold_shifts.items():
            direction = "^" if v > 0 else "v"
            print(f"  {k}: {v:+.4f} {direction}")

        print("\nQuantile of Original Threshold in Later Windows:")
        for label, qs in quantile_shifts.items():
            print(f"  {label}: original 90th pct = {qs:.1f}th percentile of distribution")

        print("\n-- Insight --")
        print("If original_90th falls below the 90th percentile in later windows,")
        print("a fixed threshold produces too many (or too few) signals, breaking cross-time validity.")
        print("=" * 72)

        metrics["threshold_shifts"] = threshold_shifts
        metrics["quantile_shifts"] = quantile_shifts
        metrics["original_90th"] = float(original_90th)

        result = AAEResult(
            rq_name=f"ThresholdDrift.{self.asset}",
            status="complete",
            metrics=metrics,
        )
        return result
