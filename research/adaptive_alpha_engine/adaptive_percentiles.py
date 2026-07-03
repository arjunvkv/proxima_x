from __future__ import annotations

import numpy as np

from research.adaptive_alpha_engine.aae_validator import (
    AAEValidator, AAEResult, HORIZONS,
    _future_returns, _numba_rolling_percentile,
)


class AdaptivePercentiles:
    def __init__(self, validator: AAEValidator, asset: str = "EURJPY"):
        self.validator = validator
        self.asset = asset

    def run(self) -> AAEResult:
        data = self.validator.load_asset_data(self.asset)
        signals = self.validator.compute_signals(data)
        es = signals["energy_storage"]
        price = signals["price"]

        future_ret = _future_returns(price, np.array(HORIZONS, dtype=np.int32))

        static_threshold = np.nanpercentile(es, 90)
        static_signal = es > static_threshold

        static_results = {}
        for hi, h in enumerate(HORIZONS):
            ret = future_ret[:, hi]
            n = min(len(static_signal), len(ret))
            mask = static_signal[:n]
            vals = ret[:n][mask]
            if len(vals) < 5:
                static_results[h] = {"mean": 0.0, "pp": 0.5, "sharpe": 0.0, "n": 0}
            else:
                m = float(np.nanmean(vals))
                s = float(np.nanstd(vals))
                static_results[h] = {
                    "mean": m,
                    "pp": float(np.mean(vals > 0)),
                    "sharpe": m / max(s, 1e-12),
                    "n": int(np.sum(mask)),
                }

        adaptive_windows = [252, 504, 756]
        adaptive_results = {}
        for w in adaptive_windows:
            rolling_90 = _numba_rolling_percentile(es.astype(np.float64), w, 90.0)
            adaptive_signal = es > rolling_90

            sig_freq = float(np.sum(adaptive_signal)) / (len(adaptive_signal) / 50000.0)
            n_sigs = int(np.sum(adaptive_signal))

            window_results = {}
            for hi, h in enumerate(HORIZONS):
                ret = future_ret[:, hi]
                n = min(len(adaptive_signal), len(ret))
                mask = adaptive_signal[:n]
                vals = ret[:n][mask]
                if len(vals) < 5:
                    window_results[h] = {"mean": 0.0, "pp": 0.5, "sharpe": 0.0, "n": 0}
                else:
                    m = float(np.nanmean(vals))
                    s = float(np.nanstd(vals))
                    window_results[h] = {
                        "mean": m,
                        "pp": float(np.mean(vals > 0)),
                        "sharpe": m / max(s, 1e-12),
                        "n": int(np.sum(mask)),
                    }

            adaptive_results[w] = {
                "signal_frequency": sig_freq,
                "total_signals": n_sigs,
                "horizons": window_results,
            }

        print("=" * 72)
        print("ADAPTIVE PERCENTILES — Static vs Rolling Threshold Comparison")
        print(f"Asset: {self.asset}")
        print("=" * 72)

        print("\n--- Static Alpha (ES > 90th percentile of all data) ---")
        print(f"{'Horizon':>8} {'Mean':>10} {'PP':>8} {'Sharpe':>8} {'N':>6}")
        for h in HORIZONS:
            r = static_results[h]
            print(f"{h:>8d} {r['mean']:>10.6f} {r['pp']:>8.4f} {r['sharpe']:>8.4f} {r['n']:>6d}")

        for w in adaptive_windows:
            ar = adaptive_results[w]
            print(f"\n--- Adaptive Alpha (rolling {w} bars — {ar['total_signals']} signals, {ar['signal_frequency']:.1f}/yr) ---")
            print(f"{'Horizon':>8} {'Mean':>10} {'PP':>8} {'Sharpe':>8} {'N':>6}")
            for h in HORIZONS:
                r = ar["horizons"][h]
                print(f"{h:>8d} {r['mean']:>10.6f} {r['pp']:>8.4f} {r['sharpe']:>8.4f} {r['n']:>6d}")

        static_pp_avg = np.mean([static_results[h]["pp"] for h in HORIZONS])
        best_adaptive_pp = 0.0
        best_adaptive_window = 0
        for w in adaptive_windows:
            pp_avg = np.mean([adaptive_results[w]["horizons"][h]["pp"] for h in HORIZONS])
            if pp_avg > best_adaptive_pp:
                best_adaptive_pp = pp_avg
                best_adaptive_window = w

        static_freq = float(np.sum(static_signal)) / (len(static_signal) / 50000.0)
        static_freq_var = 0.0
        for w in adaptive_windows:
            freq_var = abs(adaptive_results[w]["signal_frequency"] - static_freq) / max(static_freq, 1e-12)
            static_freq_var = max(static_freq_var, freq_var)

        print(f"\n--- Comparison ---")
        print(f"Static avg PP:           {static_pp_avg:.4f}")
        print(f"Best Adaptive avg PP:    {best_adaptive_pp:.4f} (window={best_adaptive_window})")
        print(f"Static signal frequency: {static_freq:.1f}/yr")
        for w in adaptive_windows:
            print(f"Adaptive ({w}) signal freq: {adaptive_results[w]['signal_frequency']:.1f}/yr")

        higher_pp = "Adaptive" if best_adaptive_pp > static_pp_avg else "Static"
        print(f"\nHigher PP:                {higher_pp}")
        print(f"Adaptive signal frequency is more stable across regimes by construction.")
        print("=" * 72)

        metrics = {
            "static": static_results,
            "adaptive": adaptive_results,
            "comparison": {
                "static_avg_pp": float(static_pp_avg),
                "best_adaptive_avg_pp": float(best_adaptive_pp),
                "best_adaptive_window": best_adaptive_window,
                "higher_pp": higher_pp,
                "static_signal_frequency": float(static_freq),
            },
        }

        result = AAEResult(
            rq_name=f"AdaptivePercentiles.{self.asset}",
            status="complete",
            metrics=metrics,
        )
        return result
