import json
import numpy as np
from proxima_v1.core.signal_engine import SignalEngine


class SignalLatency:
    PERIODS = [
        ("2020-01-01", "2022-01-01", "2020-2022"),
        ("2022-01-01", "2024-01-01", "2022-2024"),
        ("2024-01-01", "2026-01-01", "2024-2026"),
    ]

    def __init__(self, asset: str = "EURJPY"):
        self.asset = asset

    def run(self) -> dict:
        results = {}
        for start, end, label in self.PERIODS:
            eng = SignalEngine(self.asset)
            eng.precompute_full(start, end)
            full_res = np.nan_to_num(eng._full_residual, nan=0.0)
            full_es = np.nan_to_num(eng._full_es, nan=0.0)
            full_at = np.nan_to_num(eng._full_at, nan=0.0)
            n = len(full_res)
            if n < 504:
                results[label] = {"signal_frequency": 0.0, "mean_strength": 0.0, "median_strength": 0.0, "signal_duration_mean": 0.0, "signal_duration_median": 0.0, "inter_signal_gap_mean": 0.0, "threshold_90th": 0.0, "pp_at_h20": 0.0}
                continue

            rolling_res = np.full(n, 0.5)
            rolling_es = np.full(n, 0.5)
            rolling_at = np.full(n, 0.5)
            for i in range(503, n):
                r_slice = full_res[max(0, i - 503):i + 1]
                e_slice = full_es[max(0, i - 503):i + 1]
                a_slice = full_at[max(0, i - 503):i + 1]
                rolling_res[i] = float(np.sum(r_slice <= full_res[i])) / len(r_slice)
                rolling_es[i] = float(np.sum(e_slice <= full_es[i])) / len(e_slice)
                rolling_at[i] = float(np.sum(a_slice <= full_at[i])) / len(a_slice)

            composite = np.clip(0.60 * rolling_res + 0.30 * rolling_es + 0.10 * rolling_at, 0.0, 1.0)

            signal_freq = float(np.mean(composite[504:] > 0.7))
            mean_strength = float(np.mean(composite[504:]))
            median_strength = float(np.median(composite[504:]))
            threshold_90th = float(np.nanpercentile(composite[504:], 90))

            in_signal = composite > 0.7
            durations = []
            current_len = 0
            for v in in_signal[504:]:
                if v:
                    current_len += 1
                else:
                    if current_len > 0:
                        durations.append(current_len)
                        current_len = 0
            if current_len > 0:
                durations.append(current_len)

            signal_duration_mean = float(np.mean(durations)) if durations else 0.0
            signal_duration_median = float(np.median(durations)) if durations else 0.0

            signal_starts = []
            prev = False
            for j in range(504, n):
                if in_signal[j] and not prev:
                    signal_starts.append(j)
                prev = in_signal[j]

            if len(signal_starts) > 1:
                gaps = np.diff(np.array(signal_starts, dtype=np.float64))
                inter_signal_gap_mean = float(np.mean(gaps))
            else:
                inter_signal_gap_mean = 0.0

            price = eng._data.get("price", np.zeros(n))
            h20_returns = np.full(n, np.nan)
            for j in range(n - 20):
                h20_returns[j] = float(np.log(price[j + 20] / price[j]))
            signal_mask = composite > 0.7
            valid = signal_mask & ~np.isnan(h20_returns)
            if np.sum(valid) > 0:
                pp_at_h20 = float(np.mean(h20_returns[valid] > 0.0))
            else:
                pp_at_h20 = 0.5

            results[label] = {
                "signal_frequency": signal_freq,
                "mean_strength": mean_strength,
                "median_strength": median_strength,
                "signal_duration_mean": signal_duration_mean,
                "signal_duration_median": signal_duration_median,
                "inter_signal_gap_mean": inter_signal_gap_mean,
                "threshold_90th": threshold_90th,
                "pp_at_h20": pp_at_h20,
            }

        return results

    def save(self, path: str):
        results = self.run()
        with open(path, "w") as f:
            json.dump(results, f, indent=2, default=str)
