"""RQ3: Does alpha disappear under timestamp shuffling and label randomization?"""

from __future__ import annotations

import numpy as np
import numba
from numpy.typing import NDArray

from research.alpha_reality.arl_validator import ARLValidator, ARLResult, HORIZONS, _future_returns, _numba_skew


@numba.jit(nopython=True, cache=True)
def _shuffle(arr: NDArray[np.float64]) -> NDArray[np.float64]:
    n = len(arr)
    out = arr.copy()
    for i in range(n - 1, 0, -1):
        j = np.random.randint(0, i + 1)
        out[i], out[j] = out[j], out[i]
    return out


class RandomizationTest:
    def __init__(self, validator: ARLValidator, asset: str = "EURJPY", n_shuffles: int = 200):
        self.validator = validator
        self.asset = asset
        self.n_shuffles = n_shuffles

    def run(self) -> ARLResult:
        np.random.seed(42)
        data = self.validator.load_asset_data(self.asset)
        price = data["price"]

        signals = self.validator.compute_signals(data)
        alpha = self.validator.alpha_signal(signals)
        fr_all = _future_returns(price, np.array(HORIZONS, dtype=np.int32))
        n = min(len(alpha), fr_all.shape[0])
        alpha, fr = alpha[:n], fr_all[:n]

        # Test 1: Shuffle timestamps of alpha signal
        shuffled_means = {h: [] for h in HORIZONS}
        shuffled_pps = {h: [] for h in HORIZONS}
        for _ in range(self.n_shuffles):
            alpha_shuffled = _shuffle(alpha)
            for hi, h in enumerate(HORIZONS):
                eval_r = self.validator.eval_alpha(alpha_shuffled, fr, hi)
                shuffled_means[h].append(eval_r["mean"])
                shuffled_pps[h].append(eval_r["pp"])

        # Test 2: Shuffle future returns (label randomization)
        label_shuffled_means = {h: [] for h in HORIZONS}
        for _ in range(self.n_shuffles):
            for hi, h in enumerate(HORIZONS):
                fwd = fr[:, hi].copy()
                fwd_shuffled = _shuffle(fwd)
                fr_shuffled = fr.copy()
                fr_shuffled[:, hi] = fwd_shuffled
                eval_r = self.validator.eval_alpha(alpha, fr_shuffled, hi)
                label_shuffled_means[h].append(eval_r["mean"])

        # Real alpha
        real = {}
        for hi, h in enumerate(HORIZONS):
            real[h] = self.validator.eval_alpha(alpha, fr, hi)

        results = {}
        for h in HORIZONS:
            real_mean = real[h]["mean"]
            ts_mean = float(np.mean(shuffled_means[h]))
            ts_std = float(np.std(shuffled_means[h]))
            ts_z = (real_mean - ts_mean) / max(ts_std, 1e-12)

            ls_mean = float(np.mean(label_shuffled_means[h]))
            ls_std = float(np.std(label_shuffled_means[h]))
            ls_z = (real_mean - ls_mean) / max(ls_std, 1e-12)

            ts_pp_dist = np.array(shuffled_pps[h])
            real_pp = real[h]["pp"]
            ts_p_value = float(np.mean(ts_pp_dist >= real_pp))

            results[f"H{h}"] = {
                "real": real[h],
                "timestamp_shuffle_mean": ts_mean,
                "timestamp_shuffle_std": ts_std,
                "timestamp_z_score": ts_z,
                "label_shuffle_mean": ls_mean,
                "label_shuffle_std": ls_std,
                "label_z_score": ls_z,
                "timestamp_p_value": ts_p_value,
                "survives_timestamp_test": ts_z > 2.0,
                "survives_label_test": ls_z > 2.0,
            }

        h20 = results.get("H20", {})
        ts_pass = h20.get("survives_timestamp_test", False)
        ls_pass = h20.get("survives_label_test", False)

        print(f"  Randomization Test @ H20:")
        print(f"    Real mean: {h20.get('real', {}).get('mean', 0):.6f}, pp={h20.get('real', {}).get('pp', 0):.3f}")
        print(f"    Timestamp shuffle z={h20.get('timestamp_z_score', 0):.2f}, p={h20.get('timestamp_p_value', 0):.4f}")
        print(f"    Label shuffle z={h20.get('label_z_score', 0):.2f}")
        print(f"    Survives shuffle: {'YES' if ts_pass else 'NO'}, survives label: {'YES' if ls_pass else 'NO'}")

        status = "PASSED" if (ts_pass and ls_pass) else "FAILED"
        return ARLResult("randomization_test", status, metrics=results)
