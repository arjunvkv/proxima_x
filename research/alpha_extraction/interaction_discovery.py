"""RQ3: Do variable interactions create stronger separation than individual variables?"""

from __future__ import annotations

from typing import Any

import numpy as np

from research.alpha_extraction.alpha_validator import (
    AlphaValidator, AELResult, PRIMARY_VARIABLES, HORIZONS,
    _future_returns, _future_drawdown, _future_runup,
)


PAIRS = [
    ("energy_storage", "memory_density"),
    ("energy_storage", "adaptive_time"),
    ("memory_density", "adaptive_time"),
]
TRIPLE = ("energy_storage", "memory_density", "adaptive_time")


class InteractionDiscovery:
    def __init__(self, validator: AlphaValidator, asset: str = "EURJPY"):
        self.validator = validator
        self.asset = asset

    def run(self) -> AELResult:
        data = self.validator.load_asset_data(self.asset)
        signals = self.validator.compute_signals(data)
        price = data["price"]

        horizons_arr = np.array(HORIZONS, dtype=np.int32)
        fr_all = _future_returns(price, horizons_arr)

        def _decile_masks(sig: np.ndarray) -> list[np.ndarray]:
            _, masks = self.validator.decile_bins(sig)
            return masks

        pair_results = {}
        for v1, v2 in PAIRS:
            s1 = np.asarray(signals[v1], dtype=np.float64)
            s2 = np.asarray(signals[v2], dtype=np.float64)
            n = min(len(s1), len(s2), fr_all.shape[0])
            s1, s2, fr = s1[:n], s2[:n], fr_all[:n]
            m1 = _decile_masks(s1)
            m2 = _decile_masks(s2)

            # 10x10 interaction grid
            grid = {}
            best_pair = None
            best_mean = -np.inf
            worst_mean = np.inf
            for di in range(10):
                for dj in range(10):
                    pair_mask = m1[di] & m2[dj]
                    if np.sum(pair_mask) < 5:
                        continue
                    fwd = fr[:, 2]
                    stats = self.validator.bucket_statistics(fwd, pair_mask)
                    grid[f"D{di + 1}_D{dj + 1}"] = stats
                    if stats["mean"] > best_mean:
                        best_mean = stats["mean"]
                        best_pair = (di + 1, dj + 1)
                    if stats["mean"] < worst_mean:
                        worst_mean = stats["mean"]
                        worst_pair = (di + 1, dj + 1)

            single_spread = {}
            for v, masks in [(v1, m1), (v2, m2)]:
                top = np.sum(masks[-1])
                bot = np.sum(masks[0])
                if top > 5 and bot > 5:
                    t_mean = float(np.nanmean(fr[:, 2][masks[-1]]))
                    b_mean = float(np.nanmean(fr[:, 2][masks[0]]))
                    single_spread[v] = t_mean - b_mean

            interaction_spread = best_mean - worst_mean if best_pair and worst_pair else 0
            gain = interaction_spread / (np.mean(list(single_spread.values())) + 1e-12)

            pair_results[f"{v1}x{v2}"] = {
                "grid": {k: v for k, v in list(grid.items())[:20]},
                "best_pair": best_pair,
                "worst_pair": worst_pair,
                "interaction_spread": interaction_spread,
                "single_spreads": single_spread,
                "gain_vs_single": gain,
            }

            print(f"    {v1}x{v2}: spread={interaction_spread:.6f}, gain={gain:.2f}x single")

        # Triple interaction — use terciles to avoid sparsity
        s1 = np.asarray(signals[TRIPLE[0]], dtype=np.float64)
        s2 = np.asarray(signals[TRIPLE[1]], dtype=np.float64)
        s3 = np.asarray(signals[TRIPLE[2]], dtype=np.float64)
        n = min(len(s1), len(s2), len(s3), fr_all.shape[0])
        s1, s2, s3, fr = s1[:n], s2[:n], s3[:n], fr_all[:n]

        def _tercile_masks(sig: np.ndarray) -> list[np.ndarray]:
            b = [np.nanpercentile(sig, p) for p in [0, 33.3, 66.6, 100]]
            return [
                sig <= b[1],
                (sig > b[1]) & (sig <= b[2]),
                sig > b[2],
            ]

        t1, t2, t3 = _tercile_masks(s1), _tercile_masks(s2), _tercile_masks(s3)
        triple_best = -np.inf
        triple_worst = np.inf
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    tm = t1[i] & t2[j] & t3[k]
                    if np.sum(tm) < 5:
                        continue
                    m = float(np.nanmean(fr[:, 2][tm]))
                    if m > triple_best:
                        triple_best = m
                    if m < triple_worst:
                        triple_worst = m
        triple_spread = triple_best - triple_worst

        pair_results["triple"] = {
            "spread": triple_spread,
            "best_mean": triple_best,
            "worst_mean": triple_worst,
        }

        print(f"    Triple interaction spread: {triple_spread:.6f}")

        return AELResult("interaction_discovery", "COMPLETE", metrics=pair_results)
