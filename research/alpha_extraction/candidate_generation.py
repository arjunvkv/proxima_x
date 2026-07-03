"""RQ7: Generate simple candidate rules — outcome analysis only, no trading."""

from __future__ import annotations

from typing import Any

import numpy as np

from research.alpha_extraction.alpha_validator import (
    AlphaValidator, AELResult, PRIMARY_VARIABLES, HORIZONS,
    _future_returns, _future_drawdown, _future_runup,
)


class CandidateGeneration:
    def __init__(self, validator: AlphaValidator, asset: str = "EURJPY"):
        self.validator = validator
        self.asset = asset

    def run(self) -> AELResult:
        data = self.validator.load_asset_data(self.asset)
        signals = self.validator.compute_signals(data)
        price = data["price"]

        horizons_arr = np.array(HORIZONS, dtype=np.int32)
        fr_all = _future_returns(price, horizons_arr)

        candidates = {}

        # Rule 1: energy_storage > threshold AND memory_density > threshold
        for es_pct in [50, 70, 80, 90]:
            for md_pct in [50, 70, 80, 90]:
                es = np.asarray(signals["energy_storage"], dtype=np.float64)
                md = np.asarray(signals["memory_density"], dtype=np.float64)
                n = min(len(es), len(md), fr_all.shape[0])
                es, md, fr = es[:n], md[:n], fr_all[:n]

                es_thresh = np.nanpercentile(es, es_pct)
                md_thresh = np.nanpercentile(md, md_pct)
                rule_mask = (es > es_thresh) & (md > md_thresh)

                if np.sum(rule_mask) < 5:
                    continue

                rule_results = {}
                for hi, h in enumerate(HORIZONS):
                    fwd = fr[:, hi]
                    stats = self.validator.bucket_statistics(fwd, rule_mask)
                    dd = _future_drawdown(price[:len(fwd)], h)
                    ru = _future_runup(price[:len(fwd)], h)
                    stats["drawdown"] = float(np.nanmean(dd[rule_mask]))
                    stats["runup"] = float(np.nanmean(ru[rule_mask]))
                    stats["profit_probability"] = float(np.mean(fwd[rule_mask] > 0))
                    rule_results[f"H{h}"] = stats

                key = f"ES>{es_pct}_MD>{md_pct}"
                candidates[key] = rule_results

        print("  Candidate Generation (selected rules @ H20):")
        sorted_candidates = []
        for key, cr in candidates.items():
            h20 = cr.get("H20", {})
            sorted_candidates.append((key, h20.get("mean", 0), h20.get("profit_probability", 0),
                                      h20.get("drawdown", 0), h20.get("n", 0)))
        sorted_candidates.sort(key=lambda x: x[1], reverse=True)

        for key, mean, pp, dd, n in sorted_candidates[:8]:
            print(f"    {key:20s}: mean={mean:.6f}, pp={pp:.3f}, dd={dd:.4f}, n={n}")

        return AELResult("candidate_generation", "COMPLETE",
                         metrics={"candidates": candidates})
