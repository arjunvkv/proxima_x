from __future__ import annotations

from typing import Any

import numpy as np

from research.interaction_asymmetry.interaction_validator import InteractionValidator, IAEResult
from research.adaptive_alpha_engine.aae_validator import HORIZONS, TARGET_ASSETS, TIME_WINDOWS, _zscore, _future_returns


class ContradictionEngine:
    def __init__(self, validator: InteractionValidator, asset: str = "EURJPY"):
        self.validator = validator
        self.asset = asset

    def run(self) -> IAEResult:
        self.validator.load(self.asset)
        md_z = self.validator.md_z()
        es_z = self.validator.es_z()
        at_z = self.validator.at_z()

        ctype = self.validator.detect_contradictions(md_z, es_z, at_z)
        fut_ret = self.validator.fut_ret
        fwd_h20 = fut_ret[:, 2]
        lim = min(len(ctype), len(fwd_h20))
        ctype = ctype[:lim]
        fwd_h20 = fwd_h20[:lim]

        contradiction_level_metrics: dict[str, Any] = {}
        for level in range(4):
            mask = ctype == level
            count = int(np.sum(mask))
            pct = float(np.mean(mask)) * 100.0
            if count >= 5:
                vals = fwd_h20[mask]
                mean = float(np.nanmean(vals))
                std = float(np.nanstd(vals))
                pp = float(np.mean(vals > 0))
                sharpe = mean / max(std, 1e-12)
            else:
                mean = 0.0
                std = 0.0
                pp = 0.5
                sharpe = 0.0
            contradiction_level_metrics[str(level)] = {
                "mean": mean, "pp": pp, "sharpe": sharpe, "std": std, "n": count, "count": count, "pct": pct,
            }

        spec_pairs: list[tuple[str, np.ndarray]] = [
            ("memory_bullish_energy_bearish", (md_z > 0) & (es_z < 0)),
            ("memory_bearish_energy_bullish", (md_z < 0) & (es_z > 0)),
            ("memory_bearish_adaptive_bullish", (md_z < 0) & (at_z > 0)),
            ("memory_bullish_adaptive_bearish", (md_z > 0) & (at_z < 0)),
            ("energy_bullish_adaptive_bearish", (es_z > 0) & (at_z < 0)),
            ("energy_bearish_adaptive_bullish", (es_z < 0) & (at_z > 0)),
        ]
        specific_contradictions: dict[str, Any] = {}
        for name, mask in spec_pairs:
            count = int(np.sum(mask))
            if count >= 5:
                vals = fwd_h20[mask]
                mean = float(np.nanmean(vals))
                std = float(np.nanstd(vals))
                pp = float(np.mean(vals > 0))
                sharpe = mean / max(std, 1e-12)
            else:
                mean = 0.0
                std = 0.0
                pp = 0.5
                sharpe = 0.0
            specific_contradictions[name] = {
                "mean": mean, "pp": pp, "sharpe": sharpe, "std": std, "n": count, "count": count,
            }

        agree_mask = ctype == 0
        contra_mask = ctype > 0
        if np.sum(agree_mask) >= 5:
            av = fwd_h20[agree_mask]
            agree_mean = float(np.nanmean(av))
            agree_pp = float(np.mean(av > 0))
        else:
            agree_mean = 0.0
            agree_pp = 0.5
        if np.sum(contra_mask) >= 5:
            cv = fwd_h20[contra_mask]
            contra_mean = float(np.nanmean(cv))
            contra_pp = float(np.mean(cv > 0))
        else:
            contra_mean = 0.0
            contra_pp = 0.5

        agreement_vs_contradiction: dict[str, float] = {
            "agreement_mean": agree_mean,
            "agreement_pp": agree_pp,
            "contradiction_mean": contra_mean,
            "contradiction_pp": contra_pp,
        }

        print("=== Contradiction Engine (RQ5) ===")
        print("\nContradiction Level Metrics:")
        for level in range(4):
            m = contradiction_level_metrics[str(level)]
            print(f"  Level {level}: count={m['count']} pct={m['pct']:.2f}% mean={m['mean']:.6f} pp={m['pp']:.4f} sharpe={m['sharpe']:.4f}")
        print("\nSpecific Contradiction Types:")
        for name, m in specific_contradictions.items():
            print(f"  {name}: count={m['count']} mean={m['mean']:.6f} pp={m['pp']:.4f} sharpe={m['sharpe']:.4f}")
        print(f"\nAgreement vs Contradiction (H20):")
        print(f"  Agreement:     mean={agree_mean:.6f} pp={agree_pp:.4f}")
        print(f"  Contradiction: mean={contra_mean:.6f} pp={contra_pp:.4f}")

        metrics: dict[str, Any] = {
            "contradiction_level_metrics": contradiction_level_metrics,
            "specific_contradictions": specific_contradictions,
            "agreement_vs_contradiction": agreement_vs_contradiction,
        }

        return IAEResult(rq_name="RQ5_Hidden_Contradictions", status="COMPLETE", metrics=metrics)
