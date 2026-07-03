"""RQ10: Classify each variable by alpha potential."""

from __future__ import annotations

from typing import Any

import numpy as np

from research.alpha_extraction.alpha_validator import (
    AlphaValidator, AELResult, VALIDATED_VARIABLES, PRIMARY_VARIABLES, SECONDARY_VARIABLES,
    HORIZONS, _future_returns, _future_drawdown, _future_runup,
)


CLASS_LABELS = [
    "NO_EDGE",
    "RISK_ONLY",
    "CONTEXT_ONLY",
    "WEAK_ALPHA",
    "CONDITIONAL_ALPHA",
    "ROBUST_ALPHA",
]


class AlphaClassifier:
    def __init__(self, validator: AlphaValidator, asset: str = "EURJPY",
                 cross_asset_metrics: dict | None = None,
                 cross_time_metrics: dict | None = None):
        self.validator = validator
        self.asset = asset
        self.cross_asset = cross_asset_metrics or {}
        self.cross_time = cross_time_metrics or {}

    def _score_variable(self, var: str, signals: dict, price: np.ndarray,
                        fr_all: np.ndarray) -> dict:
        sig = np.asarray(signals[var], dtype=np.float64)
        n = min(len(sig), fr_all.shape[0])
        sig, fr = sig[:n], fr_all[:n]
        _, masks = self.validator.decile_bins(sig)

        h20 = 2
        top = masks[-1]
        bot = masks[0]

        t_mean = float(np.nanmean(fr[:, h20][top])) if np.sum(top) > 5 else 0.0
        b_mean = float(np.nanmean(fr[:, h20][bot])) if np.sum(bot) > 5 else 0.0
        t_pp = float(np.mean(fr[:, h20][top] > 0)) if np.sum(top) > 5 else 0.5
        b_pp = float(np.mean(fr[:, h20][bot] > 0)) if np.sum(bot) > 5 else 0.5

        overall_mean = float(np.nanmean(fr[:, h20]))
        overall_std = float(np.nanstd(fr[:, h20]))

        separation = abs(t_mean - b_mean)
        risk_ratio = float(np.nanstd(fr[:, h20][top])) / max(overall_std, 1e-12) if np.sum(top) > 5 else 1.0

        return {
            "top_decile_mean": t_mean,
            "bottom_decile_mean": b_mean,
            "separation": separation,
            "top_profit_prob": t_pp,
            "bottom_profit_prob": b_pp,
            "risk_ratio": risk_ratio,
            "overall_mean": overall_mean,
        }

    def run(self) -> AELResult:
        data = self.validator.load_asset_data(self.asset)
        signals = self.validator.compute_signals(data)
        price = data["price"]
        fr_all = _future_returns(price, np.array(HORIZONS, dtype=np.int32))

        classifications = {}
        for var in VALIDATED_VARIABLES:
            scores = self._score_variable(var, signals, price, fr_all)
            sep = scores["separation"]
            pp = scores["top_profit_prob"]
            risk_ratio = scores["risk_ratio"]
            top_mean = scores["top_decile_mean"]

            if sep < 0.0005 or abs(top_mean) < 0.0003:
                cls = "NO_EDGE"
            elif risk_ratio > 1.5 and abs(top_mean) < 0.001:
                cls = "RISK_ONLY"
            elif pp < 0.52 and abs(top_mean) < 0.001:
                cls = "CONTEXT_ONLY"
            elif sep > 0.005 and pp > 0.58:
                cls = "ROBUST_ALPHA"
            elif sep > 0.002 or pp > 0.55:
                cls = "CONDITIONAL_ALPHA"
            else:
                cls = "WEAK_ALPHA"

            classifications[var] = {"classification": cls, **scores}

        print("  Alpha Classification:")
        for var in VALIDATED_VARIABLES:
            c = classifications[var]
            print(f"    {var:25s}: {c['classification']:20s} "
                  f"(sep={c['separation']:.6f}, pp={c['top_profit_prob']:.3f}, "
                  f"top_mean={c['top_decile_mean']:.6f})")

        contains_expectancy = any(
            c["classification"] in ("ROBUST_ALPHA", "CONDITIONAL_ALPHA")
            for c in classifications.values()
        )

        print(f"\n  Proxima expectancy source: {'YES' if contains_expectancy else 'NO'}")

        if contains_expectancy:
            best = max(
                ((var, c) for var, c in classifications.items()
                 if c["classification"] in ("ROBUST_ALPHA", "CONDITIONAL_ALPHA")),
                key=lambda x: x[1]["separation"],
            )
            print(f"  Best source: {best[0]} ({best[1]['classification']})")

        return AELResult("alpha_classifier", "COMPLETE",
                         metrics={"classifications": classifications,
                                  "contains_expectancy": contains_expectancy})
