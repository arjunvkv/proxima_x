from __future__ import annotations

import numpy as np

from research.adaptive_alpha_engine.aae_validator import AAEValidator, AAEResult


class DeploymentValidator:
    def __init__(
        self,
        validator: AAEValidator,
        asset: str = "EURJPY",
        prior_results: dict[str, AAEResult] | None = None,
    ):
        self.validator = validator
        self.asset = asset
        self.prior_results = prior_results or {}

    def _get(self, rq_key: str, metric: str, default: float = 0.0, is_str: bool = False) -> float | str:
        rq = self.prior_results.get(rq_key)
        if rq is None:
            return default
        if hasattr(rq, "metrics") and isinstance(rq.metrics, dict):
            val = rq.metrics.get(metric, default)
            val = default if val is None else val
            return val if is_str else float(val)
        if isinstance(rq, dict):
            val = rq.get("metrics", {}).get(metric, default)
            val = default if val is None else val
            return val if is_str else float(val)
        return default

    @staticmethod
    def _try_key(d: dict, key: str | int, default_val=None):
        """Look up key in dict trying both the given type and its inverse."""
        if key in d:
            return d[key]
        alt = str(key) if isinstance(key, int) else (int(key) if isinstance(key, str) and key.lstrip('-').isdigit() else None)
        if alt is not None and alt in d:
            return d[alt]
        return default_val

    def _extract_nested(self, rq_key: str, *path, default: float = 0.0) -> float:
        rq = self.prior_results.get(rq_key)
        if rq is None:
            return default
        metrics = rq.metrics if hasattr(rq, "metrics") and isinstance(rq.metrics, dict) else {}
        val = metrics
        try:
            for key in path:
                val = self._try_key(val, key)
                if val is None:
                    return default
            return float(val) if val is not None else default
        except (TypeError, IndexError, ValueError):
            return default

    def run(self) -> AAEResult:
        results = self.prior_results

        # RQ1: Threshold Drift — check status
        rq1 = results.get("RQ1: Threshold Drift")
        rq1_score = 1.0 if (rq1 and hasattr(rq1, "status") and rq1.status != "ERROR") else 0.0

        # RQ2: Adaptive Percentiles — compare static vs best adaptive H20 sharpe
        rq2_static_sharpe = self._extract_nested("RQ2: Adaptive Percentiles", "static", 20, "sharpe", default=-999.0)
        adaptive_metrics = results.get("RQ2: Adaptive Percentiles")
        best_adaptive_sharpe = -999.0
        if adaptive_metrics and hasattr(adaptive_metrics, "metrics"):
            adaptive_dict = adaptive_metrics.metrics.get("adaptive", {})
            if isinstance(adaptive_dict, dict):
                for window_data in adaptive_dict.values():
                    horizons = window_data.get("horizons", {})
                    if not isinstance(horizons, dict):
                        continue
                    sh = self._try_key(horizons, 20, {}).get("sharpe", -999.0)
                    if sh is not None and sh > best_adaptive_sharpe:
                        best_adaptive_sharpe = sh
        rq2_score = 1.0 if best_adaptive_sharpe > rq2_static_sharpe else 0.0

        # RQ3: Walk-Forward
        rq3_survival = self._get("RQ3: Walk-Forward", "survival_rate", 0.0)

        # RQ4: Alpha Decay
        rq4_decay = self._get("RQ4: Alpha Decay", "classification", "DECAYING", is_str=True)
        if rq4_decay in ("STABLE", "CYCLICAL"):
            rq4_score = 1.0
        elif rq4_decay == "REGIME_DEPENDENT":
            rq4_score = 0.5
        else:
            rq4_score = 0.0

        # RQ5: Portfolio Builder — best portfolio sharpe (volatility_weight usually best)
        rq5_sharpe = self._extract_nested("RQ5: Portfolio Builder", "portfolio_metrics", "volatility_weight", "sharpe", default=0.0)
        rq5_score = 1.0 if rq5_sharpe > 0.3 else 0.0

        # RQ6: Adaptive Time Overlay — check comparison
        rq6_better = self._extract_nested("RQ6: Adaptive Time Overlay", "comparison", "better_sharpe", default=False)
        rq6_ratio = self._extract_nested("RQ6: Adaptive Time Overlay", "comparison", "improvement_ratio", default=0.0)
        rq6_score = 1.0 if rq6_better or rq6_ratio > 0.05 else 0.0

        # RQ7: Execution Stress — death_stress is None if survives all
        rq7_stress = self._extract_nested("RQ7: Execution Stress", "death_stress", default=None)
        if rq7_stress is None:
            rq7_score = 1.0
        else:
            rq7_score = min(rq7_stress / 5.0, 1.0)

        # RQ8: Capacity Model — None means no ceiling = unlimited capacity
        rq8_ceiling = self._extract_nested("RQ8: Capacity Model", "capacity_ceiling", default=None)
        if rq8_ceiling is None:
            rq8_score = 1.0
        else:
            rq8_score = min(rq8_ceiling / 1_000_000_000.0, 1.0)

        # RQ9: Live System
        rq9_win_rate = self._get("RQ9: Live System", "win_rate", 0.0)
        rq9_score = 1.0 if rq9_win_rate > 0.55 else 0.0

        # Aggregate evidence scores (each normalised to [0, 1])
        rq_scores = [
            rq1_score,
            rq2_score,
            rq3_survival,
            rq4_score,
            rq5_score,
            rq6_score,
            rq7_score,
            rq8_score,
            rq9_score,
        ]

        avg_score = float(np.mean(rq_scores))

        # Classification logic
        if avg_score > 0.85 and rq3_survival > 0.6 and (rq7_stress is None or rq7_stress >= 5.0):
            classification = "PRODUCTION_READY"
            would_deploy = True
        elif avg_score > 0.70 and rq3_survival > 0.4:
            classification = "LIVE_PILOT_READY"
            would_deploy = True
        elif avg_score > 0.50:
            classification = "PAPER_TRADE_READY"
            would_deploy = True
        elif avg_score > 0.30:
            classification = "PROMISING"
            would_deploy = False
        else:
            classification = "RESEARCH_ONLY"
            would_deploy = False

        confidence = avg_score

        print(f"\n  [Deployment Validator — {classification}]")
        print(f"  Avg Score: {avg_score:.3f}")
        print(f"  Confidence: {confidence:.3f}")
        print(f"  Would Deploy: {would_deploy}")
        print(f"  Evidence:")
        labels = [
            "RQ1 Drift", "RQ2 Percentiles", "RQ3 Survival", "RQ4 Decay",
            "RQ5 Portfolio", "RQ6 AT Overlay", "RQ7 Execution", "RQ8 Capacity",
            "RQ9 Live",
        ]
        for i, (lbl, sc) in enumerate(zip(labels, rq_scores), 1):
            print(f"    RQ{i}: {lbl} = {sc:.3f}")

        return AAEResult(
            rq_name="RQ10: Deployment Validator",
            status=classification,
            metrics={
                "avg_score": avg_score,
                "confidence": confidence,
                "would_deploy": would_deploy,
                "classification": classification,
                "rq_scores": {f"RQ{i+1}": float(s) for i, s in enumerate(rq_scores)},
            },
        )
