"""
E3 — Promotion Score.

Computes a composite promotion score for each D1-D7 observer module
across five dimensions, determining which modules deserve promotion
into execution.

Run:  python -c "from validation.promotion_score import PromotionScorer"
"""

import statistics
from typing import Dict, List


# ---------------------------------------------------------------------------
# Dimension formulas (each normalised 0-1)
# ---------------------------------------------------------------------------

def _lift_score(delta: float) -> float:
    """Predictive lift from marginal contribution delta.

    Mapping: delta -0.1 -> 0.0, 0.0 -> 0.5, 0.1 -> 1.0
    """
    return max(0.0, min(1.0, (delta + 0.1) / 0.2))


def _calibration_score(ece: float) -> float:
    """Calibration quality from expected calibration error.

    ece=0.0 -> 1.0,  ece=0.2 -> 0.0
    """
    return max(0.0, 1.0 - ece * 5)


def _robustness_score(variance: float) -> float:
    """Regime robustness — lower variance is better."""
    return max(0.0, 1.0 - min(variance * 10, 1.0))


def _cost_score(cost: float) -> float:
    """Computational cost — lower cost is better."""
    return max(0.0, 1.0 - min(cost, 1.0))


def _stability_score(disagreement_rate: float) -> float:
    """Decision stability — lower disagreement is better."""
    return max(0.0, 1.0 - disagreement_rate)


# ---------------------------------------------------------------------------
# Dimension metadata
# ---------------------------------------------------------------------------

DIMENSIONS = [
    {"key": "predictive_lift",      "weight": 0.30, "scorer": _lift_score,          "raw_key": "lift_delta"},
    {"key": "calibration_quality",  "weight": 0.25, "scorer": _calibration_score,    "raw_key": "calibration_ece"},
    {"key": "regime_robustness",    "weight": 0.20, "scorer": _robustness_score,     "raw_key": "robustness_variance"},
    {"key": "computational_cost",   "weight": 0.10, "scorer": _cost_score,           "raw_key": "cost_ratio"},
    {"key": "decision_stability",   "weight": 0.15, "scorer": _stability_score,      "raw_key": "disagreement_rate"},
]

# Ensure weights sum to 1.0
_WEIGHT_SUM = sum(d["weight"] for d in DIMENSIONS)
assert abs(_WEIGHT_SUM - 1.0) < 1e-9, f"E3 weights sum to {_WEIGHT_SUM}, expected 1.0"


# ---------------------------------------------------------------------------
# Promotion Scorer
# ---------------------------------------------------------------------------

class PromotionScorer:
    """Computes composite promotion scores for D1-D7 observer modules."""

    def __init__(self):
        self._scores: Dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(self, module_name: str,
                lift_delta: float,
                calibration_ece: float,
                robustness_variance: float = 0.0,
                cost_ratio: float = 0.1,
                disagreement_rate: float = 0.0) -> dict:
        """Compute promotion score for one module.

        Parameters
        ----------
        module_name : str
            Observer identifier (e.g. "D1", "D2", … "D7").
        lift_delta : float
            Average delta from marginal contribution compare() —
            (quality_delta + confidence_delta + consensus_delta) / 3.
        calibration_ece : float
            Expected calibration error from ProbabilityCalibrator.
        robustness_variance : float
            Variance of quality metric across synthetic regimes.
        cost_ratio : float
            Estimated computational cost per decision (0–1).  Default 0.1.
        disagreement_rate : float
            Disagreement rate from shadow execution (0–1).

        Returns
        -------
        dict
            {
                "module": str,
                "composite": float,        # weighted sum [0,1]
                "dimensions": {
                    "predictive_lift":      {"score": float, "weight": 0.30, "raw": float},
                    "calibration_quality":  {"score": float, "weight": 0.25, "raw": float},
                    "regime_robustness":    {"score": float, "weight": 0.20, "raw": float},
                    "computational_cost":   {"score": float, "weight": 0.10, "raw": float},
                    "decision_stability":   {"score": float, "weight": 0.15, "raw": float},
                },
                "promotion_eligible": bool,  # True if composite >= 0.50
            }
        """
        raw_values = {
            "lift_delta": lift_delta,
            "calibration_ece": calibration_ece,
            "robustness_variance": robustness_variance,
            "cost_ratio": cost_ratio,
            "disagreement_rate": disagreement_rate,
        }

        dimensions = {}
        composite = 0.0

        for dim in DIMENSIONS:
            raw = raw_values[dim["raw_key"]]
            score = dim["scorer"](raw)
            dimensions[dim["key"]] = {
                "score": round(score, 6),
                "weight": dim["weight"],
                "raw": raw,
            }
            composite += score * dim["weight"]

        composite = round(composite, 6)
        result = {
            "module": module_name,
            "composite": composite,
            "dimensions": dimensions,
            "promotion_eligible": composite >= 0.50,
        }
        self._scores[module_name] = result
        return result

    def compute_from_evidence(self, module_name: str,
                               loo_result: dict,
                               calibration_result: dict,
                               shadow_result: dict) -> dict:
        """Convenience method that extracts metrics from E2/E4/E5 results.

        Parameters
        ----------
        module_name : str
            Observer identifier.
        loo_result : dict
            Result from :meth:`MarginalContributionAnalyzer.compare` (E2).
            Expected keys: ``mean_quality``, ``mean_confidence``, ``mean_consensus``.
        calibration_result : dict
            Result from :meth:`ProbabilityCalibrator.stats` or
            ``calibrate_scores()`` (E5).  Expected key: ``ece`` (or ``ece_raw``).
        shadow_result : dict
            Result from :meth:`ShadowExecutionEngine.stats` or a shadow
            decision summary (E4).  Expected key: ``disagreement_rate``.

        Returns
        -------
        dict
            Same structure as :meth:`compute`.
        """
        # --- lift delta from compare() result ---
        quality_delta = loo_result.get("mean_quality", 0.0)
        confidence_delta = loo_result.get("mean_confidence", 0.0)
        consensus_delta = loo_result.get("mean_consensus", 0.0)
        lift_delta = (quality_delta + confidence_delta + consensus_delta) / 3.0

        # --- calibration ECE ---
        # The calibrator may use "ece_raw" (degenerate/platt) or "ece" (stats)
        calibration_ece = calibration_result.get("ece",
                         calibration_result.get("ece_raw", 0.0))

        # --- disagreement rate ---
        disagreement_rate = shadow_result.get("disagreement_rate", 0.0)

        # Defaults for variance and cost
        robustness_variance = 0.0
        cost_ratio = 0.1

        return self.compute(
            module_name,
            lift_delta=lift_delta,
            calibration_ece=calibration_ece,
            robustness_variance=robustness_variance,
            cost_ratio=cost_ratio,
            disagreement_rate=disagreement_rate,
        )

    def rank(self) -> List[dict]:
        """Return all computed scores sorted by composite descending."""
        return sorted(
            self._scores.values(),
            key=lambda r: r["composite"],
            reverse=True,
        )

    def promote(self, threshold: float = 0.50) -> List[str]:
        """Return list of modules with composite >= threshold."""
        return sorted([
            r["module"] for r in self._scores.values()
            if r["composite"] >= threshold
        ])

    def explain(self, module_name: str) -> str:
        """Return human-readable explanation of the score."""
        if module_name not in self._scores:
            return f"No score computed for module '{module_name}'."

        r = self._scores[module_name]
        lines = [
            f"=== Promotion Score: {r['module']} ===",
            f"  Composite Score : {r['composite']:.4f}  "
            f"{'(ELIGIBLE)' if r['promotion_eligible'] else '(below threshold)'}",
            f"  Threshold       : 0.5000",
            "",
            "  Dimension Breakdown:",
        ]
        for dim_key, dim_info in sorted(r["dimensions"].items()):
            label = dim_key.replace("_", " ").title()
            score = dim_info["score"]
            weight = dim_info["weight"]
            raw = dim_info["raw"]
            weighted = score * weight
            lines.append(
                f"    {label:25s}  score={score:.4f}  "
                f"weight={weight:.2f}  weighted={weighted:.4f}  "
                f"raw={raw:.4f}"
            )
        lines.append("")
        lines.append(f"  Weighted Composite: {r['composite']:.4f}")
        return "\n".join(lines)

    def stats(self) -> dict:
        """Return aggregate statistics over all scored modules."""
        n = len(self._scores)
        if n == 0:
            return {"modules_scored": 0, "mean_composite": 0.0, "median_composite": 0.0}

        composites = [r["composite"] for r in self._scores.values()]
        return {
            "modules_scored": n,
            "mean_composite": round(statistics.mean(composites), 6),
            "median_composite": round(statistics.median(composites), 6),
        }
