"""
E6 — Promotion Policy.

The gate between E3 scoring and staged deployment.  Determines whether a
module passes formal promotion criteria with full explanation of each check.

Run:  python -c "from validation.promotion_policy import PromotionPolicy"
"""

from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Promotion Policy
# ---------------------------------------------------------------------------

class PromotionPolicy:
    """Formal policy engine for module promotion decisions.

    Evaluates a candidate module against configurable thresholds across six
    dimensions: predictive lift, calibration quality, false-veto rate,
    drawdown impact, sample size, and regime coverage.

    Parameters
    ----------
    minimum_predictive_lift : float
        Minimum required lift delta from marginal contribution analysis (E2).
    minimum_calibration : float
        Minimum required calibration quality (1 - ECE).
    maximum_false_veto_rate : float
        Maximum acceptable false-veto rate (E2 / E4).
    maximum_drawdown_increase : float
        Maximum acceptable increase in drawdown from shadow execution.
        Negative values (reduced drawdown) always pass.
    minimum_replay_samples : int
        Minimum number of replay / backtest samples.
    minimum_regime_coverage : float
        Minimum fraction of market regimes tested.
    """

    def __init__(
        self,
        minimum_predictive_lift: float = 0.02,
        minimum_calibration: float = 0.70,
        maximum_false_veto_rate: float = 0.30,
        maximum_drawdown_increase: float = 0.05,
        minimum_replay_samples: int = 50,
        minimum_regime_coverage: float = 0.60,
    ):
        # --- thresholds (mutable via set_thresholds) ---
        self.minimum_predictive_lift = minimum_predictive_lift
        self.minimum_calibration = minimum_calibration
        self.maximum_false_veto_rate = maximum_false_veto_rate
        self.maximum_drawdown_increase = maximum_drawdown_increase
        self.minimum_replay_samples = minimum_replay_samples
        self.minimum_regime_coverage = minimum_regime_coverage

        # --- internal counters ---
        self._evaluations = 0
        self._promoted = 0
        self._rejected = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, module_name: str, scores: dict, metrics: dict) -> dict:
        """Run promotion evaluation for a single module.

        Parameters
        ----------
        module_name : str
            Observer identifier (e.g. "D1", "D2", … "D7").
        scores : dict
            Result from E3 ``PromotionScorer.compute()``.  Must contain
            ``"module"``, ``"composite"``, optionally ``"dimensions"``.
        metrics : dict
            Performance / evidence metrics dict with keys:
            ``predictive_lift_delta``, ``calibration_ece``,
            ``false_veto_rate``, ``drawdown_change``,
            ``replay_samples``, ``regime_coverage``.

        Returns
        -------
        dict
            {
                "module": str,
                "promoted": bool,
                "composite_score": float,
                "checks": { ... },
                "decision": str,
                "composite_threshold": float,
            }
        """
        self._evaluations += 1

        # --- extract composite ---
        composite_score = scores.get("composite", 0.0) if isinstance(scores, dict) else 0.0
        composite_threshold = 0.50

        # --- run each check ---
        checks = {}
        checks["predictive_lift"] = self._check_lift(metrics)
        checks["calibration"] = self._check_calibration(metrics)
        checks["false_veto_rate"] = self._check_veto(metrics)
        checks["drawdown"] = self._check_drawdown(metrics)
        checks["sample_size"] = self._check_samples(metrics)
        checks["regime_coverage"] = self._check_regime_coverage(metrics)

        # --- composite threshold gate ---
        composite_pass = composite_score >= composite_threshold
        if not composite_pass:
            checks["composite"] = {
                "pass": False,
                "value": composite_score,
                "threshold": composite_threshold,
                "detail": (
                    f"Composite score {composite_score:.4f} < {composite_threshold:.2f}; "
                    f"minimum composite required for promotion."
                ),
            }

        # --- all checks must pass ---
        all_pass = all(c["pass"] for c in checks.values())
        promoted = all_pass

        # --- decision string ---
        if promoted:
            decision = "PROMOTED"
            self._promoted += 1
        else:
            reasons = []
            for key, check in checks.items():
                if not check["pass"]:
                    reasons.append(f"{key}: {check['detail']}")
            decision = "REJECTED: " + "; ".join(reasons)
            self._rejected += 1

        return {
            "module": module_name,
            "promoted": promoted,
            "composite_score": composite_score,
            "checks": checks,
            "decision": decision,
            "composite_threshold": composite_threshold,
        }

    def set_thresholds(self, **kwargs) -> None:
        """Override any threshold after construction.

        Acceptable keyword arguments match the constructor parameter names.

        Examples
        --------
        >>> policy.set_thresholds(minimum_predictive_lift=0.05,
        ...                       minimum_replay_samples=100)
        """
        valid_keys = {
            "minimum_predictive_lift",
            "minimum_calibration",
            "maximum_false_veto_rate",
            "maximum_drawdown_increase",
            "minimum_replay_samples",
            "minimum_regime_coverage",
        }
        for key, value in kwargs.items():
            if key not in valid_keys:
                raise ValueError(
                    f"Unknown threshold '{key}'. Valid keys: {sorted(valid_keys)}"
                )
            setattr(self, key, value)

    def explain(self, module_name: str, scores: dict, metrics: dict) -> str:
        """Return a human-readable promotion report."""
        result = self.evaluate(module_name, scores, metrics)
        # Reuse evaluate; build readable explanation from result.
        lines = [
            f"=== Promotion Policy Report: {result['module']} ===",
            f"  Decision       : {result['decision']}",
            f"  Composite Score: {result['composite_score']:.4f}  "
            f"(threshold: {result['composite_threshold']:.2f})",
            "",
            "  Check Results:",
        ]

        check_labels = {
            "predictive_lift": "Predictive Lift",
            "calibration": "Calibration",
            "false_veto_rate": "False Veto Rate",
            "drawdown": "Drawdown Change",
            "sample_size": "Sample Size",
            "regime_coverage": "Regime Coverage",
        }

        for key, info in result["checks"].items():
            label = check_labels.get(key, key.replace("_", " ").title())
            status = "PASS" if info["pass"] else "FAIL"
            lines.append(
                f"    {label:20s}  {status:4s}  "
                f"value={info['value']}  "
                f"threshold={info['threshold']}  "
                f"detail={info['detail']}"
            )

        lines.append("")
        lines.append(f"  Verdict: {result['decision']}")
        return "\n".join(lines)

    def batch_evaluate(self, results: dict) -> dict:
        """Evaluate multiple modules in batch.

        Parameters
        ----------
        results : dict
            ``{module_name: (scores, metrics)}`` where each value is a
            2-tuple of the ``scores`` dict and ``metrics`` dict.

        Returns
        -------
        dict
            ``{module_name: evaluate_result}``
        """
        output = {}
        for module_name, (scores, metrics) in results.items():
            output[module_name] = self.evaluate(module_name, scores, metrics)
        return output

    def stats(self) -> dict:
        """Return aggregate statistics across all evaluations."""
        total = self._evaluations
        return {
            "evaluations": total,
            "promoted": self._promoted,
            "rejected": self._rejected,
            "promotion_rate": round(self._promoted / total, 6) if total > 0 else 0.0,
        }

    # ------------------------------------------------------------------
    # Internal check helpers
    # ------------------------------------------------------------------

    def _check_lift(self, metrics: dict) -> dict:
        value = metrics.get("predictive_lift_delta", 0.0)
        threshold = self.minimum_predictive_lift
        passed = value >= threshold
        return {
            "pass": passed,
            "value": value,
            "threshold": threshold,
            "detail": (
                f"predictive_lift_delta={value:.4f} >= {threshold:.4f}"
                if passed
                else f"predictive_lift_delta={value:.4f} < {threshold:.4f}"
            ),
        }

    def _check_calibration(self, metrics: dict) -> dict:
        ece = metrics.get("calibration_ece", 1.0)
        # calibration quality = 1 - ece; requirement: 1 - ece >= minimum_calibration
        # equivalently: ece <= 1.0 - minimum_calibration
        threshold = 1.0 - self.minimum_calibration
        passed = ece <= threshold
        cal_quality = 1.0 - ece
        min_cal = self.minimum_calibration
        return {
            "pass": passed,
            "value": ece,
            "threshold": threshold,
            "detail": (
                f"calibration_quality={cal_quality:.4f} >= {min_cal:.4f}  "
                f"(ECE={ece:.4f} <= {threshold:.4f})"
                if passed
                else f"calibration_quality={cal_quality:.4f} < {min_cal:.4f}  "
                f"(ECE={ece:.4f} > {threshold:.4f})"
            ),
        }

    def _check_veto(self, metrics: dict) -> dict:
        value = metrics.get("false_veto_rate", 1.0)
        threshold = self.maximum_false_veto_rate
        passed = value <= threshold
        return {
            "pass": passed,
            "value": value,
            "threshold": threshold,
            "detail": (
                f"false_veto_rate={value:.4f} <= {threshold:.4f}"
                if passed
                else f"false_veto_rate={value:.4f} > {threshold:.4f}"
            ),
        }

    def _check_drawdown(self, metrics: dict) -> dict:
        value = metrics.get("drawdown_change", 1.0)
        threshold = self.maximum_drawdown_increase
        passed = value <= threshold
        return {
            "pass": passed,
            "value": value,
            "threshold": threshold,
            "detail": (
                f"drawdown_change={value:.4f} <= {threshold:.4f}"
                if passed
                else f"drawdown_change={value:.4f} > {threshold:.4f}  "
                f"(positive = increased drawdown)"
            ),
        }

    def _check_samples(self, metrics: dict) -> dict:
        value = metrics.get("replay_samples", 0)
        threshold = self.minimum_replay_samples
        passed = value >= threshold
        return {
            "pass": passed,
            "value": value,
            "threshold": threshold,
            "detail": (
                f"replay_samples={value} >= {threshold}"
                if passed
                else f"replay_samples={value} < {threshold}"
            ),
        }

    def _check_regime_coverage(self, metrics: dict) -> dict:
        value = metrics.get("regime_coverage", 0.0)
        threshold = self.minimum_regime_coverage
        passed = value >= threshold
        return {
            "pass": passed,
            "value": value,
            "threshold": threshold,
            "detail": (
                f"regime_coverage={value:.4f} >= {threshold:.4f}"
                if passed
                else f"regime_coverage={value:.4f} < {threshold:.4f}"
            ),
        }
