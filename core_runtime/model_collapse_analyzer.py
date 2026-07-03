"""
Model Collapse Analyzer — Classify the root cause of OSS model collapse.

Determines whether a collapsed model suffers from a data issue, training
issue, feature collapse, or fundamental architecture collapse. This
pinpoints what subsystem needs to be fixed.

Classification types
--------------------
DATA_ISSUE
    Insufficient training data, wrong data distribution, stale data.
    Indicators: low training record count, high data staleness, low
    coverage of current regime.

TRAINING_ISSUE
    Model was trained incorrectly, hyperparameters wrong, training
    didn't converge.
    Indicators: training completed but model outputs degenerate, high
    training loss, flat weights.

FEATURE_COLLAPSE
    Input features lack discriminative power for current market conditions.
    Indicators: features produce same values regardless of market state,
    low feature variance.

ARCHITECTURE_COLLAPSE
    OSS model architecture fundamentally cannot represent the decision
    boundary.
    Indicators: model always outputs p_cont ≈ 0.50 regardless of input
    variation, structural 50/50 default.

Usage::

    from proxima_x.core_runtime.model_collapse_analyzer import (
        ModelCollapseAnalyzer,
    )

    analyzer = ModelCollapseAnalyzer()
    analyzer.feed_diagnostics(
        surface_diagnostic_report=...,
        duality_report=...,
        entropy_report=...,
        training_metadata=None,
    )
    result = analyzer.analyze()
    print(analyzer.get_analysis_summary())
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API — explicitly listed so ``from module import *`` is safe.
# ---------------------------------------------------------------------------
__all__ = [
    "ModelCollapseAnalyzer",
    "DATA_ISSUE",
    "TRAINING_ISSUE",
    "FEATURE_COLLAPSE",
    "ARCHITECTURE_COLLAPSE",
]

# ---------------------------------------------------------------------------
# Collapse type constants
# ---------------------------------------------------------------------------

DATA_ISSUE = "DATA_ISSUE"
TRAINING_ISSUE = "TRAINING_ISSUE"
FEATURE_COLLAPSE = "FEATURE_COLLAPSE"
ARCHITECTURE_COLLAPSE = "ARCHITECTURE_COLLAPSE"

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp *value* to the inclusive interval [*lo*, *hi*]."""
    return max(lo, min(value, hi))


def _is_flat(sequence: List[float], relative_tolerance: float = 1e-6) -> bool:
    """Return ``True`` if all values in *sequence* are nearly identical."""
    if len(sequence) < 2:
        return True
    mn = min(sequence)
    mx = max(sequence)
    if mx == 0.0 and mn == 0.0:
        return True
    span = abs(mx - mn)
    # If the span is below a relative tolerance or an absolute epsilon,
    # consider the sequence flat.
    return span < (relative_tolerance * max(abs(mx), abs(mn)) + 1e-12)


# ===================================================================
# Internal implementation class
# ===================================================================


class _ModelCollapseAnalyzer:
    """Classify OSS model collapse using fed diagnostic data.

    This class should **not** be instantiated directly.  Use the
    module-level :func:`ModelCollapseAnalyzer` factory instead.
    """

    # ------------------------------------------------------------------
    # Decision thresholds
    # ------------------------------------------------------------------
    MIN_TRAINING_RECORDS: int = 10
    """If training record count is below this threshold, flag DATA_ISSUE."""

    P_CONT_CENTER: float = 0.50
    """The value around which p_cont is considered "centered" when collapsed."""

    P_CONT_TOLERANCE: float = 0.02
    """If p_cont deviates less than this from 0.50 across inputs, flag
    ARCHITECTURE_COLLAPSE."""

    MIN_FEATURE_VARIANCE: float = 1e-4
    """If feature variance is below this threshold, flag FEATURE_COLLAPSE."""

    MAX_TRAINING_LOSS_THRESHOLD: float = 2.0
    """If final training loss exceeds this, flag TRAINING_ISSUE."""

    HIGH_DATA_STALENESS_THRESHOLD: float = 7 * 24 * 3600  # 7 days in seconds
    """If data staleness exceeds this (seconds), contribute to DATA_ISSUE."""

    def __init__(self, instance_id: str = "default") -> None:
        self._instance_id: str = instance_id

        # Diagnostic data stores (populated via feed_diagnostics)
        self._surface_diagnostic: Optional[Dict[str, Any]] = None
        self._duality_report: Optional[Dict[str, Any]] = None
        self._entropy_report: Optional[Dict[str, Any]] = None
        self._training_metadata: Optional[Dict[str, Any]] = None

        # Cached analysis result
        self._last_analysis: Optional[Dict[str, Any]] = None

        # Whether any data has been ingested
        self._has_data: bool = False

        logger.info(
            "[MODEL_COLLAPSE_ANALYZER] Instance '%s' initialised.",
            instance_id,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def feed_diagnostics(
        self,
        surface_diagnostic_report: Optional[Dict[str, Any]] = None,
        duality_report: Optional[Dict[str, Any]] = None,
        entropy_report: Optional[Dict[str, Any]] = None,
        training_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Feed all available diagnostic data into the analyzer.

        Parameters
        ----------
        surface_diagnostic_report : dict, optional
            OSS surface diagnostic containing keys such as
            ``training_record_count``, ``p_cont_outputs`` (list of floats),
            ``data_staleness_seconds``, ``regime_coverage``, etc.

        duality_report : dict, optional
            Duality analysis report.  May contain keys like
            ``feature_variances``, ``feature_means``, etc.

        entropy_report : dict, optional
            Entropy / decision diversity report.  May contain keys like
            ``decision_entropy``, ``output_diversity``, etc.

        training_metadata : dict, optional
            Training metadata such as ``training_completed``, ``final_loss``,
            ``hyperparameters``, ``weight_stats``, ``training_duration``, etc.
        """
        if surface_diagnostic_report is not None:
            self._surface_diagnostic = dict(surface_diagnostic_report)
        if duality_report is not None:
            self._duality_report = dict(duality_report)
        if entropy_report is not None:
            self._entropy_report = dict(entropy_report)
        if training_metadata is not None:
            self._training_metadata = dict(training_metadata)

        # Invalidate cached analysis
        self._last_analysis = None
        self._has_data = True

        logger.debug(
            "[MODEL_COLLAPSE_ANALYZER] Diagnostics fed — surface=%s, "
            "duality=%s, entropy=%s, training=%s",
            surface_diagnostic_report is not None,
            duality_report is not None,
            entropy_report is not None,
            training_metadata is not None,
        )

    # ------------------------------------------------------------------

    def analyze(self) -> Dict[str, Any]:
        """Run classification logic and produce a collapse analysis.

        Uses the decision tree described in the module docstring.

        Returns
        -------
        dict with keys:
            primary_cause : str
                One of ``DATA_ISSUE``, ``TRAINING_ISSUE``,
                ``FEATURE_COLLAPSE``, ``ARCHITECTURE_COLLAPSE``.
            confidence : float
                Confidence in the classification (0.0 – 1.0).
            contributing_factors : list of str
                Human-readable factors that contributed to the decision.
            evidence : dict
                Key indicators that led to this classification.
            recommendation : str
                Suggested next action.
            alternative_hypotheses : list of str
                Other possible causes that were not ruled out.
        """
        # Collect evidence from all available reports
        evidence: Dict[str, Any] = self._gather_evidence()

        # Extract commonly-used evidence fields
        training_record_count = evidence.get("training_record_count")
        p_cont_outputs: List[float] = evidence.get("p_cont_outputs", [])
        training_completed: bool = evidence.get("training_completed", False) is True
        training_was_flawed: bool = evidence.get("training_was_flawed", False) is True
        final_loss: Optional[float] = evidence.get("final_loss")

        # Skip classification if we have no diagnostic data at all
        if not self._has_data or not any(
            [p_cont_outputs, training_record_count is not None, final_loss is not None]
        ):
            self._last_analysis = self._build_result(
                primary_cause=DATA_ISSUE,
                confidence=0.50,
                contributing_factors=[
                    "Insufficient diagnostic data to classify confidently. "
                    "Defaulting to DATA_ISSUE as the most common cause."
                ],
                evidence=evidence,
                recommendation=(
                    "Collect more diagnostic data: training metadata, "
                    "surface diagnostics, and feature variance reports."
                ),
                alternative_hypotheses=[
                    "TRAINING_ISSUE — cannot rule out without training metadata.",
                    "FEATURE_COLLAPSE — cannot rule out without feature variance data.",
                    "ARCHITECTURE_COLLAPSE — possible if outputs are structurally degenerate.",
                ],
            )
            return self._last_analysis

        # Derived flags
        outputs_flat = _is_flat(p_cont_outputs) if p_cont_outputs else False
        outputs_near_50 = self._outputs_near_50(p_cont_outputs) if p_cont_outputs else False
        outputs_degenerate = outputs_flat or outputs_near_50
        features_vary = self._features_vary(evidence)

        # Run decision-tree classification
        primary_cause: str = DATA_ISSUE
        confidence: float = 0.0
        contributing_factors: List[str] = []
        alternative_hypotheses: List[str] = []

        # ═══════════════════════════════════════════════════════════════
        # Step 1 — Low training data → DATA_ISSUE
        # ═══════════════════════════════════════════════════════════════
        if training_record_count is not None and training_record_count < self.MIN_TRAINING_RECORDS:
            primary_cause = DATA_ISSUE
            confidence = _clamp(
                0.5 + 0.5 * (1.0 - training_record_count / self.MIN_TRAINING_RECORDS)
            )
            contributing_factors.append(
                f"Training record count ({training_record_count}) is below "
                f"the minimum threshold ({self.MIN_TRAINING_RECORDS})."
            )
            if evidence.get("data_staleness_seconds", 0) > self.HIGH_DATA_STALENESS_THRESHOLD:
                contributing_factors.append(
                    "Data is stale (>{:.0f}s old).".format(self.HIGH_DATA_STALENESS_THRESHOLD)
                )
            if evidence.get("regime_coverage", 1.0) < 0.3:
                contributing_factors.append(
                    f"Regime coverage ({evidence['regime_coverage']:.2f}) is low."
                )

            recommendation = (
                "Collect more training data — at least "
                f"{self.MIN_TRAINING_RECORDS} records. "
                "Check data freshness and regime coverage."
            )

            if training_completed:
                alternative_hypotheses.append(
                    "Training did complete, so a TRAINING_ISSUE is possible "
                    "if hyperparameters were inappropriate for the data."
                )

            self._last_analysis = self._build_result(
                primary_cause=primary_cause,
                confidence=confidence,
                contributing_factors=contributing_factors,
                evidence=evidence,
                recommendation=recommendation,
                alternative_hypotheses=alternative_hypotheses,
            )
            return self._last_analysis

        # ═══════════════════════════════════════════════════════════════
        # Step 5 (early) — Explicitly bad training params → TRAINING_ISSUE
        # ═══════════════════════════════════════════════════════════════
        if training_was_flawed:
            primary_cause = TRAINING_ISSUE
            confidence = 0.80
            contributing_factors.append(
                "Training was explicitly performed with bad parameters "
                "or did not complete successfully."
            )
            if final_loss is not None and final_loss > self.MAX_TRAINING_LOSS_THRESHOLD:
                contributing_factors.append(
                    f"Final training loss ({final_loss:.2f}) exceeds "
                    f"threshold ({self.MAX_TRAINING_LOSS_THRESHOLD})."
                )
            recommendation = (
                "Fix training configuration and retrain. Verify "
                "hyperparameters, data splits, and convergence criteria."
            )
            alternative_hypotheses.append(
                "DATA_ISSUE — if the training data itself is flawed, "
                "no amount of retraining will help."
            )
            if not features_vary:
                alternative_hypotheses.append(
                    "FEATURE_COLLAPSE — flawed training may have damaged "
                    "the feature representation."
                )

            self._last_analysis = self._build_result(
                primary_cause=primary_cause,
                confidence=confidence,
                contributing_factors=contributing_factors,
                evidence=evidence,
                recommendation=recommendation,
                alternative_hypotheses=alternative_hypotheses,
            )
            return self._last_analysis

        # ═══════════════════════════════════════════════════════════════
        # Steps 2-4 — Degenerate outputs → check features
        # ═══════════════════════════════════════════════════════════════
        if outputs_degenerate:
            if not features_vary:
                # ── Step 4: Outputs degenerate AND features don't vary
                #            → FEATURE_COLLAPSE
                primary_cause = FEATURE_COLLAPSE
                confidence = 0.85
                contributing_factors.append(
                    "Model outputs are degenerate (flat or stuck at ~0.50) "
                    "AND input features lack variance — they produce nearly "
                    "the same values regardless of market state."
                )
                contributing_factors.append(
                    "Feature variances are below the minimum threshold "
                    f"({self.MIN_FEATURE_VARIANCE})."
                )
                recommendation = (
                    "Investigate feature pipeline. Features lack "
                    "discriminative power for current market conditions. "
                    "Consider feature engineering, regime-adaptive features, "
                    "or adding market-state indicators."
                )
                alternative_hypotheses.append(
                    "ARCHITECTURE_COLLAPSE — if features were healthy, "
                    "the architecture might still be the root cause."
                )
                if training_was_flawed:
                    alternative_hypotheses.append(
                        "TRAINING_ISSUE — training may have used bad "
                        "hyperparameters that damaged the features."
                    )

            elif outputs_near_50:
                # ── Step 3: Features vary but outputs near 0.50
                #            → ARCHITECTURE_COLLAPSE
                primary_cause = ARCHITECTURE_COLLAPSE
                confidence = 0.90
                contributing_factors.append(
                    "Input features vary meaningfully across samples, "
                    "but the model always outputs p_cont "
                    f"≈ {self.P_CONT_CENTER:.2f} regardless of input."
                )
                contributing_factors.append(
                    "The model architecture cannot project feature "
                    "variation into a discriminative output; it defaults "
                    "to a structural 50/50 decision."
                )
                recommendation = (
                    "The OSS model architecture cannot represent the "
                    "required decision boundary. Consider a more "
                    "expressive architecture (deeper network, different "
                    "activation functions, or a non-linear model)."
                )
                alternative_hypotheses.append(
                    "FEATURE_COLLAPSE — if the feature variation is "
                    "not economically meaningful, the pipeline may "
                    "still need improvement."
                )
                if final_loss is not None and final_loss > self.MAX_TRAINING_LOSS_THRESHOLD:
                    alternative_hypotheses.append(
                        "TRAINING_ISSUE — high final loss suggests "
                        "training may not have converged properly."
                    )

            else:
                # Outputs are flat (identical values) but not near 0.50.
                # This is unusual — likely a training problem.
                primary_cause = TRAINING_ISSUE
                confidence = 0.75
                contributing_factors.append(
                    "Model outputs are flat (all identical values) but "
                    "not centered at 0.50, and features do vary. This "
                    "suggests training degeneracy rather than architecture."
                )
                if final_loss is not None and final_loss > self.MAX_TRAINING_LOSS_THRESHOLD:
                    contributing_factors.append(
                        f"Final training loss ({final_loss:.2f}) exceeds "
                        f"threshold ({self.MAX_TRAINING_LOSS_THRESHOLD})."
                    )
                recommendation = (
                    "Investigate training process. Flat outputs with "
                    "varying features suggest weights collapsed or "
                    "training diverged. Check learning rate, gradients, "
                    "and weight initialisation."
                )
                alternative_hypotheses.append(
                    "ARCHITECTURE_COLLAPSE — the architecture may have "
                    "a degeneracy that maps all inputs to a constant "
                    "output value."
                )
                if not features_vary:
                    alternative_hypotheses.append(
                        "FEATURE_COLLAPSE — flat outputs could result "
                        "from features that collapsed to near-constant values."
                    )

            self._last_analysis = self._build_result(
                primary_cause=primary_cause,
                confidence=confidence,
                contributing_factors=contributing_factors,
                evidence=evidence,
                recommendation=recommendation,
                alternative_hypotheses=alternative_hypotheses,
            )
            return self._last_analysis

        # ═══════════════════════════════════════════════════════════════
        # Training completed, outputs are not degenerate
        # ═══════════════════════════════════════════════════════════════
        if training_completed:
            # Check for high loss → TRAINING_ISSUE
            if final_loss is not None and final_loss > self.MAX_TRAINING_LOSS_THRESHOLD:
                primary_cause = TRAINING_ISSUE
                confidence = _clamp(
                    0.5 + 0.5 * (final_loss - self.MAX_TRAINING_LOSS_THRESHOLD)
                    / final_loss
                )
                contributing_factors.append(
                    f"Final training loss ({final_loss:.2f}) exceeds "
                    f"threshold ({self.MAX_TRAINING_LOSS_THRESHOLD})."
                )
                recommendation = (
                    "Training completed but with high loss. Review "
                    "hyperparameters, learning rate schedule, and "
                    "training data quality."
                )
                alternative_hypotheses.append(
                    "DATA_ISSUE — sufficient records exist but distribution "
                    "may be wrong, causing high loss despite convergence."
                )

            elif evidence.get("regime_coverage", 1.0) < 0.3:
                primary_cause = DATA_ISSUE
                confidence = 0.70
                contributing_factors.append(
                    "Regime coverage is low despite adequate training."
                )
                recommendation = (
                    "Expand training data to cover a broader set of "
                    "market regimes."
                )
                alternative_hypotheses.append(
                    "TRAINING_ISSUE — model may have overfit to the "
                    "narrow regime it was trained on."
                )

            else:
                # No strong collapse signal
                primary_cause = DATA_ISSUE
                confidence = 0.35
                contributing_factors.append(
                    "No strong collapse signal detected. Model outputs "
                    "vary and training completed normally."
                )
                recommendation = (
                    "Monitor closely. No immediate collapse detected, "
                    "but periodic re-evaluation is recommended."
                )
                alternative_hypotheses = [
                    "TRAINING_ISSUE — subtle underfitting may not "
                    "manifest as degenerate outputs.",
                    "FEATURE_COLLAPSE — features may degrade gradually "
                    "before complete collapse.",
                ]

        # ═══════════════════════════════════════════════════════════════
        # No training completion info and no degenerate outputs
        # ═══════════════════════════════════════════════════════════════
        else:
            if p_cont_outputs and outputs_near_50:
                primary_cause = ARCHITECTURE_COLLAPSE
                confidence = 0.65
                contributing_factors.append(
                    "Model outputs are consistently near 0.50, suggesting "
                    "the architecture defaults to a 50/50 decision. "
                    "No training metadata available to confirm."
                )
                recommendation = (
                    "Audit model architecture. Structural 50/50 outputs "
                    "indicate the model cannot differentiate inputs."
                )
                alternative_hypotheses.append(
                    "FEATURE_COLLAPSE — if feature pipeline is broken, "
                    "even a good architecture would produce null outputs."
                )

            else:
                primary_cause = DATA_ISSUE
                confidence = 0.50
                contributing_factors.append(
                    "Insufficient diagnostic data to classify confidently. "
                    "Defaulting to DATA_ISSUE as the most common cause."
                )
                recommendation = (
                    "Collect more diagnostic data: training metadata, "
                    "surface diagnostics, and feature variance reports."
                )
                alternative_hypotheses = [
                    "TRAINING_ISSUE — cannot rule out without training metadata.",
                    "FEATURE_COLLAPSE — cannot rule out without feature variance data.",
                    "ARCHITECTURE_COLLAPSE — possible if outputs are structurally degenerate.",
                ]

        self._last_analysis = self._build_result(
            primary_cause=primary_cause,
            confidence=confidence,
            contributing_factors=contributing_factors,
            evidence=evidence,
            recommendation=recommendation,
            alternative_hypotheses=alternative_hypotheses,
        )
        return self._last_analysis

    # ------------------------------------------------------------------

    def get_analysis_summary(self) -> str:
        """Return a one-line verdict string.

        Returns
        -------
        str
            Short human-readable verdict.  Returns a placeholder message
            if no analysis has been run yet.
        """
        if self._last_analysis is None:
            return "No analysis performed yet. Call .analyze() first."

        cause = self._last_analysis["primary_cause"]
        conf = self._last_analysis["confidence"]
        labels = {
            DATA_ISSUE: "Data Issue",
            TRAINING_ISSUE: "Training Issue",
            FEATURE_COLLAPSE: "Feature Collapse",
            ARCHITECTURE_COLLAPSE: "Architecture Collapse",
        }
        label = labels.get(cause, cause)
        return (
            f"Model collapse classified as {label} "
            f"(confidence={conf:.2f})"
        )

    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all stored diagnostic data and cached analysis."""
        self._surface_diagnostic = None
        self._duality_report = None
        self._entropy_report = None
        self._training_metadata = None
        self._last_analysis = None
        self._has_data = False
        logger.info(
            "[MODEL_COLLAPSE_ANALYZER] Instance '%s' reset.",
            self._instance_id,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _gather_evidence(self) -> Dict[str, Any]:
        """Collate relevant indicators from all fed reports into a flat dict.

        Returns
        -------
        dict
            Normalised evidence dictionary used by the decision tree.
        """
        evidence: Dict[str, Any] = {}

        # --- From surface diagnostic ---
        sd = self._surface_diagnostic or {}
        evidence["training_record_count"] = sd.get("training_record_count")
        evidence["p_cont_outputs"] = sd.get("p_cont_outputs", [])
        evidence["data_staleness_seconds"] = sd.get("data_staleness_seconds", 0)
        evidence["regime_coverage"] = sd.get("regime_coverage", 1.0)
        evidence["model_outputs"] = sd.get("model_outputs", [])

        # --- From duality report (feature diagnostics) ---
        dr = self._duality_report or {}
        evidence["feature_variances"] = dr.get("feature_variances", [])
        evidence["feature_means"] = dr.get("feature_means", [])
        evidence["feature_identities"] = dr.get("feature_identities", [])

        # --- From entropy report ---
        er = self._entropy_report or {}
        evidence["decision_entropy"] = er.get("decision_entropy")
        evidence["output_diversity"] = er.get("output_diversity")

        # --- From training metadata ---
        tm = self._training_metadata or {}
        evidence["training_completed"] = tm.get("training_completed", False)
        evidence["final_loss"] = tm.get("final_loss")
        evidence["hyperparameters"] = tm.get("hyperparameters")
        evidence["weight_stats"] = tm.get("weight_stats", {})
        evidence["training_duration"] = tm.get("training_duration")
        evidence["training_was_flawed"] = tm.get(
            "training_was_flawed", False
        )

        return evidence

    # ------------------------------------------------------------------

    def _outputs_near_50(self, p_cont_outputs: List[float]) -> bool:
        """Check whether the model outputs are all near 0.50."""
        if not p_cont_outputs:
            return False
        deviations = [
            abs(o - self.P_CONT_CENTER) for o in p_cont_outputs
        ]
        max_deviation = max(deviations)
        return max_deviation < self.P_CONT_TOLERANCE

    # ------------------------------------------------------------------

    def _features_vary(self, evidence: Dict[str, Any]) -> bool:
        """Determine whether input features show meaningful variation.

        Examines feature variances from the duality report.  Falls back
        to checking feature means if variances are not present.
        """
        variances = evidence.get("feature_variances", [])
        if variances:
            # If any feature has variance above the minimum threshold,
            # then features vary.
            return any(v > self.MIN_FEATURE_VARIANCE for v in variances)

        # Fallback: check feature means for variation
        means = evidence.get("feature_means", [])
        if len(means) >= 2:
            return not _is_flat(means)

        # If we have p_cont outputs but no feature data, assume features
        # vary (we'll determine collapse from output behaviour).
        if evidence.get("p_cont_outputs"):
            return True

        # No information — conservative assumption: features don't vary.
        return False

    # ------------------------------------------------------------------

    def _build_result(
        self,
        primary_cause: str,
        confidence: float,
        contributing_factors: List[str],
        evidence: Dict[str, Any],
        recommendation: str,
        alternative_hypotheses: List[str],
    ) -> Dict[str, Any]:
        """Assemble the analysis result dict."""
        return {
            "primary_cause": primary_cause,
            "confidence": round(confidence, 4),
            "contributing_factors": contributing_factors,
            "evidence": evidence,
            "recommendation": recommendation,
            "alternative_hypotheses": alternative_hypotheses,
        }

    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "fed" if self._has_data else "empty"
        return (
            f"<_ModelCollapseAnalyzer id='{self._instance_id}' "
            f"status={status}>"
        )


# ===================================================================
# Singleton accessor pattern
# ===================================================================

_instances: Dict[str, _ModelCollapseAnalyzer] = {}


def ModelCollapseAnalyzer(instance_id: str = "default") -> _ModelCollapseAnalyzer:
    """Return a shared ``_ModelCollapseAnalyzer`` instance.

    This is the **only** way to obtain a model collapse analyzer.
    It implements a simple registry of singletons keyed by
    *instance_id*.

    Parameters
    ----------
    instance_id : str
        Identifier for the analyzer instance.  Use ``"default"`` (or
        omit) for the global singleton.  Pass a unique string to
        create/maintain an independent analyzer for a specific
        subsystem.

    Returns
    -------
    _ModelCollapseAnalyzer
        The shared instance for the given *instance_id*.

    Usage::

        from proxima_x.core_runtime.model_collapse_analyzer import (
            ModelCollapseAnalyzer,
        )

        analyzer = ModelCollapseAnalyzer()
        analyzer.feed_diagnostics(...)
        result = analyzer.analyze()
    """
    if instance_id not in _instances:
        _instances[instance_id] = _ModelCollapseAnalyzer(
            instance_id=instance_id,
        )
    return _instances[instance_id]


# ===================================================================
# Quick self-test (only when run directly)
# ===================================================================

if __name__ == "__main__":
    import pprint

    # ------------------------------------------------------------------
    # Scenario 1: DATA_ISSUE — low training record count
    # ------------------------------------------------------------------
    print("=" * 72)
    print("Scenario 1: DATA_ISSUE — low training record count")
    print("=" * 72)

    analyzer_1 = ModelCollapseAnalyzer("test_data_issue")
    analyzer_1.feed_diagnostics(
        surface_diagnostic_report={
            "training_record_count": 3,
            "p_cont_outputs": [0.48, 0.52, 0.49],
            "data_staleness_seconds": 10 * 24 * 3600,  # 10 days
            "regime_coverage": 0.15,
            "model_outputs": [0.48, 0.52, 0.49],
        },
        duality_report={
            "feature_variances": [0.01, 0.02, 0.005],
            "feature_means": [0.3, 0.5, 0.7],
        },
        entropy_report={
            "decision_entropy": 0.6,
            "output_diversity": 0.4,
        },
        training_metadata={
            "training_completed": True,
            "final_loss": 0.5,
            "hyperparameters": {"lr": 0.001},
        },
    )
    result_1 = analyzer_1.analyze()
    print(getattr(analyzer_1, "get_analysis_summary", lambda: analyzer_1.get_analysis_summary())())
    pprint.pprint(result_1, width=120, sort_dicts=False, depth=3)
    print()

    # ------------------------------------------------------------------
    # Scenario 2: ARCHITECTURE_COLLAPSE — p_cont always ~0.50
    # ------------------------------------------------------------------
    print("=" * 72)
    print("Scenario 2: ARCHITECTURE_COLLAPSE — p_cont always ~0.50")
    print("=" * 72)

    analyzer_2 = ModelCollapseAnalyzer("test_arch_collapse")
    analyzer_2.feed_diagnostics(
        surface_diagnostic_report={
            "training_record_count": 500,
            "p_cont_outputs": [0.502, 0.498, 0.501, 0.499, 0.500],
            "data_staleness_seconds": 3600,
            "regime_coverage": 0.85,
            "model_outputs": [0.502, 0.498, 0.501, 0.499, 0.500],
        },
        duality_report={
            "feature_variances": [0.15, 0.22, 0.08, 0.31],
            "feature_means": [0.4, 0.6, 0.3, 0.7],
        },
        entropy_report={
            "decision_entropy": 0.05,
            "output_diversity": 0.02,
        },
        training_metadata={
            "training_completed": True,
            "final_loss": 0.45,
            "hyperparameters": {"lr": 0.001, "layers": 2, "units": 16},
            "weight_stats": {"mean": 0.0, "std": 0.01},
        },
    )
    result_2 = analyzer_2.analyze()
    print(analyzer_2.get_analysis_summary())
    pprint.pprint(result_2, width=120, sort_dicts=False, depth=3)
    print()

    # ------------------------------------------------------------------
    # Scenario 3: FEATURE_COLLAPSE — features lack variance
    # ------------------------------------------------------------------
    print("=" * 72)
    print("Scenario 3: FEATURE_COLLAPSE — features lack variance")
    print("=" * 72)

    analyzer_3 = ModelCollapseAnalyzer("test_feature_collapse")
    analyzer_3.feed_diagnostics(
        surface_diagnostic_report={
            "training_record_count": 200,
            "p_cont_outputs": [0.45, 0.45, 0.45, 0.45, 0.45],
            "data_staleness_seconds": 7200,
            "regime_coverage": 0.60,
            "model_outputs": [0.45, 0.45, 0.45, 0.45, 0.45],
        },
        duality_report={
            "feature_variances": [1e-6, 2e-7, 5e-8],
            "feature_means": [0.5, 0.5, 0.5],
        },
        entropy_report={
            "decision_entropy": 0.1,
            "output_diversity": 0.05,
        },
        training_metadata={
            "training_completed": True,
            "final_loss": 0.6,
            "hyperparameters": {"lr": 0.001},
        },
    )
    result_3 = analyzer_3.analyze()
    print(analyzer_3.get_analysis_summary())
    pprint.pprint(result_3, width=120, sort_dicts=False, depth=3)
    print()

    # ------------------------------------------------------------------
    # Scenario 4: TRAINING_ISSUE — bad hyperparameters, high loss
    # ------------------------------------------------------------------
    print("=" * 72)
    print("Scenario 4: TRAINING_ISSUE — bad hyperparameters, high loss")
    print("=" * 72)

    analyzer_4 = ModelCollapseAnalyzer("test_training_issue")
    analyzer_4.feed_diagnostics(
        surface_diagnostic_report={
            "training_record_count": 100,
            "p_cont_outputs": [0.30, 0.32, 0.29, 0.31, 0.33],
            "data_staleness_seconds": 3600,
            "regime_coverage": 0.70,
            "model_outputs": [0.30, 0.32, 0.29, 0.31, 0.33],
        },
        duality_report={
            "feature_variances": [0.10, 0.15, 0.08],
            "feature_means": [0.3, 0.6, 0.5],
        },
        entropy_report={
            "decision_entropy": 0.3,
            "output_diversity": 0.25,
        },
        training_metadata={
            "training_completed": True,
            "final_loss": 3.5,
            "hyperparameters": {"lr": 10.0, "batch_size": 2},
            "weight_stats": {"mean": 0.0, "std": 0.001},
            "training_was_flawed": True,
        },
    )
    result_4 = analyzer_4.analyze()
    print(analyzer_4.get_analysis_summary())
    pprint.pprint(result_4, width=120, sort_dicts=False, depth=3)
    print()

    # ------------------------------------------------------------------
    # Scenario 5: No diagnostic data yet
    # ------------------------------------------------------------------
    print("=" * 72)
    print("Scenario 5: No diagnostic data yet")
    print("=" * 72)

    analyzer_5 = ModelCollapseAnalyzer("test_empty")
    print(analyzer_5.get_analysis_summary())
    result_5 = analyzer_5.analyze()
    pprint.pprint(result_5, width=120, sort_dicts=False, depth=3)
    print()

    # ------------------------------------------------------------------
    # Scenario 6: Reset and re-analyze
    # ------------------------------------------------------------------
    print("=" * 72)
    print("Scenario 6: Reset and re-analyze")
    print("=" * 72)

    analyzer_1.reset()
    print(f"After reset: {analyzer_1.get_analysis_summary()}")
    analyzer_1.feed_diagnostics(
        surface_diagnostic_report={
            "training_record_count": 1000,
            "p_cont_outputs": [0.48, 0.52, 0.49, 0.51, 0.50],
            "data_staleness_seconds": 100,
            "regime_coverage": 0.95,
            "model_outputs": [0.48, 0.52, 0.49, 0.51, 0.50],
        },
        duality_report={
            "feature_variances": [0.12, 0.18, 0.09],
            "feature_means": [0.4, 0.5, 0.6],
        },
        training_metadata={
            "training_completed": True,
            "final_loss": 0.3,
            "hyperparameters": {"lr": 0.001},
        },
    )
    result_6 = analyzer_1.analyze()
    print(analyzer_1.get_analysis_summary())
    pprint.pprint(result_6, width=120, sort_dicts=False, depth=3)

    print("\n✅ Self-test complete.")
