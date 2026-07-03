"""
System Signal Classification — final decision engine that classifies the
entire system into one of five categories based on all CSRF module outputs.

This is the classification point after all evidence is collected.  It
aggregates outputs from:

  * ``signal_truth_labeler``        — true_alpha_source
  * ``collapse_causality_tracker``  — primary cause, collapse_progression
  * ``decision_surface_visualizer`` — verdict (PLATEAU / PARTIAL / HEALTHY)
  * ``alt_signal_validity_test``    — overall_verdict (VALID / QUESTIONABLE / INVALID)
  * ``signal_entanglement_index``   — verdict (INDEPENDENT / LOW / MODERATE /
                                       HIGH / FULLY_ENTANGLED)
  * ``reality_consistency_gate``    — global_verdict

The five classification categories
-----------------------------------

+--------------------+------+----------------------------------------------------+
| Class              | Code | Meaning                                            |
+====================+======+====================================================+
| SIGNAL-SANITIZED   | A    | Healthy but conservative — signals exist but are   |
|                    |      | filtered heavily.                                  |
+--------------------+------+----------------------------------------------------+
| SIGNAL-COLLAPSED   | B    | Your current suspicion — OSS produces no           |
|                    |      | directional signal.                                |
+--------------------+------+----------------------------------------------------+
| FEATURE-DEGENERATE | C    | Input features lack discriminative power.          |
+--------------------+------+----------------------------------------------------+
| MODEL-MISSPECIFIED | D    | OSS architecture cannot represent the decision     |
|                    |      | boundary.                                          |
+--------------------+------+----------------------------------------------------+
| TRUE NO-ALPHA      | E    | Market regime is effectively unpredictable at this |
|                    |      | resolution.                                        |
+--------------------+------+----------------------------------------------------+
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — classification catalogue
# ---------------------------------------------------------------------------

CLASSIFICATION_MAP = {
    "A": {
        "classification_name": "SIGNAL-SANITIZED",
        "description": (
            "Signals exist but are filtered heavily. The system detects some "
            "directional information but applies conservative thresholds, "
            "resulting in a low pass rate."
        ),
        "recommended_action": (
            "Review signal filtering thresholds and consider relaxing "
            "conservative constraints to capture more opportunities."
        ),
    },
    "B": {
        "classification_name": "SIGNAL-COLLAPSED",
        "description": (
            "OSS specifically produces no directional signal. The OSS surface "
            "is collapsed/plateaued while ALT remains valid, indicating the "
            "problem is isolated to the OSS architecture or training."
        ),
        "recommended_action": (
            "Investigate OSS training pipeline, data distribution, and model "
            "architecture. Consider replacing OSS with ALT as interim."
        ),
    },
    "C": {
        "classification_name": "FEATURE-DEGENERATE",
        "description": (
            "Input features lack discriminative power. Both OSS and ALT are "
            "flat/degenerate, suggesting the underlying feature space provides "
            "no useful information for the current market regime."
        ),
        "recommended_action": (
            "Design new feature engineering pipeline. Explore alternative data "
            "sources, different instrument resolutions, or fundamentally "
            "different feature families."
        ),
    },
    "D": {
        "classification_name": "MODEL-MISSPECIFIED",
        "description": (
            "OSS architecture cannot represent the decision boundary. The OSS "
            "surface shows partial signal but training issues prevent it from "
            "converging to a useful representation."
        ),
        "recommended_action": (
            "Audit OSS model architecture, loss function, and training "
            "procedure. Consider architectural changes such as different "
            "layer depths, activation functions, or regularisation."
        ),
    },
    "E": {
        "classification_name": "TRUE NO-ALPHA",
        "description": (
            "Market regime is effectively unpredictable at this resolution. "
            "No consistent reality exists, neither OSS nor ALT correlates "
            "with forward returns, and entropy is low across all signals."
        ),
        "recommended_action": (
            "Consider pausing trading. Explore higher-timeframe or "
            "cross-asset signals. Accept that the current regime may be "
            "inherently unpredictable at the chosen resolution."
        ),
    },
}

# Thresholds used in the decision tree
LOW_PASS_RATE_THRESHOLD = 0.35   # pass rate below this → sanitised
LOW_ENTROPY_THRESHOLD = 0.30     # entanglement / signal entropy below this
HIGH_FLAT_THRESHOLD = 0.75       # flat rate above this → degenerate

# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------
_instances: Dict[str, "_SystemSignalClassification"] = {}


def SystemSignalClassification(instance_id="default"):
    """Singleton accessor for ``_SystemSignalClassification``.

    Parameters
    ----------
    instance_id : str
        Unique identifier.  Callers sharing the same *instance_id* share the
        same underlying classifier instance.

    Returns
    -------
    _SystemSignalClassification
    """
    if instance_id not in _instances:
        _instances[instance_id] = _SystemSignalClassification(instance_id)
    return _instances[instance_id]


# ---------------------------------------------------------------------------
# Expected CSRF sources
# ---------------------------------------------------------------------------
EXPECTED_SOURCES = frozenset({
    "signal_truth_labeler",
    "collapse_causality_tracker",
    "decision_surface_visualizer",
    "alt_signal_validity_test",
    "signal_entanglement_index",
    "reality_consistency_gate",
})


# ---------------------------------------------------------------------------
# Internal implementation
# ---------------------------------------------------------------------------

class _SystemSignalClassification:
    """Final decision engine that classifies the entire system.

    Stores the latest evidence payload from each CSRF module and, when
    :meth:`classify` is called, evaluates the combined evidence to produce
    one of five classifications.

    Parameters
    ----------
    instance_id : str
        Identifier for this classifier instance (used by singleton registry).
    """

    def __init__(self, instance_id="default"):
        self._instance_id = instance_id

        # Each key maps to the *latest* payload received from that source.
        # ``None`` means the source has not yet fed data.
        self._evidence: Dict[str, Any] = {src: None for src in EXPECTED_SOURCES}

        logger.info(
            "SystemSignalClassification(%r) initialised — expects %d sources",
            instance_id,
            len(EXPECTED_SOURCES),
        )

    # ------------------------------------------------------------------
    # Feed evidence API
    # ------------------------------------------------------------------

    def feed_evidence(self, source: str, data: dict):
        """Feed data from any CSRF source.

        Parameters
        ----------
        source : str
            One of the six expected CSRF source names.
        data : dict
            Source-specific evidence payload.
        """
        if source not in EXPECTED_SOURCES:
            logger.warning(
                "Unknown evidence source '%s' — ignored. Expected: %s",
                source, sorted(EXPECTED_SOURCES),
            )
            return
        self._evidence[source] = data
        logger.debug(
            "evidence updated: %s → %s",
            source,
            {k: v for k, v in data.items()
             if k not in ("per_feature", "symbol_results", "causal_chain")},
        )

    # ------------------------------------------------------------------
    # Evidence access
    # ------------------------------------------------------------------

    def get_readiness_checklist(self) -> Dict[str, bool]:
        """Return a dict mapping each source name to whether evidence has been fed.

        Returns
        -------
        dict
            ``{source_name: True/False, ...}``
        """
        return {src: self._evidence[src] is not None for src in EXPECTED_SOURCES}

    def is_ready(self) -> bool:
        """Return ``True`` iff every expected source has been fed at least once."""
        return all(v is not None for v in self._evidence.values())

    def missing_sources(self) -> List[str]:
        """Return the list of source names that have not yet been fed."""
        return [src for src, val in self._evidence.items() if val is None]

    def reset(self):
        """Clear all stored evidence from every source."""
        for src in self._evidence:
            self._evidence[src] = None
        logger.info("SystemSignalClassification(%r) reset", self._instance_id)

    # ------------------------------------------------------------------
    # Public classification methods
    # ------------------------------------------------------------------

    def get_all_classifications(self) -> Dict[str, dict]:
        """Return all possible classifications with their supporting evidence.

        Returns
        -------
        dict
            Mapping from classification code (``"A"``–``"E"``) to a dict
            containing the full name, description, and recommended action.
        """
        return {
            code: {
                "classification": code,
                "classification_name": info["classification_name"],
                "description": info["description"],
                "recommended_action": info["recommended_action"],
            }
            for code, info in sorted(CLASSIFICATION_MAP.items())
        }

    def classify(self) -> Dict[str, Any]:
        """Evaluate all evidence and produce a final classification.

        Returns
        -------
        dict with keys:
            classification          — ``"A"`` | ``"B"`` | ``"C"`` | ``"D"`` | ``"E"``
            classification_name     — full human-readable name
            confidence              — float in [0.0, 1.0]
            evidence_summary        — short human-readable summary
            supporting_signals      — list of evidence items that support the verdict
            contradicting_signals   — list of evidence items that contradict the verdict
            recommended_action      — suggested next step
        """
        missing = self.missing_sources()
        if missing:
            logger.info("Insufficient evidence — missing: %s", missing)
            return {
                "classification": "?",
                "classification_name": "INSUFFICIENT_EVIDENCE",
                "confidence": 0.0,
                "evidence_summary": f"Missing evidence from: {', '.join(missing)}",
                "supporting_signals": [],
                "contradicting_signals": [f"Missing source: {src}" for src in missing],
                "recommended_action": "Feed evidence from all six CSRF sources before classifying.",
            }

        # ----- Extract key signals -------------------------------------------
        truth = self._evidence["signal_truth_labeler"] or {}
        causality = self._evidence["collapse_causality_tracker"] or {}
        surface = self._evidence["decision_surface_visualizer"] or {}
        alt_test = self._evidence["alt_signal_validity_test"] or {}
        entanglement = self._evidence["signal_entanglement_index"] or {}
        reality = self._evidence["reality_consistency_gate"] or {}

        # Flatten nested dicts for convenience
        true_alpha_source = truth.get("true_alpha_source", "INCONCLUSIVE")

        primary_cause = causality.get("primary_cause", "UNKNOWN")
        collapse_progression = causality.get("collapse_progression", "EARLY")

        surface_verdict = surface.get("verdict", "INSUFFICIENT_DATA")

        alt_verdict = alt_test.get("overall_verdict", "INVALID")

        entanglement_verdict = entanglement.get("verdict", "NONE")
        # Allow per-symbol entanglement as fallback
        symbol_results = entanglement.get("symbol_results", None)
        if symbol_results:
            avg_ei = sum(r.get("entanglement_index", 0.0) for r in symbol_results) / len(symbol_results)
        else:
            avg_ei = entanglement.get("entanglement_index",
                                       entanglement.get("average_entanglement", 1.0))
        # Map entanglement level to pass-rate proxy
        entanglement_level = entanglement_verdict  # e.g. "HIGH", "LOW"

        reality_verdict = reality.get("global_verdict", "NO_REALITY")

        # Derive composite indicators
        truth_is_neither = (true_alpha_source == "NEITHER")
        reality_is_no_reality = (reality_verdict == "NO_REALITY")

        oss_is_plateau = (surface_verdict == "PLATEAU")
        oss_is_partial = (surface_verdict == "PARTIAL")
        oss_is_healthy = (surface_verdict == "HEALTHY")

        alt_is_valid = (alt_verdict == "VALID")
        alt_is_questionable = (alt_verdict == "QUESTIONABLE")
        alt_is_invalid = (alt_verdict == "INVALID")
        # Also treat "ALT IS ALSO PLATEAU" — we infer plateau from surface_verdict
        both_plateau = oss_is_plateau and (alt_verdict in ("INVALID", "QUESTIONABLE"))

        entanglement_is_low = entanglement_verdict in ("INDEPENDENT", "LOW") or avg_ei < LOW_ENTROPY_THRESHOLD
        all_entropy_low = entanglement_is_low and (
            avg_ei < LOW_ENTROPY_THRESHOLD
        )

        is_training_issue = (primary_cause in (
            "FEATURE_VARIANCE_LOSS",
            "NORMALIZATION_SATURATION",
            "ENTROPY_COMPRESSION",
            "TRAINING_ISSUE",
        ))

        # Estimate a "pass rate" — how often signals get through filtering
        # Derived from surface + alt verdict + entanglement
        pass_rate = self._estimate_pass_rate(
            surface_verdict, alt_verdict, entanglement_level,
        )

        signals_exist = (
            oss_is_healthy or oss_is_partial or alt_is_valid
        )
        pass_rate_low = pass_rate < LOW_PASS_RATE_THRESHOLD

        # ----- Decision tree -------------------------------------------------
        supporting: List[str] = []
        contradicting: List[str] = []

        # ---- Check E: TRUE NO-ALPHA -----------------------------------------
        # If truth_labeler says NEITHER AND reality says NO_REALITY
        # AND all entropy is low → TRUE NO-ALPHA (E)
        if truth_is_neither and reality_is_no_reality and all_entropy_low:
            classification_code = "E"
            confidence = min(1.0, 0.7 + (1.0 - avg_ei) * 0.3)
            supporting = [
                f"Signal truth labeler reports '{true_alpha_source}' — no source correlates with returns",
                f"Reality consistency gate reports '{reality_verdict}' — no consistent market truth",
                f"Entanglement is low (EI≈{avg_ei:.3f}) — no signals carry useful information",
            ]
            contradicting = []
            evidence_summary = (
                f"Neither OSS nor ALT correlates with forward returns "
                f"(true_alpha_source={true_alpha_source!r}), reality is absent "
                f"({reality_verdict!r}), and all signal entropy is low "
                f"(EI≈{avg_ei:.3f}). The market regime is unpredictable at this resolution."
            )

        # ---- Check B: SIGNAL-COLLAPSED --------------------------------------
        # If OSS is plateau AND ALT is valid AND entanglement is LOW
        elif oss_is_plateau and alt_is_valid and entanglement_is_low:
            classification_code = "B"
            confidence = min(1.0, 0.65 + (1.0 - avg_ei) * 0.35)
            supporting = [
                f"Decision surface reports '{surface_verdict}' — OSS surface is collapsed",
                f"ALT validity test reports '{alt_verdict}' — ALT signals are valid",
                f"Entanglement is low ({entanglement_verdict}, EI≈{avg_ei:.3f}) — signals not entangled",
            ]
            if truth_is_neither:
                contradicting.append(
                    f"Truth labeler reports '{true_alpha_source}' — no source correlates with returns"
                )
            evidence_summary = (
                f"OSS is plateaued ({surface_verdict!r}) while ALT is valid "
                f"({alt_verdict!r}) and entanglement is low "
                f"({entanglement_verdict}, EI≈{avg_ei:.3f}). "
                f"This points to an OSS-specific collapse."
            )

        # ---- Check C: FEATURE-DEGENERATE ------------------------------------
        # If OSS is plateau AND ALT is ALSO plateau / invalid
        elif oss_is_plateau and not alt_is_valid:
            classification_code = "C"
            confidence = min(1.0, 0.6 + (0.4 if alt_is_invalid else 0.2))
            supporting = [
                f"Decision surface reports '{surface_verdict}' — OSS is plateaued",
                f"ALT validity test reports '{alt_verdict}' — ALT also invalid/questionable",
            ]
            if reality_is_no_reality:
                supporting.append(
                    f"Reality consistency gate reports '{reality_verdict}' — no consistent market truth"
                )
            contradicting = []
            evidence_summary = (
                f"Both OSS ({surface_verdict!r}) and ALT "
                f"({alt_verdict!r}) are degenerate. "
                f"The underlying feature space lacks discriminative power."
            )

        # ---- Check D: MODEL-MISSPECIFIED ------------------------------------
        # If OSS is PARTIAL AND causality says TRAINING_ISSUE
        elif oss_is_partial and is_training_issue:
            classification_code = "D"
            confidence = min(1.0, 0.6 + (0.3 if collapse_progression in ("LATE", "FULL") else 0.0))
            supporting = [
                f"Decision surface reports '{surface_verdict}' — partial signal exists",
                f"Causality tracker identifies primary cause: '{primary_cause}'",
                f"Collapse progression: '{collapse_progression}'",
            ]
            if entanglement_verdict in ("HIGH", "FULLY_ENTANGLED"):
                contradicting.append(
                    f"Entanglement is {entanglement_verdict} — signals are coupled"
                )
            evidence_summary = (
                f"OSS shows partial signal ({surface_verdict!r}) but the causality "
                f"tracker identifies a training/data issue "
                f"(primary_cause={primary_cause!r}, "
                f"progression={collapse_progression!r}). "
                f"The architecture cannot represent the decision boundary."
            )

        # ---- Check A: SIGNAL-SANITIZED (default when signals exist but filtered)
        elif signals_exist and pass_rate_low:
            classification_code = "A"
            confidence = min(1.0, 0.5 + pass_rate * 0.5)
            supporting = [
                f"Signals detected: surface={surface_verdict!r}, ALT={alt_verdict!r}",
                f"Pass rate is low ({pass_rate:.2f}) — heavy filtering in effect",
            ]
            if reality_verdict in ("CONSISTENT_REALITY", "PARTIAL_REALITY"):
                supporting.append(f"Reality consistency: {reality_verdict!r}")
            contradicting = []
            if reality_is_no_reality:
                contradicting.append("Reality gate reports 'NO_REALITY' despite signals")
            evidence_summary = (
                f"Signals exist (surface={surface_verdict!r}, ALT={alt_verdict!r}) "
                f"but the effective pass rate is low ({pass_rate:.2f}). "
                f"The system is conservatively filtering most signals."
            )

        # ---- Fallback -------------------------------------------------------
        else:
            classification_code = "?"
            confidence = 0.0
            supporting = []
            contradicting = [
                f"Surface={surface_verdict!r}, ALT={alt_verdict!r}, "
                f"truth={true_alpha_source!r}, reality={reality_verdict!r}, "
                f"entanglement={entanglement_verdict!r}, causality={primary_cause!r}",
            ]
            evidence_summary = (
                "Evidence does not match any expected classification pattern. "
                "Manual review required."
            )

        classification_name = CLASSIFICATION_MAP.get(classification_code, {}).get(
            "classification_name", "UNKNOWN"
        )
        recommended_action = CLASSIFICATION_MAP.get(classification_code, {}).get(
            "recommended_action",
            "Inspect raw evidence from all six CSRF sources manually.",
        )

        return {
            "classification": classification_code,
            "classification_name": classification_name,
            "confidence": round(confidence, 4),
            "evidence_summary": evidence_summary,
            "supporting_signals": supporting,
            "contradicting_signals": contradicting,
            "recommended_action": recommended_action,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_pass_rate(
        surface_verdict: str,
        alt_verdict: str,
        entanglement_level: str,
    ) -> float:
        """Estimate the effective signal pass rate from available verdicts.

        Returns a float in [0.0, 1.0] where higher means more signals pass
        through filtering.
        """
        # Base pass rate from surface verdict
        surface_map = {
            "HEALTHY": 0.8,
            "PARTIAL": 0.5,
            "PLATEAU": 0.15,
            "INSUFFICIENT_DATA": 0.0,
        }
        base = surface_map.get(surface_verdict, 0.0)

        # ALT modifier
        alt_map = {
            "VALID": 0.2,
            "QUESTIONABLE": 0.0,
            "INVALID": -0.1,
        }
        alt_mod = alt_map.get(alt_verdict, 0.0)

        # Entanglement modifier — high entanglement means signals are redundant
        ent_map = {
            "INDEPENDENT": 0.1,
            "LOW": 0.05,
            "MODERATE": 0.0,
            "HIGH": -0.05,
            "FULLY_ENTANGLED": -0.1,
        }
        ent_mod = ent_map.get(entanglement_level, 0.0)

        return max(0.0, min(1.0, base + alt_mod + ent_mod))

    # ------------------------------------------------------------------
    # Convenience — feed from specific sources
    # ------------------------------------------------------------------

    def feed_signal_truth_labeler(self, data: dict):
        """Feed evidence from ``signal_truth_labeler``.

        Expected keys: ``true_alpha_source``, ``oss_accuracy``, ``alt_accuracy``.
        """
        self.feed_evidence("signal_truth_labeler", data)

    def feed_collapse_causality_tracker(self, data: dict):
        """Feed evidence from ``collapse_causality_tracker``.

        Expected keys: ``primary_cause``, ``collapse_progression``.
        """
        self.feed_evidence("collapse_causality_tracker", data)

    def feed_decision_surface_visualizer(self, data: dict):
        """Feed evidence from ``decision_surface_visualizer``.

        Expected keys: ``verdict`` (PLATEAU / PARTIAL / HEALTHY).
        """
        self.feed_evidence("decision_surface_visualizer", data)

    def feed_alt_signal_validity_test(self, data: dict):
        """Feed evidence from ``alt_signal_validity_test``.

        Expected keys: ``overall_verdict`` (VALID / QUESTIONABLE / INVALID).
        """
        self.feed_evidence("alt_signal_validity_test", data)

    def feed_signal_entanglement_index(self, data: dict):
        """Feed evidence from ``signal_entanglement_index``.

        Expected keys: ``verdict`` (INDEPENDENT / LOW / MODERATE / HIGH /
        FULLY_ENTANGLED), ``entanglement_index``, ``average_entanglement``,
        ``symbol_results``.
        """
        self.feed_evidence("signal_entanglement_index", data)

    def feed_reality_consistency_gate(self, data: dict):
        """Feed evidence from ``reality_consistency_gate``.

        Expected keys: ``global_verdict`` (CONSISTENT_REALITY / PARTIAL_REALITY /
        NO_REALITY).
        """
        self.feed_evidence("reality_consistency_gate", data)


# ===================================================================
# Self-test
# ===================================================================

def _selftest():
    """Exercise all five classification scenarios to verify the decision tree.

    Scenarios
    ---------
    1. TRUE NO-ALPHA (E)   — NEITHER + NO_REALITY + low entropy
    2. SIGNAL-COLLAPSED (B) — OSS plateau + ALT valid + low entanglement
    3. FEATURE-DEGENERATE (C) — OSS plateau + ALT invalid
    4. MODEL-MISSPECIFIED (D) — OSS partial + training issue causality
    5. SIGNAL-SANITIZED (A) — signals exist + low pass rate
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info("=" * 60)
    logger.info("SystemSignalClassification — Self-Test")
    logger.info("=" * 60)

    passed = 0
    failed = 0

    def run_scenario(label, expected_code, feed_fn):
        """Feed evidence, call classify(), assert code, log result."""
        nonlocal passed, failed
        classifier = SystemSignalClassification(f"selftest_{label}")
        feed_fn(classifier)
        result = classifier.classify()
        code = result["classification"]

        ok = code == expected_code
        if ok:
            passed += 1
            logger.info(
                "  [PASS] %-30s → %s (%s)  conf=%.2f",
                label,
                code,
                result["classification_name"],
                result["confidence"],
            )
        else:
            failed += 1
            logger.warning(
                "  [FAIL] %-30s expected %s, got %s (%s)  conf=%.2f",
                label,
                expected_code,
                code,
                result["classification_name"],
                result["confidence"],
            )
            logger.warning("         summary: %s", result["evidence_summary"])
        return result

    # ===== Scenario 1: TRUE NO-ALPHA (E) ====================================
    # truth_labeler says NEITHER, reality says NO_REALITY, low entropy

    def _feed_e(g):
        g.feed_signal_truth_labeler({
            "true_alpha_source": "NEITHER",
            "oss_accuracy": 0.42,
            "alt_accuracy": 0.39,
            "samples": 500,
        })
        g.feed_collapse_causality_tracker({
            "primary_cause": "FEATURE_VARIANCE_LOSS",
            "collapse_progression": "FULL",
        })
        g.feed_decision_surface_visualizer({
            "verdict": "PLATEAU",
            "is_plateau": True,
        })
        g.feed_alt_signal_validity_test({
            "overall_verdict": "INVALID",
        })
        g.feed_signal_entanglement_index({
            "verdict": "INDEPENDENT",
            "average_entanglement": 0.05,
            "symbol_results": [
                {"symbol": "EURUSD", "entanglement_index": 0.04},
                {"symbol": "USDJPY", "entanglement_index": 0.06},
            ],
        })
        g.feed_reality_consistency_gate({
            "global_verdict": "NO_REALITY",
        })

    run_scenario("true_no_alpha", "E", _feed_e)

    # ===== Scenario 2: SIGNAL-COLLAPSED (B) =================================
    # OSS plateau + ALT valid + low entanglement

    def _feed_b(g):
        g.feed_signal_truth_labeler({
            "true_alpha_source": "ALT",
            "oss_accuracy": 0.48,
            "alt_accuracy": 0.62,
            "samples": 500,
        })
        g.feed_collapse_causality_tracker({
            "primary_cause": "NORMALIZATION_SATURATION",
            "collapse_progression": "LATE",
        })
        g.feed_decision_surface_visualizer({
            "verdict": "PLATEAU",
            "is_plateau": True,
        })
        g.feed_alt_signal_validity_test({
            "overall_verdict": "VALID",
        })
        g.feed_signal_entanglement_index({
            "verdict": "LOW",
            "average_entanglement": 0.15,
            "symbol_results": [
                {"symbol": "EURUSD", "entanglement_index": 0.12},
                {"symbol": "USDJPY", "entanglement_index": 0.18},
            ],
        })
        g.feed_reality_consistency_gate({
            "global_verdict": "PARTIAL_REALITY",
        })

    run_scenario("signal_collapsed", "B", _feed_b)

    # ===== Scenario 3: FEATURE-DEGENERATE (C) ===============================
    # OSS plateau + ALT invalid

    def _feed_c(g):
        g.feed_signal_truth_labeler({
            "true_alpha_source": "NEITHER",
            "oss_accuracy": 0.45,
            "alt_accuracy": 0.41,
            "samples": 500,
        })
        g.feed_collapse_causality_tracker({
            "primary_cause": "FEATURE_VARIANCE_LOSS",
            "collapse_progression": "FULL",
        })
        g.feed_decision_surface_visualizer({
            "verdict": "PLATEAU",
            "is_plateau": True,
        })
        g.feed_alt_signal_validity_test({
            "overall_verdict": "INVALID",
        })
        g.feed_signal_entanglement_index({
            "verdict": "MODERATE",
            "average_entanglement": 0.45,
            "symbol_results": [
                {"symbol": "EURUSD", "entanglement_index": 0.42},
                {"symbol": "USDJPY", "entanglement_index": 0.48},
            ],
        })
        g.feed_reality_consistency_gate({
            "global_verdict": "NO_REALITY",
        })

    run_scenario("feature_degenerate", "C", _feed_c)

    # ===== Scenario 4: MODEL-MISSPECIFIED (D) ===============================
    # OSS partial + causality says training issue

    def _feed_d(g):
        g.feed_signal_truth_labeler({
            "true_alpha_source": "INCONCLUSIVE",
            "oss_accuracy": 0.52,
            "alt_accuracy": 0.48,
            "samples": 500,
        })
        g.feed_collapse_causality_tracker({
            "primary_cause": "NORMALIZATION_SATURATION",
            "collapse_progression": "MID",
        })
        g.feed_decision_surface_visualizer({
            "verdict": "PARTIAL",
            "is_plateau": False,
        })
        g.feed_alt_signal_validity_test({
            "overall_verdict": "QUESTIONABLE",
        })
        g.feed_signal_entanglement_index({
            "verdict": "LOW",
            "average_entanglement": 0.20,
            "symbol_results": [
                {"symbol": "EURUSD", "entanglement_index": 0.18},
                {"symbol": "USDJPY", "entanglement_index": 0.22},
            ],
        })
        g.feed_reality_consistency_gate({
            "global_verdict": "PARTIAL_REALITY",
        })

    run_scenario("model_misspecified", "D", _feed_d)

    # ===== Scenario 5: SIGNAL-SANITIZED (A) =================================
    # Signals exist but pass rate is low

    def _feed_a(g):
        g.feed_signal_truth_labeler({
            "true_alpha_source": "OSS",
            "oss_accuracy": 0.55,
            "alt_accuracy": 0.47,
            "samples": 500,
        })
        g.feed_collapse_causality_tracker({
            "primary_cause": "UNKNOWN",
            "collapse_progression": "EARLY",
        })
        g.feed_decision_surface_visualizer({
            "verdict": "PARTIAL",
            "is_plateau": False,
        })
        g.feed_alt_signal_validity_test({
            "overall_verdict": "INVALID",
        })
        g.feed_signal_entanglement_index({
            "verdict": "FULLY_ENTANGLED",
            "average_entanglement": 0.92,
            "symbol_results": [
                {"symbol": "EURUSD", "entanglement_index": 0.91},
                {"symbol": "USDJPY", "entanglement_index": 0.93},
            ],
        })
        g.feed_reality_consistency_gate({
            "global_verdict": "CONSISTENT_REALITY",
        })

    run_scenario("signal_sanitized", "A", _feed_a)

    # ===== Scenario 6: Missing evidence ======================================

    def _feed_missing(g):
        # Only feed one source
        g.feed_decision_surface_visualizer({
            "verdict": "HEALTHY",
        })

    run_scenario("missing_evidence", "?", _feed_missing)

    # ===== Additional checks =================================================

    # get_all_classifications returns all 5
    classifier = SystemSignalClassification("selftest_all")
    all_classes = classifier.get_all_classifications()
    assert len(all_classes) == 5, f"Expected 5 classes, got {len(all_classes)}"
    for code in ("A", "B", "C", "D", "E"):
        assert code in all_classes, f"Missing classification code {code}"
    logger.info("  [PASS] %-30s → all 5 classifications returned", "get_all_classifications")

    # Readiness checks
    gate_r = SystemSignalClassification("selftest_readiness")
    checklist = gate_r.get_readiness_checklist()
    assert not gate_r.is_ready(), "Should NOT be ready before any feeds"
    assert all(v is False for v in checklist.values()), \
        "All checklist entries should be False initially"
    logger.info("  [PASS] %-30s → is_ready=False, all sources unchecked", "readiness_before")

    gate_r.feed_signal_truth_labeler({"true_alpha_source": "OSS"})
    assert not gate_r.is_ready(), "Should still NOT be ready after 1 of 6 feeds"
    logger.info("  [PASS] %-30s → is_ready=False after 1/6 feeds", "readiness_partial")

    # Feed remaining 5
    gate_r.feed_collapse_causality_tracker({"primary_cause": "UNKNOWN"})
    gate_r.feed_decision_surface_visualizer({"verdict": "HEALTHY"})
    gate_r.feed_alt_signal_validity_test({"overall_verdict": "VALID"})
    gate_r.feed_signal_entanglement_index({"verdict": "INDEPENDENT"})
    gate_r.feed_reality_consistency_gate({"global_verdict": "CONSISTENT_REALITY"})
    assert gate_r.is_ready(), "SHOULD be ready after all 6 feeds"
    logger.info("  [PASS] %-30s → is_ready=True after 6/6 feeds", "readiness_full")

    # Reset
    gate_r.reset()
    assert not gate_r.is_ready(), "Should NOT be ready after reset"
    logger.info("  [PASS] %-30s → is_ready=False after reset", "readiness_after_reset")

    passed += 7  # get_all_classifications (1) + readiness (4) + reset (1) + missing (1 counted above)

    # ===== Summary ===========================================================
    logger.info("-" * 60)
    total = passed + failed
    logger.info(
        "Results:  %d / %d passed  (%s)",
        passed,
        total,
        "ALL PASSED" if failed == 0 else f"{failed} FAILED",
    )

    if failed > 0:
        logger.error(">>> SELF-TEST FAILED <<<")
    else:
        logger.info(">>> SELF-TEST PASSED <<<")


if __name__ == "__main__":
    _selftest()
