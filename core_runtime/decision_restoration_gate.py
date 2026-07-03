"""
Decision Restoration Gate — decides the next phase after SDIL diagnostic batch
completes.

Inputs from ALL other SDIL modules:

  * ``alt_signal_generator``        — ALT signal statistics
  * ``signal_duality_engine``       — duality report
  * ``oss_surface_diagnostic``      — OSS diagnostic
  * ``validation_signal_injector``  — injection mode status
  * ``execution_independence_test`` — execution layer test verdict
  * ``signal_space_entropy``        — global entropy assessment
  * ``model_collapse_analyzer``     — collapse root cause

Outputs one of five verdicts:

  REPAIR_OSS
      OSS has signal but collapsed at a specific regime → fix training / data.
  REPLACE_OSS
      OSS irrecoverably collapsed, ALT works → replace with ALT.
  HYBRID
      Both have partial signal → use ensemble.
  DISCARD_OSS
      Both collapsed → signal space degenerate, need new data pipeline.
  INCONCLUSIVE
      Insufficient data from one or more sources.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------
_instances = {}


def DecisionRestorationGate(instance_id="default"):
    """Singleton accessor for ``_DecisionRestorationGate``.

    Parameters
    ----------
    instance_id : str
        Unique identifier.  Callers sharing the same *instance_id* share the
        same underlying gate object.

    Returns
    -------
    _DecisionRestorationGate
    """
    if instance_id not in _instances:
        _instances[instance_id] = _DecisionRestorationGate(instance_id)
    return _instances[instance_id]


# ---------------------------------------------------------------------------
# Internal implementation
# ---------------------------------------------------------------------------
EXPECTED_SOURCES = frozenset({
    "alt_signal_generator",
    "signal_duality_engine",
    "oss_surface_diagnostic",
    "validation_signal_injector",
    "execution_independence_test",
    "signal_space_entropy",
    "model_collapse_analyzer",
})


class _DecisionRestorationGate:
    """Decides the next restoration phase after SDIL diagnostics complete.

    Stores the latest report from each SDIL module and, when ``decide()`` is
    called, evaluates the combined evidence to produce a single verdict.
    """

    def __init__(self, instance_id="default"):
        self._instance_id = instance_id

        # Each key maps to the *latest* payload received from that source.
        # ``None`` means the source has not yet fed data.
        self._data: Dict[str, Any] = {src: None for src in EXPECTED_SOURCES}

        logger.info(
            "DecisionRestorationGate(%r) initialised — expects %d sources",
            instance_id,
            len(EXPECTED_SOURCES),
        )

    # ------------------------------------------------------------------
    # Feed API  (one method per source, plus a generic ``feed``)
    # ------------------------------------------------------------------

    def feed_alt_signal_generator(self, stats: dict):
        """Store ALT signal generator statistics.

        Expected keys: ``total_signals``, ``buy_pct``, ``sell_pct``,
        ``flat_pct``, ``mean_confidence``, ``signal_std``.
        """
        self._data["alt_signal_generator"] = stats
        logger.debug("alt_signal_generator updated: flat_pct=%.2f", stats.get("flat_pct", -1))

    def feed_signal_duality_engine(self, report: dict):
        """Store duality engine report (aggregated summary or per-symbol).

        The gate uses the aggregated summary; pass either the output of
        ``get_summary()`` or a dict with equivalent keys.
        """
        self._data["signal_duality_engine"] = report
        logger.debug("signal_duality_engine updated: verdict=%s", report.get("verdict", "?"))

    def feed_oss_surface_diagnostic(self, diagnostic: dict):
        """Store OSS surface diagnostic.

        Expected keys: ``collapsed`` (bool), ``regime`` (str),
        ``signal_quality`` (float), ``details`` (str).
        """
        self._data["oss_surface_diagnostic"] = diagnostic
        logger.debug(
            "oss_surface_diagnostic updated: collapsed=%s",
            diagnostic.get("collapsed", "?"),
        )

    def feed_validation_signal_injector(self, stats: dict):
        """Store signal injection mode status.

        Expected keys: ``active`` (bool), ``mode`` (str),
        ``injected_count`` (int), ``last_injection`` (str).
        """
        self._data["validation_signal_injector"] = stats
        logger.debug("validation_signal_injector updated: active=%s", stats.get("active", "?"))

    def feed_execution_independence_test(self, verdict: dict):
        """Store execution layer independence test verdict.

        Expected keys: ``independent`` (bool), ``confidence`` (float),
        ``details`` (str).
        """
        self._data["execution_independence_test"] = verdict
        logger.debug(
            "execution_independence_test updated: independent=%s",
            verdict.get("independent", "?"),
        )

    def feed_signal_space_entropy(self, assessment: dict):
        """Store global entropy assessment.

        Expected keys: ``entropy`` (float), ``regime`` (str,
        e.g. ``"degenerate"``), ``confidence`` (float).
        """
        self._data["signal_space_entropy"] = assessment
        logger.debug(
            "signal_space_entropy updated: entropy=%.4f",
            assessment.get("entropy", -1),
        )

    def feed_model_collapse_analyzer(self, analysis: dict):
        """Store collapse root-cause analysis.

        Expected keys: ``collapse_detected`` (bool), ``root_cause`` (str),
        ``severity`` (str), ``recommendation`` (str).
        """
        self._data["model_collapse_analyzer"] = analysis
        logger.debug(
            "model_collapse_analyzer updated: collapse_detected=%s",
            analysis.get("collapse_detected", "?"),
        )

    def feed(self, source: str, payload: dict):
        """Generic feed — route *payload* to the correct dedicated method.

        Parameters
        ----------
        source : str
            One of the seven expected source names.
        payload : dict
            Source-specific data dict.
        """
        feeder = {
            "alt_signal_generator": self.feed_alt_signal_generator,
            "signal_duality_engine": self.feed_signal_duality_engine,
            "oss_surface_diagnostic": self.feed_oss_surface_diagnostic,
            "validation_signal_injector": self.feed_validation_signal_injector,
            "execution_independence_test": self.feed_execution_independence_test,
            "signal_space_entropy": self.feed_signal_space_entropy,
            "model_collapse_analyzer": self.feed_model_collapse_analyzer,
        }
        if source not in feeder:
            logger.warning("Unknown source '%s' — ignored", source)
            return
        feeder[source](payload)

    # ------------------------------------------------------------------
    # Readiness
    # ------------------------------------------------------------------

    def get_readiness_checklist(self) -> Dict[str, bool]:
        """Return a dict mapping each source name to whether data has been fed.

        Returns
        -------
        dict
            ``{source_name: True/False, ...}``
        """
        return {src: self._data[src] is not None for src in EXPECTED_SOURCES}

    def is_ready(self) -> bool:
        """Return ``True`` iff every expected source has been fed at least once."""
        return all(v is not None for v in self._data.values())

    def missing_sources(self) -> List[str]:
        """Return the list of source names that have not yet been fed."""
        return [src for src, val in self._data.items() if val is None]

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self):
        """Clear all stored data from every source."""
        for src in self._data:
            self._data[src] = None
        logger.info("DecisionRestorationGate(%r) reset", self._instance_id)

    # ------------------------------------------------------------------
    # Core decision logic
    # ------------------------------------------------------------------

    def decide(
        self,
        alt_stats: Optional[dict] = None,
        duality_report: Optional[dict] = None,
        oss_diagnostic: Optional[dict] = None,
        injector_stats: Optional[dict] = None,
        exec_test: Optional[dict] = None,
        entropy_assessment: Optional[dict] = None,
        collapse_analysis: Optional[dict] = None,
    ) -> dict:
        """Evaluate all evidence and produce a verdict.

        Parameters
        ----------
        alt_stats : dict or None
            ALT signal generator statistics.  If provided, stored internally.
        duality_report : dict or None
            Duality engine report (aggregated summary preferred).
        oss_diagnostic : dict or None
            OSS surface diagnostic.
        injector_stats : dict or None
            Validation signal injector status.
        exec_test : dict or None
            Execution independence test verdict.
        entropy_assessment : dict or None
            Global entropy assessment.
        collapse_analysis : dict or None
            Model collapse root-cause analysis.

        Any argument that is not ``None`` is both used immediately and stored
        via the corresponding ``feed_*()`` method.
        """
        # Store any new data that was passed in
        if alt_stats is not None:
            self.feed_alt_signal_generator(alt_stats)
        if duality_report is not None:
            self.feed_signal_duality_engine(duality_report)
        if oss_diagnostic is not None:
            self.feed_oss_surface_diagnostic(oss_diagnostic)
        if injector_stats is not None:
            self.feed_validation_signal_injector(injector_stats)
        if exec_test is not None:
            self.feed_execution_independence_test(exec_test)
        if entropy_assessment is not None:
            self.feed_signal_space_entropy(entropy_assessment)
        if collapse_analysis is not None:
            self.feed_model_collapse_analyzer(collapse_analysis)

        # ----- Check for missing data ---------------------------------------
        missing = self.missing_sources()
        if missing:
            logger.info("Insufficient data — missing: %s", missing)
            return {
                "verdict": "INCONCLUSIVE",
                "confidence": 0.0,
                "reasoning": f"Missing data from: {', '.join(missing)}",
                "next_steps": [f"Feed data from: {src}" for src in missing],
                "warnings": ["Decision deferred until all sources report."],
            }

        # ----- Extract key signals -------------------------------------------
        alt = self._data["alt_signal_generator"]
        duality = self._data["signal_duality_engine"]
        oss = self._data["oss_surface_diagnostic"]
        injector = self._data["validation_signal_injector"]
        exec_test = self._data["execution_independence_test"]
        entropy = self._data["signal_space_entropy"]
        collapse = self._data["model_collapse_analyzer"]

        alt_flat_pct = alt.get("flat_pct", 100.0)
        alt_mean_conf = alt.get("mean_confidence", 0.0)
        alt_signal_std = alt.get("signal_std", 0.0)

        oss_collapsed = oss.get("collapsed", True)
        oss_signal_quality = oss.get("signal_quality", 0.0)

        duality_verdict = duality.get("verdict", "CONFLICTING")
        global_oss_flat = duality.get("global_oss_flat_rate", 1.0)
        global_alt_flat = duality.get("global_alt_flat_rate", 1.0)

        entropy_regime = entropy.get("regime", "unknown")
        entropy_value = entropy.get("entropy", 0.0)
        entropy_conf = entropy.get("confidence", 0.0)

        collapse_detected = collapse.get("collapse_detected", False)
        collapse_root = collapse.get("root_cause", "unknown")
        collapse_severity = collapse.get("severity", "low")

        alt_produces_signals = alt_flat_pct < 80.0 and alt_mean_conf > 0.2
        oss_uniquely_flat = (
            duality_verdict == "OSS_UNIQUELY_FLAT"
            or (global_oss_flat > 0.80 and global_alt_flat < 0.30)
        )
        both_produce = (
            global_oss_flat < 0.30
            and global_alt_flat < 0.30
            and duality_verdict in ("BOTH_ACTIVE", "CONFLICTING")
        )
        both_flat = (
            global_oss_flat > 0.80
            and global_alt_flat > 0.80
        ) or duality_verdict == "BOTH_FLAT"

        oss_signal_decent = oss_signal_quality >= 0.5
        space_degenerate = entropy_regime == "degenerate" and entropy_conf > 0.6

        # ----- Decision tree -------------------------------------------------
        warnings: List[str] = []

        # CASE A: both collapsed / space degenerate
        if both_flat or space_degenerate:
            if both_flat:
                reasoning = (
                    f"Both OSS (flat rate {global_oss_flat:.0%}) and ALT "
                    f"(flat rate {global_alt_flat:.0%}) are predominantly flat. "
                    "Signal space appears degenerate."
                )
            else:
                reasoning = (
                    f"Signal space entropy assessment reports regime "
                    f"'{entropy_regime}' (entropy={entropy_value:.4f}, "
                    f"confidence={entropy_conf:.2f}). Combined with high "
                    f"OSS flat rate ({global_oss_flat:.0%}) and ALT flat rate "
                    f"({global_alt_flat:.0%}), the signal space is degenerate."
                )

            if collapse_detected:
                reasoning += (
                    f" Model collapse confirmed (severity={collapse_severity}, "
                    f"root_cause={collapse_root})."
                )

            return {
                "verdict": "DISCARD_OSS",
                "confidence": round(min(1.0, entropy_conf + 0.2), 4),
                "reasoning": reasoning,
                "next_steps": [
                    "Design new data pipeline with different feature sources",
                    "Review data collection methodology for systemic biases",
                    "Consider switching to fundamentally different market data",
                    "Run ablation study to identify information-dead features",
                ],
                "warnings": [
                    "Discarding OSS means losing all prior investment in the model",
                    "New pipeline will require full re-validation (weeks)",
                    "Ensure ALT is also evaluated — it may also be flat",
                ],
            }

        # CASE B: OSS not collapsed but low quality → repair first
        #          (checked before HYBRID so low-quality OSS routes to repair
        #           even when ALT also produces signals)
        if (not oss_collapsed and oss_signal_quality < 0.5) or (
            not oss_collapsed and oss_signal_quality < 0.6
            and duality_verdict in ("CONFLICTING",)
        ):
            reasoning = (
                f"OSS is not collapsed (collapsed={oss_collapsed}) but signal "
                f"quality is low ({oss_signal_quality:.3f}). "
            )
            if alt_produces_signals:
                reasoning += (
                    f"ALT is producing signals (flat_pct={alt_flat_pct:.1f}%) which "
                    "suggests signal is available — OSS training/data needs fixing."
                )
            else:
                reasoning += (
                    "ALT is also weak — the issue may be partially data-related, "
                    "but OSS should be repaired before considering discard."
                )

            if collapse_detected:
                reasoning += (
                    f" Model collapse detected ({collapse_root}); repair should "
                    "address the root cause."
                )

            next_steps = [
                "Audit OSS training data for concept drift",
                "Re-train OSS with expanded feature set",
                "Adjust hyperparameters (learning rate, regularization)",
                "Run A/B comparison: old OSS vs repaired OSS on recent regime",
            ]

            warnings.append(
                "Repair may fail if underlying data pipeline is the root cause"
            )
            if collapse_detected:
                warnings.append(
                    f"Collapse severity '{collapse_severity}' — may require "
                    "architectural rewrite, not just retraining"
                )

            return {
                "verdict": "REPAIR_OSS",
                "confidence": round(
                    min(1.0, (1.0 - oss_signal_quality) + 0.1), 4
                ),
                "reasoning": reasoning,
                "next_steps": next_steps,
                "warnings": warnings,
            }

        # CASE C: ALT works, OSS uniquely flat → replace temporarily
        if alt_produces_signals and oss_uniquely_flat:
            reasoning = (
                f"ALT produces directional signals (flat_pct={alt_flat_pct:.1f}%, "
                f"confidence={alt_mean_conf:.3f}) while OSS is uniquely flat "
                f"(oss_flat_rate={global_oss_flat:.0%}, alt_flat_rate={global_alt_flat:.0%}). "
            )
            if collapse_detected:
                reasoning += (
                    f" Model collapse detected (root_cause={collapse_root}). "
                    "Recommend replacing OSS with ALT as a temporary measure."
                )
            else:
                reasoning += (
                    "This suggests the OSS architecture is unable to extract "
                    "signal from current market conditions. ALT can serve as interim."
                )

            next_steps = [
                "Deploy ALT as primary signal source immediately",
                "Set up parallel shadow tracking to log OSS-originated signals",
                "Investigate OSS architecture failure — feature engineering or model capacity",
                "Schedule OSS retraining or architecture revision",
            ]

            warnings.append(
                "ALT is a simple baseline — prolonged use may degrade performance"
            )
            if collapse_detected:
                warnings.append(
                    f"Collapse root cause '{collapse_root}' must be addressed "
                    "before OSS can be restored"
                )

            return {
                "verdict": "REPLACE_OSS",
                "confidence": round(
                    min(1.0, alt_mean_conf + (0.2 if collapse_detected else 0.0)), 4
                ),
                "reasoning": reasoning,
                "next_steps": next_steps,
                "warnings": warnings,
            }

        # CASE D: both produce signals with decent quality → ensemble
        if (both_produce or (alt_produces_signals and oss_signal_decent)):
            reasoning = (
                f"Both OSS (collapsed={oss_collapsed}, quality={oss_signal_quality:.3f}) "
                f"and ALT (flat_pct={alt_flat_pct:.1f}%, confidence={alt_mean_conf:.3f}) "
                "produce directional signals. "
            )

            if global_oss_flat < 0.30 and global_alt_flat < 0.30:
                reasoning += (
                    "Duality engine confirms both are active (oss_flat_rate="
                    f"{global_oss_flat:.0%}, alt_flat_rate={global_alt_flat:.0%}). "
                )

            reasoning += "An ensemble approach can exploit both signal sources."

            next_steps = [
                "Design hybrid weighting scheme (signal-quality-weighted average)",
                "Set up ensemble signal combiner module",
                "Monitor per-source Sharpe ratio to adjust weights dynamically",
                "Run backtest on hybrid before production deployment",
            ]

            warnings.append(
                "Ensemble complexity adds maintenance overhead"
            )
            if not exec_test.get("independent", True):
                warnings.append(
                    "Execution layer not independent — ensemble benefits may be reduced"
                )

            return {
                "verdict": "HYBRID",
                "confidence": round(
                    min(1.0, (oss_signal_quality + alt_mean_conf) / 2.0 + 0.1), 4
                ),
                "reasoning": reasoning,
                "next_steps": next_steps,
                "warnings": warnings,
            }

        # CASE E: fallback — should not normally be reached
        logger.warning(
            "Fallthrough in decision logic — all sources present but none of the "
            "expected patterns matched.  alt_flat=%.2f alt_conf=%.2f "
            "oss_collapsed=%s oss_quality=%.3f duality_verdict=%s",
            alt_flat_pct,
            alt_mean_conf,
            oss_collapsed,
            oss_signal_quality,
            duality_verdict,
        )
        return {
            "verdict": "INCONCLUSIVE",
            "confidence": 0.0,
            "reasoning": (
                "All sources available but evidence does not match any expected "
                "pattern.  Manual review required."
            ),
            "next_steps": [
                "Inspect raw data from all seven sources manually",
                "Check for inconsistent or contradictory signals",
            ],
            "warnings": [
                "Unrecognised pattern — automated decision may be unreliable",
            ],
        }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
def _selftest():
    """Exercise multiple decision scenarios to verify logic."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info("=" * 60)
    logger.info("DecisionRestorationGate — Self-Test")
    logger.info("=" * 60)

    passed = 0
    failed = 0

    # ------------------------------------------------------------------
    def run_scenario(label, verdict_expected, feed_fn):
        """Feed data, call decide(), assert verdict, log result."""
        nonlocal passed, failed
        gate = DecisionRestorationGate(f"selftest_{label}")
        feed_fn(gate)
        result = gate.decide()
        verdict = result["verdict"]

        ok = verdict == verdict_expected
        if ok:
            passed += 1
            logger.info(
                "  [PASS] %-30s → %s  (conf=%.2f)",
                label,
                verdict,
                result["confidence"],
            )
        else:
            failed += 1
            logger.warning(
                "  [FAIL] %-30s expected %s, got %s  (conf=%.2f)",
                label,
                verdict_expected,
                verdict,
                result["confidence"],
            )
            logger.warning("         reasoning: %s", result["reasoning"])
        return result

    # ===== Scenario 1: REPLACE_OSS =====================================
    # ALT produces signals, OSS is uniquely flat
    def _feed_replace(g):
        g.feed_alt_signal_generator({
            "total_signals": 1000,
            "buy_pct": 35.0,
            "sell_pct": 30.0,
            "flat_pct": 35.0,
            "mean_confidence": 0.65,
            "signal_std": 0.72,
        })
        g.feed_signal_duality_engine({
            "total_observations": 2000,
            "global_agreement_rate": 0.20,
            "global_divergence_rate": 0.80,
            "global_oss_flat_rate": 0.92,
            "global_alt_flat_rate": 0.18,
            "verdict": "OSS_UNIQUELY_FLAT",
        })
        g.feed_oss_surface_diagnostic({
            "collapsed": True,
            "regime": "low_volatility_noise",
            "signal_quality": 0.12,
            "details": "OSS outputs mostly zero / constant",
        })
        g.feed_validation_signal_injector({
            "active": False,
            "mode": "idle",
            "injected_count": 0,
            "last_injection": "N/A",
        })
        g.feed_execution_independence_test({
            "independent": True,
            "confidence": 0.92,
            "details": "Execution layer independent from signal generation",
        })
        g.feed_signal_space_entropy({
            "entropy": 0.35,
            "regime": "low_diversity",
            "confidence": 0.78,
        })
        g.feed_model_collapse_analyzer({
            "collapse_detected": True,
            "root_cause": "feature_standardization_drift",
            "severity": "high",
            "recommendation": "Recompute feature moments on recent window",
        })

    run_scenario("replace_oss", "REPLACE_OSS", _feed_replace)

    # ===== Scenario 2: HYBRID ==========================================
    # Both produce signals
    def _feed_hybrid(g):
        g.feed_alt_signal_generator({
            "total_signals": 800,
            "buy_pct": 28.0,
            "sell_pct": 32.0,
            "flat_pct": 40.0,
            "mean_confidence": 0.55,
            "signal_std": 0.68,
        })
        g.feed_signal_duality_engine({
            "total_observations": 1600,
            "global_agreement_rate": 0.45,
            "global_divergence_rate": 0.55,
            "global_oss_flat_rate": 0.25,
            "global_alt_flat_rate": 0.28,
            "verdict": "BOTH_ACTIVE",
        })
        g.feed_oss_surface_diagnostic({
            "collapsed": False,
            "regime": "normal",
            "signal_quality": 0.62,
            "details": "OSS producing directional signals with moderate quality",
        })
        g.feed_validation_signal_injector({
            "active": False,
            "mode": "idle",
            "injected_count": 0,
            "last_injection": "N/A",
        })
        g.feed_execution_independence_test({
            "independent": True,
            "confidence": 0.88,
            "details": "Execution layer independent",
        })
        g.feed_signal_space_entropy({
            "entropy": 0.72,
            "regime": "diverse",
            "confidence": 0.85,
        })
        g.feed_model_collapse_analyzer({
            "collapse_detected": False,
            "root_cause": "N/A",
            "severity": "none",
            "recommendation": "No action needed",
        })

    run_scenario("hybrid", "HYBRID", _feed_hybrid)

    # ===== Scenario 3: DISCARD_OSS =====================================
    # Both flat
    def _feed_discard(g):
        g.feed_alt_signal_generator({
            "total_signals": 500,
            "buy_pct": 5.0,
            "sell_pct": 5.0,
            "flat_pct": 90.0,
            "mean_confidence": 0.10,
            "signal_std": 0.30,
        })
        g.feed_signal_duality_engine({
            "total_observations": 1000,
            "global_agreement_rate": 0.95,
            "global_divergence_rate": 0.05,
            "global_oss_flat_rate": 0.95,
            "global_alt_flat_rate": 0.92,
            "verdict": "BOTH_FLAT",
        })
        g.feed_oss_surface_diagnostic({
            "collapsed": True,
            "regime": "degenerate",
            "signal_quality": 0.05,
            "details": "OSS completely collapsed",
        })
        g.feed_validation_signal_injector({
            "active": True,
            "mode": "fallback",
            "injected_count": 300,
            "last_injection": "2026-06-30T10:00:00",
        })
        g.feed_execution_independence_test({
            "independent": True,
            "confidence": 0.95,
            "details": "Execution layer independent",
        })
        g.feed_signal_space_entropy({
            "entropy": 0.12,
            "regime": "degenerate",
            "confidence": 0.91,
        })
        g.feed_model_collapse_analyzer({
            "collapse_detected": True,
            "root_cause": "complete_information_loss",
            "severity": "critical",
            "recommendation": "Full pipeline redesign required",
        })

    run_scenario("discard_oss", "DISCARD_OSS", _feed_discard)

    # ===== Scenario 4: REPAIR_OSS ======================================
    # OSS not collapsed but low quality
    def _feed_repair(g):
        g.feed_alt_signal_generator({
            "total_signals": 600,
            "buy_pct": 20.0,
            "sell_pct": 25.0,
            "flat_pct": 55.0,
            "mean_confidence": 0.35,
            "signal_std": 0.55,
        })
        g.feed_signal_duality_engine({
            "total_observations": 1200,
            "global_agreement_rate": 0.40,
            "global_divergence_rate": 0.60,
            "global_oss_flat_rate": 0.60,
            "global_alt_flat_rate": 0.45,
            "verdict": "CONFLICTING",
        })
        g.feed_oss_surface_diagnostic({
            "collapsed": False,
            "regime": "mixed",
            "signal_quality": 0.30,
            "details": "OSS producing weak signals, high noise",
        })
        g.feed_validation_signal_injector({
            "active": False,
            "mode": "idle",
            "injected_count": 0,
            "last_injection": "N/A",
        })
        g.feed_execution_independence_test({
            "independent": True,
            "confidence": 0.80,
            "details": "Execution layer independent",
        })
        g.feed_signal_space_entropy({
            "entropy": 0.55,
            "regime": "moderate",
            "confidence": 0.70,
        })
        g.feed_model_collapse_analyzer({
            "collapse_detected": False,
            "root_cause": "N/A",
            "severity": "none",
            "recommendation": "Review training regime",
        })

    run_scenario("repair_oss", "REPAIR_OSS", _feed_repair)

    # ===== Scenario 5: INCONCLUSIVE (missing data) ======================
    def _feed_inconclusive(g):
        # Only feed one source
        g.feed_alt_signal_generator({
            "total_signals": 200,
            "flat_pct": 30.0,
            "mean_confidence": 0.60,
        })

    run_scenario("inconclusive_missing_data", "INCONCLUSIVE", _feed_inconclusive)

    # ===== Scenario 6: Readiness checks =================================
    gate_r = DecisionRestorationGate("selftest_readiness")
    checklist = gate_r.get_readiness_checklist()
    assert not gate_r.is_ready(), "Gate should NOT be ready before any feeds"
    assert all(v is False for v in checklist.values()), \
        "All checklist entries should be False initially"
    logger.info("  [PASS] %-30s → is_ready=False, all sources unchecked", "readiness_before")

    gate_r.feed_alt_signal_generator({"flat_pct": 10.0})
    assert not gate_r.is_ready(), "Gate should still NOT be ready after 1 of 7 feeds"
    logger.info("  [PASS] %-30s → is_ready=False after 1/7 feeds", "readiness_partial")

    # Feed remaining 6
    gate_r.feed_signal_duality_engine({"verdict": "BOTH_ACTIVE"})
    gate_r.feed_oss_surface_diagnostic({"collapsed": False})
    gate_r.feed_validation_signal_injector({"active": False})
    gate_r.feed_execution_independence_test({"independent": True})
    gate_r.feed_signal_space_entropy({"entropy": 0.5})
    gate_r.feed_model_collapse_analyzer({"collapse_detected": False})
    assert gate_r.is_ready(), "Gate SHOULD be ready after all 7 feeds"
    logger.info("  [PASS] %-30s → is_ready=True after 7/7 feeds", "readiness_full")

    # Reset
    gate_r.reset()
    assert not gate_r.is_ready(), "Gate should NOT be ready after reset"
    logger.info("  [PASS] %-30s → is_ready=False after reset", "readiness_after_reset")

    passed += 4  # readiness checks

    # ===== Summary =====================================================
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
