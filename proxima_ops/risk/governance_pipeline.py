"""Governance Pipeline — Unified Forecaster + Classifier + Governor.

Integrates ClusterGeometryForecaster (pre-RFE regime prediction) with
RegimeTimeScaleClassifier and ExecutionGovernor without modifying any
existing module.

Flow
----
cluster_states → GeometryForecaster → geo forecasts
                                    ↓
                  For each symbol: resolve meta-type
                      - PRE_COLLAPSE → force SLOW_DISSOLUTION
                      - FAST_INSTABILITY → force FAST_TRANSITION
                      - else → defer to pressure-based classifier
                                    ↓
                  Governor with resolved meta-type params
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from .cluster_geometry_forecaster import (
    ClusterGeometryForecaster,
    PreRegimeType,
)
from .execution_governor import GovernorState
from .regime_classifier import (
    GovernorParameterMapper,
    RegimeMetaType,
    RegimeTimeScaleClassifier,
)
from .signal_manifold import symbol_to_primary_cluster

logger = logging.getLogger("proxima_ops.risk.governance_pipeline")

GEO_OVERRIDE_CONFIDENCE_THRESHOLD: float = 0.50
"""Minimum geometry confidence to override pressure-based classification."""

FORCED_REGIME_MAP: dict = {
    PreRegimeType.PRE_COLLAPSE: RegimeMetaType.SLOW_DISSOLUTION,
    PreRegimeType.FAST_INSTABILITY: RegimeMetaType.FAST_TRANSITION,
}


def _resolve_meta_type(
    symbol: str,
    geo_forecasts: Dict[str, Any],
    classifier_meta: str,
    classifier_conf: float,
) -> dict:
    """Resolve symbol meta-type by checking geometry override rules.
    
    Returns
    -------
    dict with keys: meta_type, source, override_active, confidence, lead_estimate
    """
    cluster = symbol_to_primary_cluster(symbol)
    cluster_forecast = geo_forecasts.get("forecasts", {}).get(cluster, {})
    geo_regime = cluster_forecast.get("pre_regime", PreRegimeType.STRUCTURAL_STABILITY)
    geo_conf = cluster_forecast.get("confidence", 0.0)
    lead = cluster_forecast.get("lead_estimate", 0)

    # Rule 1-2: PRE_COLLAPSE / FAST_INSTABILITY with sufficient confidence → override
    if geo_regime in FORCED_REGIME_MAP and geo_conf >= GEO_OVERRIDE_CONFIDENCE_THRESHOLD:
        forced = FORCED_REGIME_MAP[geo_regime]
        override_conf = round(max(0.6, geo_conf), 4)
        return {
            "meta_type": forced,
            "source": f"geometry_{geo_regime}",
            "override_active": True,
            "confidence": override_conf,
            "lead_estimate": lead,
        }

    # Rule 3: EARLY_EXPANSION → boost classifier confidence
    if geo_regime == PreRegimeType.EARLY_EXPANSION:
        boosted = round(min(1.0, classifier_conf + 0.1), 4)
        return {
            "meta_type": classifier_meta,
            "source": "classifier_boosted_by_geometry",
            "override_active": False,
            "confidence": boosted,
            "lead_estimate": 0,
        }

    # Rule 4-5: STRUCTURAL_STABILITY or low confidence → defer to classifier
    return {
        "meta_type": classifier_meta,
        "source": "classifier",
        "override_active": False,
        "confidence": classifier_conf,
        "lead_estimate": 0,
    }


class GovernancePipeline:
    """Unified governance pipeline.
    
    Wraps ClusterGeometryForecaster + RegimeTimeScaleClassifier without
    modifying either module.
    """

    def __init__(
        self,
        geometry_forecaster: Optional[ClusterGeometryForecaster] = None,
        classifier: Optional[RegimeTimeScaleClassifier] = None,
    ) -> None:
        self.geometry = geometry_forecaster or ClusterGeometryForecaster()
        self.classifier = classifier or RegimeTimeScaleClassifier()
        self._history: Dict[str, List[dict]] = defaultdict(list)

    def evaluate(
        self,
        cluster_states: Dict[str, Any],
        rfe_output: Dict[str, Any],
        price_history: Optional[Dict[str, List[float]]] = None,
    ) -> Dict[str, Any]:
        """Full governance pipeline evaluation.
        
        Parameters
        ----------
        cluster_states : dict
            Current cluster states from SignalManifoldProjector.
        rfe_output : dict
            Output from RFEArbitrationLayer.evaluate().
        price_history : dict, optional
            Symbol → list of prices.
        
        Returns
        -------
        dict with decisions, summary, regime_classifications, geometry_forecasts,
             pipeline_info, timestamp.
        """
        # Step 1: Run geometry forecaster (pre-RFE regime prediction)
        geo_result = self.geometry.evaluate(cluster_states)
        geo_forecasts = geo_result.get("forecasts", {})
        logger.info("[GOVPIPE_DEBUG] Step1 geometry_forecaster: clusters=%s",
                    {k: {"pre_regime": v.get("pre_regime", "N/A"), "confidence": v.get("confidence", 0)}
                     for k, v in geo_forecasts.items()})

        # Step 2: Run classifier (pressure-based regime classification)
        classifier_result = self.classifier.evaluate(rfe_output, price_history)
        # Boundary assertion: classifier result must have expected keys
        expected_keys = {"decisions", "regime_classifications", "timestamp"}
        missing = expected_keys - set(classifier_result.keys())
        if missing:
            logger.warning(
                "[CONTRACT_VIOLATION] RegimeTimeScaleClassifier.evaluate() missing "
                "expected keys: %s. rfe_output keys: %s, price_history type: %s",
                missing, list(rfe_output.keys()),
                type(price_history).__name__ if price_history else "None",
            )
        classifier_regimes = classifier_result.get("regime_classifications", {})

        # Step 3: Resolve conflicts per symbol
        decisions: Dict[str, dict] = {}
        regime_classifications: dict = {}
        all_states: list = []
        trades_pending_exit: list = []
        pipeline_overrides: list = []

        classifier_decisions = classifier_result.get("decisions", {})
        for symbol in sorted(classifier_decisions.keys()):
            cd = classifier_decisions[symbol]
            cr = classifier_regimes.get(symbol, {})

            classifier_meta = cr.get("meta_type", RegimeMetaType.STABLE_FLOW)
            classifier_conf = cr.get("confidence", 0.5)

            resolved = _resolve_meta_type(
                symbol, geo_result, classifier_meta, classifier_conf
            )
            meta_type = resolved["meta_type"]
            adjustments = self.classifier.param_mapper.get_params(meta_type)

            logger.info(
                "[GOVPIPE_DEBUG] symbol=%s classifier_meta=%s classifier_conf=%.4f "
                "geo_override=%s resolved_meta=%s source=%s override_active=%s",
                symbol, classifier_meta, classifier_conf,
                resolved.get("override_active", False),
                meta_type, resolved.get("source", "?"),
                resolved.get("override_active", False)
            )
            if not resolved.get("override_active", False) and classifier_conf < GEO_OVERRIDE_CONFIDENCE_THRESHOLD:
                logger.info(
                    "[GOVPIPE_DEBUG] RULE_CHECK symbol=%s rule=geo_override condition=classifier_conf=%.4f < threshold=%.2f "
                    "result=NO_OVERRIDE defer_to=classifier(%s)",
                    symbol, classifier_conf, GEO_OVERRIDE_CONFIDENCE_THRESHOLD, classifier_meta
                )

            if resolved["override_active"]:
                pipeline_overrides.append({
                    "symbol": symbol,
                    "classifier_regime": classifier_meta,
                    "overridden_to": meta_type,
                    "source": resolved["source"],
                    "confidence": resolved["confidence"],
                    "lead_estimate": resolved["lead_estimate"],
                })

            # Apply resolved regime to the per-symbol governor
            gov = self.classifier.get_governor(symbol)
            if gov is not None:
                GovernorParameterMapper.apply_to_governor(gov, adjustments)

                # Re-run governor with new regime params
                ev = rfe_output.get("evaluations", {}).get(symbol, {})
                single_rfe = {
                    "evaluations": {symbol: ev},
                    "summary": rfe_output.get("summary", {}),
                    "transitions": {}, "temporal": {}, "breaches": [],
                    "timestamp": rfe_output.get("timestamp", ""),
                }
                single_result = gov.evaluate(single_rfe, price_history)
                revised_decision = single_result["decisions"][symbol]
                logger.info(
                    "[GOVPIPE_DEBUG] governor_eval symbol=%s governor_state=%s rfe_pressure=%.4f "
                    "action_type=%s adjustments=%s",
                    symbol, revised_decision.get("governor_state", "N/A"),
                    revised_decision.get("rfe_pressure", 0),
                    revised_decision.get("action", {}).get("type", "NONE"),
                    adjustments
                )
            else:
                revised_decision = cd
                logger.info(
                    "[GOVPIPE_DEBUG] governor_eval symbol=%s no_gov_found using_classifier_decision=%s",
                    symbol, cd.get("action", {}).get("type", "NONE")
                )

            decisions[symbol] = revised_decision
            regime_classifications[symbol] = {
                "meta_type": meta_type,
                "confidence": resolved["confidence"],
                "source": resolved["source"],
                "override_active": resolved["override_active"],
                "classifier_meta": classifier_meta,
                "cluster": symbol_to_primary_cluster(symbol),
                "lead_estimate": resolved["lead_estimate"],
                "adjustments": adjustments,
            }
            all_states.append(revised_decision.get("governor_state", "HOLD"))

            if revised_decision.get("action", {}).get("type") in ("CLOSE", "CLOSE_PARTIAL"):
                trades_pending_exit.append(symbol)

        # Step 4: Build summary
        max_index = max(
            (GovernorState.index(s) for s in all_states),
            default=0,
        )
        max_state = GovernorState.ORDER[max_index] if all_states else GovernorState.HOLD

        if max_index >= GovernorState.index(GovernorState.EXIT):
            system_safety = "UNSAFE"
        elif max_index >= GovernorState.index(GovernorState.CONDITIONAL_EXIT):
            system_safety = "WARNING"
        else:
            system_safety = "SAFE"

        meta_type_counts: Dict[str, int] = defaultdict(int)
        for rc in regime_classifications.values():
            meta_type_counts[rc["meta_type"]] += 1
        dominant_regime = max(meta_type_counts, key=meta_type_counts.get) if meta_type_counts else "UNKNOWN"

        logger.info(
            "[GOVPIPE_DEBUG] pipeline_summary symbols=%d max_state=%s system_safety=%s "
            "dominant_regime=%s overrides=%d trades_pending_exit=%d",
            len(decisions), max_state, system_safety,
            dominant_regime, len(pipeline_overrides), len(trades_pending_exit)
        )
        if len(decisions) == 0:
            logger.info(
                "[GOVPIPE_DEBUG] PIPELINE_BLOCKING: no symbols in classifier_decisions empty=%s",
                not bool(classifier_decisions)
            )

        summary = {
            "any_exit_allowed": max_index >= GovernorState.index(GovernorState.CONDITIONAL_EXIT),
            "max_governor_state": max_state,
            "trades_pending_exit": trades_pending_exit,
            "system_safety": system_safety,
            "dominant_regime": dominant_regime,
            "regime_diversity": dict(meta_type_counts),
        }

        return {
            "decisions": decisions,
            "summary": summary,
            "regime_classifications": regime_classifications,
            "geometry_forecasts": geo_forecasts,
            "pipeline_info": {
                "overrides": pipeline_overrides,
                "override_count": len(pipeline_overrides),
            },
            "timestamp": datetime.now().isoformat(),
        }

    def reset(self) -> None:
        self.geometry.reset()
        self.classifier.reset()
        self._history.clear()


def format_pipeline_dashboard(result: dict) -> str:
    """Render the governance pipeline dashboard."""
    lines: list = []
    lines.append("")
    lines.append("GOVERNANCE PIPELINE — FORECASTER + CLASSIFIER + GOVERNOR")
    lines.append("=" * 78)

    decisions = result.get("decisions", {})
    summary = result.get("summary", {})
    regime_classifications = result.get("regime_classifications", {})
    geo_forecasts = result.get("geometry_forecasts", {})
    pipeline_info = result.get("pipeline_info", {})

    if not decisions:
        lines.append("  (no trades evaluated)")
        lines.append("")
        lines.append("=" * 78)
        return "\n".join(lines)

    header = (
        f"{'Trade':<14s} {'Geo-Regime':<20s} {'Override':<10s} "
        f"{'Regime':<18s} {'GState':<14s} {'Exit?':<6s}"
    )
    lines.append(header)
    lines.append("-" * 78)

    for symbol in sorted(decisions.keys()):
        d = decisions[symbol]
        rc = regime_classifications.get(symbol, {})

        cluster = symbol_to_primary_cluster(symbol)
        cluster_forecast = geo_forecasts.get(cluster, {})
        geo_str = cluster_forecast.get("pre_regime", "N/A")[:18]

        override_str = "YES→" + rc.get("meta_type", "")[:8] if rc.get("override_active") else "No"
        regime_str = rc.get("meta_type", "N/A")[:16]
        gstate = d.get("governor_state", "N/A")
        exit_flag = "YES" if d.get("action", {}).get("type") in ("CLOSE", "CLOSE_PARTIAL") else "NO"

        lines.append(
            f"{symbol:<14s} {geo_str:<20s} {override_str:<10s} "
            f"{regime_str:<18s} {gstate:<14s} {exit_flag:<6s}"
        )

    lines.append("")
    lines.append(f"SYSTEM SAFETY: {summary.get('system_safety', 'UNKNOWN')}")
    lines.append(f"Pipeline overrides: {pipeline_info.get('override_count', 0)}")
    for ov in pipeline_info.get("overrides", []):
        lines.append(
            f"  {ov['symbol']}: classifier {ov['classifier_regime']} → "
            f"{ov['overridden_to']} ({ov['source']}, lead ~{ov.get('lead_estimate', '?')} cycles)"
        )

    lines.append("")
    lines.append("REGIME DISTRIBUTION:")
    for regime, count in summary.get("regime_diversity", {}).items():
        lines.append(f"  {regime}: {count}")
    lines.append("")
    lines.append("=" * 78)
    return "\n".join(lines)
