"""Market Observability Filter — Pre-Perception Observability Gate.

Sits ABOVE the entire pipeline as a pre-perception filter:
    MOF -> Signal -> Geometry -> Classifier -> Governor -> Bridge

Classifies whether the market is currently interpretable at all, BEFORE
any system logic runs.

Distinguishes between:
- Low-volatility regime (valid market state)
- Low-information regime (invalid observation state)
"""

from __future__ import annotations

import json
import logging
import os
from collections import Counter
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ObservabilityState(Enum):
    INFORMATION_RICH = "INFORMATION_RICH"
    STRUCTURE_LIMITED = "STRUCTURE_LIMITED"
    INFORMATION_DEGRADED = "INFORMATION_DEGRADED"


class ActionPermission(Enum):
    FULL = "FULL"
    REDUCED = "REDUCED"
    BLOCKED = "BLOCKED"
    ALLOW_WITH_WARNING = "ALLOW_WITH_WARNING"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_CONFIDENCE_THRESHOLD = 0.15
"""Below this is noise floor."""

MEAN_COHERENCE_THRESHOLD = 0.20
"""Below this indicates structure loss."""

ENTROPY_CEILING = 0.85
"""Above this the signal is too random."""

DEGENERATE_CLUSTER_COHERENCE_THRESHOLD = 0.05
"""Clusters with coherence below this are degenerate."""

DEGENERATE_CLUSTER_RATIO_FORCE_DEGRADED = 0.5
"""Force INFORMATION_DEGRADED when ratio exceeds this."""

MIN_CONFIDENCE_FORCE_DEGRADED = 0.10
"""Force INFORMATION_DEGRADED when mean confidence is below this."""

INFORMATION_RICH_THRESHOLD = 0.65
STRUCTURE_LIMITED_THRESHOLD = 0.35

OBSERVABILITY_WEIGHT_COHERENCE = 0.70
OBSERVABILITY_WEIGHT_CONFIDENCE = 0.20
OBSERVABILITY_WEIGHT_STABILITY = 0.10

HISTOGRAM_BINS = [i / 10.0 for i in range(11)]
"""Confidence histogram bins: [0.0, 0.1, ..., 1.0]."""


# ---------------------------------------------------------------------------
# Market Observability Filter
# ---------------------------------------------------------------------------


class MarketObservabilityFilter:
    """Pre-perception filter that classifies market observability state.

    Evaluates OSS signal quality, cluster coherence health, and signal
    entropy to determine whether the market is currently interpretable.

    Usage
    -----
        mof = MarketObservabilityFilter()
        result = mof.evaluate(cluster_states, signals_data)
        # result["observability_state"] -> ObservabilityState
        # result["action_permission"] -> ActionPermission
    """

    def __init__(self, bootstrap_mode: bool = False) -> None:
        """
        Initialize MOF.

        Parameters
        ----------
        bootstrap_mode : bool
            If True, INFORMATION_DEGRADED does not block execution but logs
            a WARNING instead. Intended ONLY for controlled bootstrap cycles
            to break cold-start degeneracy. Default: False.
        """
        self._bootstrap_mode = bootstrap_mode
        self._last_result: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        cluster_states: Dict[str, Dict[str, Any]],
        signals_data: List[Dict[str, Any]],
        lifecycle_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Evaluate market observability from cluster + signal data.

        Parameters
        ----------
        cluster_states : dict
            Current cluster states from SignalManifoldProjector.
            Each value must contain at least 'coherence' (float).
        signals_data : list of dict
            Raw OSS signal dicts. Each must contain:
                - symbol (str)
                - direction (int): -1, 0, +1
                - confidence (float): 0..1
                - ecdf (float, optional)
                - drift (int, optional)
        lifecycle_data : dict, optional
            Lifecycle state dict (from lifecycle_state.json) for trade
            continuity analysis. If None, attempts to load from default path.

        Returns
        -------
        dict with keys:
            - observability_state (str): ObservabilityState value
            - action_permission (str): ActionPermission value
            - observability_score (float): composite score 0..1
            - oss_confidence_quality (dict): confidence metrics
            - coherence_health (dict): coherence metrics
            - signal_entropy (dict): entropy metrics
            - trade_continuity (dict): trade continuity metrics
            - latency_drift (dict): latency drift metrics
            - symbol_entropy (dict): symbol entropy metrics
            - components (dict): raw component values
            - force_triggers (list): force-degraded trigger reasons
            - timestamp (str): ISO timestamp
        """
        # A. OSS Confidence Quality
        confidence_quality = self._evaluate_oss_confidence(signals_data)

        # B. Cluster Coherence Health
        coherence_health = self._evaluate_coherence_health(cluster_states)

        # C. Signal Entropy
        signal_entropy = self._evaluate_signal_entropy(signals_data)

        # D. New Observability Dimensions (diagnostic only)
        trade_continuity = self._evaluate_trade_continuity(lifecycle_data)
        latency_drift = self._evaluate_latency_drift(signals_data)
        symbol_entropy = self._evaluate_symbol_entropy(signals_data)

        # E. Observability Score (composite — original 3 components only)
        observability_score = self._compute_observability_score(
            coherence_health["quality"],
            confidence_quality["quality"],
            signal_entropy["stability_quality"],
        )

        # State classification
        force_triggers: List[str] = []
        degenerate_ratio = coherence_health["degenerate_cluster_ratio"]
        mean_confidence = confidence_quality["mean_oss_confidence"]

        if degenerate_ratio > DEGENERATE_CLUSTER_RATIO_FORCE_DEGRADED:
            force_triggers.append(
                f"degenerate_cluster_ratio={degenerate_ratio:.3f} > {DEGENERATE_CLUSTER_RATIO_FORCE_DEGRADED}"
            )

        if mean_confidence < MIN_CONFIDENCE_FORCE_DEGRADED:
            force_triggers.append(
                f"mean_oss_confidence={mean_confidence:.4f} < {MIN_CONFIDENCE_FORCE_DEGRADED}"
            )

        if force_triggers:
            observability_state = ObservabilityState.INFORMATION_DEGRADED
            # In bootstrap mode, allow execution with warning instead of blocking
            if self._bootstrap_mode:
                action_permission = ActionPermission.ALLOW_WITH_WARNING
                logger.warning(
                    "[MOF_GATE_BOOTSTRAP] INFORMATION_DEGRADED — allowing execution with warning "
                    f"(observability_score={observability_score:.4f}, "
                    f"degenerate_ratio={degenerate_ratio:.3f})"
                )
            else:
                action_permission = ActionPermission.BLOCKED
        elif observability_score >= INFORMATION_RICH_THRESHOLD:
            observability_state = ObservabilityState.INFORMATION_RICH
            action_permission = ActionPermission.FULL
        elif observability_score >= STRUCTURE_LIMITED_THRESHOLD:
            observability_state = ObservabilityState.STRUCTURE_LIMITED
            action_permission = ActionPermission.REDUCED
        else:
            observability_state = ObservabilityState.INFORMATION_DEGRADED
            # In bootstrap mode, allow execution with warning instead of blocking
            if self._bootstrap_mode:
                action_permission = ActionPermission.ALLOW_WITH_WARNING
                logger.warning(
                    "[MOF_GATE_BOOTSTRAP] INFORMATION_DEGRADED — allowing execution with warning "
                    f"(observability_score={observability_score:.4f})"
                )
            else:
                action_permission = ActionPermission.BLOCKED

        result: Dict[str, Any] = {
            "observability_state": observability_state.value,
            "action_permission": action_permission.value,
            "observability_score": round(observability_score, 4),
            "oss_confidence_quality": confidence_quality,
            "coherence_health": coherence_health,
            "signal_entropy": signal_entropy,
            "trade_continuity": trade_continuity,
            "latency_drift": latency_drift,
            "symbol_entropy": symbol_entropy,
            "components": {
                # Original 3 — drive the observability_score
                "coherence_quality": round(coherence_health["quality"], 4),
                "oss_confidence_quality": round(confidence_quality["quality"], 4),
                "stability_quality": round(signal_entropy["stability_quality"], 4),
                # New diagnostic dimensions
                "trade_continuity_quality": round(trade_continuity["quality"], 4),
                "latency_drift_quality": round(latency_drift["quality"], 4),
                "symbol_entropy_quality": round(symbol_entropy["quality"], 4),
            },
            "force_triggers": force_triggers,
        }

        self._last_result = result
        return result

    def get_last_result(self) -> Optional[Dict[str, Any]]:
        """Return the most recent ``evaluate()`` output."""
        return self._last_result

    # ------------------------------------------------------------------
    # Component Evaluators
    # ------------------------------------------------------------------

    @staticmethod
    def _evaluate_oss_confidence(
        signals_data: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Evaluate OSS signal confidence quality.

        Returns dict with:
            - mean_oss_confidence (float)
            - oss_confidence_distribution (dict): bin -> count
            - signals_below_threshold (int)
            - total_signals (int)
            - quality (float): 0..1
        """
        if not signals_data:
            return {
                "mean_oss_confidence": 0.0,
                "oss_confidence_distribution": {},
                "signals_below_threshold": 0,
                "total_signals": 0,
                "quality": 0.0,
            }

        confidences = np.array(
            [float(s.get("confidence", 0.0)) for s in signals_data],
            dtype=float,
        )
        mean_conf = float(np.mean(confidences))
        below_threshold = int(np.sum(confidences < MIN_CONFIDENCE_THRESHOLD))
        total = len(confidences)

        # Histogram
        bin_edges = np.array(HISTOGRAM_BINS)
        indices = np.digitize(confidences, bin_edges, right=False) - 1
        indices = np.clip(indices, 0, len(bin_edges) - 2)
        counts = np.bincount(indices, minlength=len(bin_edges) - 1)
        distribution = {
            f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}": int(counts[i])
            for i in range(len(bin_edges) - 1)
            if counts[i] > 0
        }

        # Quality: mean confidence normalized to [0, 1]
        quality = min(1.0, mean_conf)

        return {
            "mean_oss_confidence": round(mean_conf, 4),
            "oss_confidence_distribution": distribution,
            "signals_below_threshold": below_threshold,
            "total_signals": total,
            "quality": round(quality, 4),
        }

    @staticmethod
    def _evaluate_coherence_health(
        cluster_states: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Evaluate cluster coherence health.

        Only clusters with active_symbols > 0 contribute to the degenerate
        cluster ratio — inactive clusters (no data observed) are not counted
        as degenerate. This prevents single-symbol regimes from being
        misclassified as INFORMATION_DEGRADED.

        Returns dict with:
            - coherence_distribution (list): coherence values per cluster
            - degenerate_cluster_count (int): degenerate among *active* clusters
            - total_clusters (int): total cluster count (incl. inactive)
            - active_cluster_count (int): clusters with active_symbols > 0
            - degenerate_cluster_ratio (float): degenerate / active
            - mean_coherence (float): mean over ALL clusters
            - quality (float): 0..1
        """
        if not cluster_states:
            return {
                "coherence_distribution": [],
                "degenerate_cluster_count": 0,
                "total_clusters": 0,
                "active_cluster_count": 0,
                "degenerate_cluster_ratio": 0.0,
                "mean_coherence": 0.0,
                "quality": 0.0,
            }

        all_coherences = [
            float(state.get("coherence", 0.0))
            for state in cluster_states.values()
        ]
        coherence_arr = np.array(all_coherences, dtype=float)
        total = len(coherence_arr)
        mean_coherence = float(np.mean(coherence_arr))

        # ---- Degenerate ratio: only active clusters count ----
        # A cluster is "active" if it has observed data (active_symbols > 0).
        # Clusters with no data are simply unobserved, not degenerate.
        active_count = 0
        degenerate_count = 0
        for state in cluster_states.values():
            active_symbols = int(state.get("active_symbols", 1))  # default 1 for backward compat
            coherence = float(state.get("coherence", 0.0))
            if active_symbols > 0:
                active_count += 1
                if coherence < DEGENERATE_CLUSTER_COHERENCE_THRESHOLD:
                    degenerate_count += 1

        degenerate_ratio = degenerate_count / active_count if active_count > 0 else 0.0

        # Quality: how far mean coherence is above threshold, capped at 1.0
        if mean_coherence >= MEAN_COHERENCE_THRESHOLD:
            quality = min(1.0, mean_coherence)
        else:
            quality = mean_coherence / MEAN_COHERENCE_THRESHOLD

        return {
            "coherence_distribution": [round(c, 4) for c in all_coherences],
            "degenerate_cluster_count": degenerate_count,
            "total_clusters": total,
            "active_cluster_count": active_count,
            "degenerate_cluster_ratio": round(degenerate_ratio, 4),
            "mean_coherence": round(mean_coherence, 4),
            "quality": round(quality, 4),
        }

    @staticmethod
    def _evaluate_signal_entropy(
        signals_data: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Evaluate signal entropy / diversity.

        Returns dict with:
            - signal_diversity (int): unique symbols with non-zero direction
            - cross_symbol_agreement_rate (float): fraction agreeing on majority
            - signal_entropy (float): 1 - agreement_rate
            - stability_quality (float): 0..1
        """
        if not signals_data:
            return {
                "signal_diversity": 0,
                "cross_symbol_agreement_rate": 0.0,
                "signal_entropy": 1.0,
                "stability_quality": 0.0,
            }

        # Get non-neutral signals
        directions = np.array(
            [float(s.get("direction", 0)) for s in signals_data],
            dtype=float,
        )
        non_neutral_mask = directions != 0
        non_neutral_dirs = directions[non_neutral_mask]

        if len(non_neutral_dirs) == 0:
            return {
                "signal_diversity": 0,
                "cross_symbol_agreement_rate": 0.0,
                "signal_entropy": 1.0,
                "stability_quality": 0.0,
            }

        # Signal diversity: unique symbols with non-zero direction
        unique_symbols = set()
        for s in signals_data:
            if s.get("direction", 0) != 0:
                unique_symbols.add(s.get("symbol", ""))
        unique_symbols.discard("")

        # Cross-symbol agreement rate: fraction agreeing with majority direction
        buy_count = int(np.sum(non_neutral_dirs > 0))
        sell_count = int(np.sum(non_neutral_dirs < 0))
        total_non_neutral = len(non_neutral_dirs)
        majority_count = max(buy_count, sell_count)
        agreement_rate = majority_count / total_non_neutral if total_non_neutral > 0 else 0.0

        # Entropy: disagreement fraction
        entropy = 1.0 - agreement_rate

        # Stability quality: inverse of entropy, normalized to ceiling
        if entropy >= ENTROPY_CEILING:
            stability_quality = 0.0
        else:
            stability_quality = 1.0 - (entropy / ENTROPY_CEILING)

        return {
            "signal_diversity": len(unique_symbols),
            "cross_symbol_agreement_rate": round(agreement_rate, 4),
            "signal_entropy": round(entropy, 4),
            "stability_quality": round(stability_quality, 4),
        }

    # ------------------------------------------------------------------
    # New Observability Dimensions (Batch 6.1)
    # ------------------------------------------------------------------

    @staticmethod
    def _evaluate_trade_continuity(
        lifecycle_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Evaluate trade continuity quality.

        Measures how well the system tracks trade lifecycles across time.
        Checks for lifecycle stage consistency, orphan signals, and
        OPENED/CLOSED balance.

        Parameters
        ----------
        lifecycle_data : dict, optional
            Lifecycle state dict. If None, tries to load from default
            ``state/lifecycle_state.json`` path.

        Returns
        -------
        dict with keys:
            - total_signals (int)
            - opened_count (int)
            - closed_count (int)
            - orphan_count (int)
            - orphan_ratio (float)
            - quality (float): 0..1
        """
        if lifecycle_data is None:
            # Attempt to load from default path
            try:
                # Navigate from this file to state/lifecycle_state.json
                _path = os.path.join(
                    os.path.dirname(__file__),
                    "..", "..", "..", "state", "lifecycle_state.json",
                )
                with open(_path, "r") as _f:
                    lifecycle_data = json.load(_f)
            except (FileNotFoundError, json.JSONDecodeError, OSError):
                return {
                    "total_signals": 0,
                    "opened_count": 0,
                    "closed_count": 0,
                    "orphan_count": 0,
                    "orphan_ratio": 0.0,
                    "quality": 0.0,
                }

        signals = lifecycle_data.get("signals", []) if isinstance(lifecycle_data, dict) else []
        if not signals:
            return {
                "total_signals": 0,
                "opened_count": 0,
                "closed_count": 0,
                "orphan_count": 0,
                "orphan_ratio": 0.0,
                "quality": 0.0,
            }

        total = len(signals)
        opened = sum(1 for s in signals if s.get("stage") == "OPENED")
        closed = sum(1 for s in signals if s.get("stage") == "CLOSED")
        orphaned = sum(1 for s in signals if s.get("stage") in ("ORPHANED",))
        other = total - opened - closed - orphaned  # e.g. PENDING, LIMBO

        orphan_ratio = orphaned / total if total > 0 else 0.0

        # Quality: 1.0 if perfect cycle tracking, penalized by:
        #   - orphans (70% weight) — signals that lost their lifecycle
        #   - non-closed, non-opened "stuck" signals (30% weight)
        stuck_ratio = other / total if total > 0 else 0.0
        quality = 1.0 - (orphan_ratio * 0.70 + stuck_ratio * 0.30)
        quality = max(0.0, min(1.0, quality))

        return {
            "total_signals": total,
            "opened_count": opened,
            "closed_count": closed,
            "orphan_count": orphaned,
            "orphan_ratio": round(orphan_ratio, 4),
            "quality": round(quality, 4),
        }

    @staticmethod
    def _evaluate_latency_drift(
        signals_data: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Evaluate execution latency drift.

        Measures the gap between signal generation and execution/rejection
        (how long signals stay in LIMBO/OPENED before being reconciled).

        Parameters
        ----------
        signals_data : list of dict
            Raw OSS signal dicts (must contain ``generated_at`` and
            optionally ``submitted_at`` / ``accepted_at``).

        Returns
        -------
        dict with keys:
            - average_latency_seconds (float)
            - max_latency_seconds (float)
            - signals_with_timestamps (int)
            - quality (float): 0..1
        """
        latencies = []
        for s in signals_data:
            generated = s.get("generated_at")
            accepted = s.get("accepted_at")
            submitted = s.get("submitted_at")

            if generated is not None:
                # Prefer accepted_at as the execution confirmation,
                # fall back to submitted_at if accepted is missing.
                end_ts = accepted if accepted is not None else submitted
                if end_ts is not None:
                    latency = float(end_ts) - float(generated)
                    if latency >= 0:
                        latencies.append(latency)

        if not latencies:
            return {
                "average_latency_seconds": 0.0,
                "max_latency_seconds": 0.0,
                "signals_with_timestamps": 0,
                "quality": 0.5,  # neutral — no timestamps available
            }

        avg_latency = float(np.mean(latencies))
        max_latency = float(np.max(latencies))

        # Quality: exponential decay with average latency.
        #   < 1 s    → ~0.82+
        #   < 5 s    → ~0.37+
        #   > 30 s   → ~0.00
        # Threshold of 5 seconds means at 5s average latency, quality = exp(-1) ≈ 0.368
        threshold = 5.0
        quality = float(np.exp(-avg_latency / threshold))
        quality = max(0.0, min(1.0, quality))

        return {
            "average_latency_seconds": round(avg_latency, 4),
            "max_latency_seconds": round(max_latency, 4),
            "signals_with_timestamps": len(latencies),
            "quality": round(quality, 4),
        }

    @staticmethod
    def _evaluate_symbol_entropy(
        signals_data: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Evaluate symbol participation entropy.

        Measures diversity of symbols being traded using Shannon entropy.
        Low entropy = degeneracy (few symbols dominate).
        Higher entropy = better information surface.

        Parameters
        ----------
        signals_data : list of dict
            Raw OSS signal dicts (each must contain ``symbol``).

        Returns
        -------
        dict with keys:
            - symbol_counts (dict): symbol -> count
            - unique_symbols (int)
            - shannon_entropy (float): raw entropy in bits
            - normalized_entropy (float): entropy / log2(unique_symbols)
            - quality (float): 0..1
        """
        if not signals_data:
            return {
                "symbol_counts": {},
                "unique_symbols": 0,
                "shannon_entropy": 0.0,
                "normalized_entropy": 0.0,
                "quality": 0.0,
            }

        # Count signals per symbol
        symbol_counts = Counter(s.get("symbol", "") for s in signals_data)
        symbol_counts.pop("", None)  # remove blank symbols

        if not symbol_counts:
            return {
                "symbol_counts": {},
                "unique_symbols": 0,
                "shannon_entropy": 0.0,
                "normalized_entropy": 0.0,
                "quality": 0.0,
            }

        total = sum(symbol_counts.values())
        unique = len(symbol_counts)

        # Shannon entropy: H = -sum(p_i * log2(p_i))
        entropy = 0.0
        for count in symbol_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log2(p)

        # Normalize by log2(unique) for max possible entropy
        if unique > 1:
            max_entropy = np.log2(unique)
            normalized = entropy / max_entropy if max_entropy > 0 else 0.0
        else:
            normalized = 0.0  # single symbol = no diversity

        # Quality = normalized entropy (higher = more diverse)
        quality = normalized
        quality = max(0.0, min(1.0, quality))

        return {
            "symbol_counts": dict(symbol_counts.most_common()),
            "unique_symbols": unique,
            "shannon_entropy": round(float(entropy), 4),
            "normalized_entropy": round(float(normalized), 4),
            "quality": round(quality, 4),
        }

    # ------------------------------------------------------------------
    # Observability Score Computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_observability_score(
        coherence_quality: float,
        confidence_quality: float,
        stability_quality: float,
    ) -> float:
        """Compute weighted composite observability score.

        OBSERVABILITY_SCORE = 0.70 * coherence_quality
                           + 0.20 * oss_confidence_quality
                           + 0.10 * stability_quality
        """
        score = (
            OBSERVABILITY_WEIGHT_COHERENCE * coherence_quality
            + OBSERVABILITY_WEIGHT_CONFIDENCE * confidence_quality
            + OBSERVABILITY_WEIGHT_STABILITY * stability_quality
        )
        return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Pipeline Integration Wrapper
# ---------------------------------------------------------------------------


def evaluate_with_mof(
    mof: MarketObservabilityFilter,
    pipeline: "GovernancePipeline",  # noqa: F821
    cluster_states: Dict[str, Any],
    rfe_output: Dict[str, Any],
    signals_data: List[Dict[str, Any]],
    price_history: Optional[Dict[str, List[float]]] = None,
) -> Dict[str, Any]:
    """Run MOF pre-filter, then pipeline if permission allows.

    Parameters
    ----------
    mof : MarketObservabilityFilter
        MOF instance.
    pipeline : GovernancePipeline
        Pipeline instance.
    cluster_states : dict
        Current cluster states.
    rfe_output : dict
        RFEArbitrationLayer output.
    signals_data : list of dict
        Raw OSS signal dicts.
    price_history : dict, optional
        Symbol -> list of prices.

    Returns
    -------
    dict with keys:
        - mof (dict): MOF evaluation result
        - pipeline_blocked (bool)
        - pipeline_result (dict or None): pipeline result if not blocked
    """
    mof_result = mof.evaluate(cluster_states, signals_data)

    if mof_result["action_permission"] == ActionPermission.BLOCKED.value:
        return {
            "mof": mof_result,
            "pipeline_blocked": True,
            "pipeline_result": None,
        }

    pipeline_result = pipeline.evaluate(cluster_states, rfe_output, price_history)
    pipeline_result["mof"] = mof_result
    return {
        "mof": mof_result,
        "pipeline_blocked": False,
        "pipeline_result": pipeline_result,
    }
