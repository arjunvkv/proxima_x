"""Cluster Geometry Forecaster — Pre-RFE Regime Prediction.

Detects regime transitions from cluster geometry (coherence, divergence,
net_direction trajectories) BEFORE they translate into RFE pressure.

Core Insight
------------
The current RegimeTimeScaleClassifier detects regimes from RFE pressure
history — but Slow Decay (D-grade failure) is only visible in pressure
AFTER the recovery dip appears at cycle 13. By then, the governor has
already false-exited at cycle 10.

The signature of Slow Decay is:

    coherence collapse precedes price collapse

Detecting coherence curvature change, divergence acceleration, and
symmetry breakdown can predict Slow Decay 5-10 cycles before it shows
up in RFE pressure.

Four pre-pressure regime classes:
- EARLY_EXPANSION: New directional consensus forming (coherence rising)
- STRUCTURAL_STABILITY: Steady state, no geometry deformation
- PRE_COLLAPSE: Coherence decaying, divergence accelerating — Slow Decay
  precursor
- FAST_INSTABILITY: Sudden coherence drop, rapid divergence spike
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Pre-Regime Type Constants
# ---------------------------------------------------------------------------


class PreRegimeType:
    """Pre-pressure regime types — detected from cluster geometry alone.

    These regimes are precursors that manifest in geometric features
    (coherence, divergence, net_direction) BEFORE they appear in RFE
    pressure data, giving 5-10 cycle lead time for regime adjustment.

    EARLY_EXPANSION
        New directional consensus forming. Coherence is rising, divergence
        shrinking. Indicates a healthy trend-building phase.

    STRUCTURAL_STABILITY
        Steady state with no significant geometry deformation. The cluster
        has stable coherence, low divergence, balanced net direction.

    PRE_COLLAPSE
        Coherence is decaying, divergence accelerating. This is the
        signature of Slow Decay (SLOW_DISSOLUTION) forming. If detected
        early, the regime classifier can switch to SLOW_DISSOLUTION params
        before the governor false-exits.

    FAST_INSTABILITY
        Sudden coherence collapse with rapid divergence spike. Indicates
        an imminent FAST_TRANSITION event — shock, spike, or liquidity
        crisis.
    """

    EARLY_EXPANSION = "EARLY_EXPANSION"
    STRUCTURAL_STABILITY = "STRUCTURAL_STABILITY"
    PRE_COLLAPSE = "PRE_COLLAPSE"
    FAST_INSTABILITY = "FAST_INSTABILITY"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _linear_regression_slope(values: List[float]) -> float:
    """Compute slope of simple linear regression over index vs value.

    Returns 0.0 for degenerate inputs (fewer than 2 points or zero
    denominator).
    """
    n = len(values)
    if n < 2:
        return 0.0
    x_vals = list(range(n))
    x_mean = (n - 1) / 2.0
    y_mean = sum(values) / n

    num = sum((x_vals[i] - x_mean) * (values[i] - y_mean) for i in range(n))
    den = sum((x_vals[i] - x_mean) ** 2 for i in range(n))

    return num / den if den > 0 else 0.0


def _second_derivative(values: List[float]) -> float:
    """Approximate second derivative of a sequence.

    For a sequence [a, b, c, d, e]:
    first differences: [b-a, c-b, d-c, e-d]
    second differences: [(c-b)-(b-a), (d-c)-(c-b), (e-d)-(d-c)]

    Returns the mean of second differences, or 0.0 for short inputs.

    A positive result means the sequence is accelerating upward (convex),
    a negative result means decelerating (concave).
    """
    if len(values) < 3:
        return 0.0

    first_diff = [values[i] - values[i - 1] for i in range(1, len(values))]
    second_diff = [
        first_diff[i] - first_diff[i - 1] for i in range(1, len(first_diff))
    ]

    return sum(second_diff) / len(second_diff) if second_diff else 0.0


# ---------------------------------------------------------------------------
# Cluster <-> Symbol mapping
# ---------------------------------------------------------------------------


# Re-export the symbol_to_primary_cluster from signal_manifold
# to avoid circular imports, we define it inline here.
_CLUSTERS: Dict[str, List[str]] = {
    "EUR": ["EURUSD", "EURJPY", "EURGBP", "EURCHF", "EURAUD", "EURNZD", "EURCAD"],
    "USD": ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD"],
    "JPY": ["USDJPY", "EURJPY", "GBPJPY", "CHFJPY", "AUDJPY", "NZDJPY", "CADJPY"],
    "AUD_NZD": [
        "AUDUSD", "NZDUSD", "AUDJPY", "NZDJPY", "AUDCHF", "NZDCHF",
        "AUDCAD", "NZDCAD", "AUDNZD", "EURAUD", "EURNZD", "GBPAUD", "GBPNZD",
    ],
    "CHF": ["USDCHF", "EURCHF", "GBPCHF", "CHFJPY", "AUDCHF", "NZDCHF", "CADCHF"],
    "GBP": ["GBPUSD", "GBPJPY", "GBPCHF", "GBPAUD", "GBPNZD", "GBPCAD", "EURGBP"],
    "CAD": ["USDCAD", "CADJPY", "CADCHF", "AUDCAD", "NZDCAD", "GBPCAD", "EURCAD"],
}


def symbol_to_primary_cluster(symbol: str) -> str:
    """Return the first cluster that *symbol* belongs to."""
    for cname, members in _CLUSTERS.items():
        if symbol in members:
            return cname
    return "USD"


# ---------------------------------------------------------------------------
# GeometryTracker
# ---------------------------------------------------------------------------


class GeometryTracker:
    """Tracks cluster geometry history for regime forecasting.

    Maintains rolling window of cluster state snapshots for each cluster:
    - coherence_history: rolling list (max 30 entries)
    - divergence_history: rolling list
    - net_direction_history: rolling list

    Parameters
    ----------
    max_history : int
        Maximum number of snapshots to retain per cluster (default 30).
    """

    def __init__(self, max_history: int = 30) -> None:
        self.max_history = max_history

        # Per-cluster history: cluster_name -> {field: [values]}
        self._history: Dict[str, Dict[str, List[float]]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, cluster_states: Dict[str, Dict[str, Any]]) -> None:
        """Append current cluster state snapshot to each cluster's history.

        Parameters
        ----------
        cluster_states : dict
            Mapping of cluster name -> cluster state dict.
            Expected keys per cluster: 'coherence', 'divergence',
            'net_direction'.
        """
        for cluster, state in cluster_states.items():
            if cluster not in self._history:
                self._history[cluster] = {
                    "coherence": [],
                    "divergence": [],
                    "net_direction": [],
                }

            h = self._history[cluster]

            # Append each field, enforcing max history length
            for field in ("coherence", "divergence", "net_direction"):
                val = state.get(field, 0.0)
                h[field].append(val)
                if len(h[field]) > self.max_history:
                    h[field].pop(0)

    def get_history(
        self, cluster: str, field: str = "coherence"
    ) -> List[float]:
        """Return the full history for a cluster field."""
        return list(self._history.get(cluster, {}).get(field, []))

    def get_curvature(
        self, cluster: str, field: str = "coherence", window: int = 5
    ) -> float:
        """Second derivative of field over window.

        How the acceleration of the field is changing.

        For coherence:
        - Positive = coherence accelerating upward (expansion strengthening)
        - Negative = coherence decelerating (collapse starting)

        Parameters
        ----------
        cluster : str
            Cluster name (e.g. 'JPY', 'EUR').
        field : str
            Field to compute curvature for ('coherence', 'divergence',
            'net_direction').
        window : int
            Number of recent entries to analyse (default 5).

        Returns
        -------
        float
            Second derivative approximation. Returns 0.0 if insufficient
            history.
        """
        series = self._history.get(cluster, {}).get(field, [])
        if len(series) < max(3, window):
            return 0.0
        relevant = series[-window:]
        return _second_derivative(relevant)

    def get_acceleration(
        self, cluster: str, field: str = "divergence", window: int = 5
    ) -> float:
        """First derivative of first derivative = how fast the field is
        ACCELERATING.

        For divergence:
        - High positive = rapid disagreement growth (PRE_COLLAPSE signal)
        - Near zero = stable divergence
        - Negative = divergence shrinking (consensus building)

        This is identical to curvature but semantically focused on
        divergence acceleration. Implementation applies second derivative
        to the field's recent window.

        Parameters
        ----------
        cluster : str
            Cluster name.
        field : str
            Field to analyse (default 'divergence').
        window : int
            Recent window size (default 5).

        Returns
        -------
        float
            Acceleration estimate. 0.0 if insufficient history.
        """
        series = self._history.get(cluster, {}).get(field, [])
        if len(series) < max(3, window):
            return 0.0
        relevant = series[-window:]
        return _second_derivative(relevant)

    def get_symmetry_imbalance(
        self, cluster: str, window: int = 5
    ) -> float:
        """Skew of net_direction over window.

        0 = perfectly balanced (no directional bias shift)
        Positive = bullish skew (net direction trending positive)
        Negative = bearish skew (net direction trending negative)

        High |value| with dropping coherence = structural asymmetry,
        a PRE_COLLAPSE indicator.

        Parameters
        ----------
        cluster : str
            Cluster name.
        window : int
            Recent window size (default 5).

        Returns
        -------
        float
            Mean net_direction over the window. Returns 0.0 if
            insufficient history.
        """
        series = self._history.get(cluster, {}).get("net_direction", [])
        if len(series) < window:
            return 0.0
        relevant = series[-window:]
        return sum(relevant) / len(relevant)

    def get_coherence_slope(
        self, cluster: str, window: int = 5
    ) -> float:
        """Linear regression slope of coherence over window.

        Negative = coherence declining over time.
        Positive = coherence increasing.

        Parameters
        ----------
        cluster : str
            Cluster name.
        window : int
            Recent window size (default 5).

        Returns
        -------
        float
            Slope of coherence. 0.0 if insufficient history.
        """
        series = self._history.get(cluster, {}).get("coherence", [])
        if len(series) < window:
            return 0.0
        relevant = series[-window:]
        return _linear_regression_slope(relevant)

    @property
    def known_clusters(self) -> List[str]:
        """Return the list of clusters that have history."""
        return list(self._history.keys())

    def reset(self) -> None:
        """Clear all tracked history."""
        self._history.clear()


# ---------------------------------------------------------------------------
# GeometryClassifier
# ---------------------------------------------------------------------------


class GeometryClassifier:
    """Classifies pre-pressure regime from geometric features.

    Decision rules (applied per-cluster):

    PRE_COLLAPSE (SLOW_DISSOLUTION precursor):
    - coherence_curvature < -0.05 (coherence decelerating)
    - divergence_acceleration > 0.05 (disagreement accelerating)
    - coherence_slope < -0.02 (coherence trending down)

    FAST_INSTABILITY:
    - coherence_curvature < -0.15 (sharp deceleration)
    - divergence_acceleration > 0.15 (rapid disagreement growth)
    - abs(symmetry_imbalance) > 0.3 (strong asymmetry)

    EARLY_EXPANSION:
    - coherence_curvature > 0.05 (coherence accelerating upward)
    - coherence_slope > 0.02 (coherence trending up)

    STRUCTURAL_STABILITY (default):
    - None of the above conditions met

    Thresholds can be overridden at construction time.
    """

    def __init__(
        self,
        pre_collapse_curvature_threshold: float = -0.05,
        pre_collapse_acceleration_threshold: float = 0.05,
        pre_collapse_slope_threshold: float = -0.02,
        fast_curvature_threshold: float = -0.15,
        fast_acceleration_threshold: float = 0.15,
        fast_asymmetry_threshold: float = 0.3,
        expansion_curvature_threshold: float = 0.05,
        expansion_slope_threshold: float = 0.02,
    ) -> None:
        self._pc_curv = pre_collapse_curvature_threshold
        self._pc_accel = pre_collapse_acceleration_threshold
        self._pc_slope = pre_collapse_slope_threshold
        self._fi_curv = fast_curvature_threshold
        self._fi_accel = fast_acceleration_threshold
        self._fi_asym = fast_asymmetry_threshold
        self._ee_curv = expansion_curvature_threshold
        self._ee_slope = expansion_slope_threshold

    def classify(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Classify pre-pressure regime from geometric features.

        Parameters
        ----------
        features : dict
            Must contain:
            - coherence_curvature (float)
            - divergence_acceleration (float)
            - symmetry_imbalance (float)
            - coherence_slope (float)

        Returns
        -------
        dict
            - pre_regime (str): PreRegimeType constant
            - confidence (float): 0.0 to 1.0
            - trigger_features (dict): features that triggered the
              classification
        """
        cc = features.get("coherence_curvature", 0.0)
        da = features.get("divergence_acceleration", 0.0)
        si = features.get("symmetry_imbalance", 0.0)
        cs = features.get("coherence_slope", 0.0)

        # Check FAST_INSTABILITY FIRST (higher severity — overrides PRE_COLLAPSE)
        # Because FAST_INSTABILITY features often also satisfy PRE_COLLAPSE
        # thresholds, we must check the more severe condition first.
        conditions_met = sum(
            [
                cc < self._fi_curv,
                da > self._fi_accel,
                abs(si) > self._fi_asym,
            ]
        )
        if conditions_met >= 2:
            confidence = self._compute_fast_confidence(cc, da, si)
            return {
                "pre_regime": PreRegimeType.FAST_INSTABILITY,
                "confidence": round(confidence, 4),
                "trigger_features": {
                    "coherence_curvature": cc,
                    "divergence_acceleration": da,
                    "symmetry_imbalance": si,
                },
            }

        # Check PRE_COLLAPSE (slow dissolution precursor)
        if cc < self._pc_curv and da > self._pc_accel and cs < self._pc_slope:
            confidence = self._compute_pre_collapse_confidence(
                cc, da, cs
            )
            return {
                "pre_regime": PreRegimeType.PRE_COLLAPSE,
                "confidence": round(confidence, 4),
                "trigger_features": {
                    "coherence_curvature": cc,
                    "divergence_acceleration": da,
                    "coherence_slope": cs,
                },
            }

        # Check EARLY_EXPANSION
        if cc > self._ee_curv and cs > self._ee_slope:
            confidence = self._compute_expansion_confidence(cc, cs)
            return {
                "pre_regime": PreRegimeType.EARLY_EXPANSION,
                "confidence": round(confidence, 4),
                "trigger_features": {
                    "coherence_curvature": cc,
                    "coherence_slope": cs,
                },
            }

        # Default: STRUCTURAL_STABILITY
        stability_confidence = self._compute_stability_confidence(
            cc, da, si, cs
        )
        return {
            "pre_regime": PreRegimeType.STRUCTURAL_STABILITY,
            "confidence": round(stability_confidence, 4),
            "trigger_features": {},
        }

    # ------------------------------------------------------------------
    # Confidence computation
    # ------------------------------------------------------------------

    def _compute_pre_collapse_confidence(
        self, curvature: float, acceleration: float, slope: float
    ) -> float:
        """Confidence increases with feature magnitude beyond thresholds."""
        curv_margin = min(1.0, abs(curvature - self._pc_curv) / 0.15)
        accel_margin = min(1.0, (acceleration - self._pc_accel) / 0.15)
        slope_margin = min(1.0, abs(slope - self._pc_slope) / 0.10)
        avg = (curv_margin + accel_margin + slope_margin) / 3.0
        return 0.5 + 0.5 * avg

    def _compute_fast_confidence(
        self, curvature: float, acceleration: float, symmetry: float
    ) -> float:
        """Confidence increases with severity of geometric shock."""
        curv_margin = min(1.0, abs(curvature - self._fi_curv) / 0.20)
        accel_margin = min(1.0, (acceleration - self._fi_accel) / 0.20)
        asym_margin = min(1.0, (abs(symmetry) - self._fi_asym) / 0.20)
        avg = (curv_margin + accel_margin + asym_margin) / 3.0
        return 0.5 + 0.5 * avg

    def _compute_expansion_confidence(
        self, curvature: float, slope: float
    ) -> float:
        """Confidence increases with strength of expansion signal."""
        curv_margin = min(1.0, (curvature - self._ee_curv) / 0.15)
        slope_margin = min(1.0, (slope - self._ee_slope) / 0.10)
        avg = (curv_margin + slope_margin) / 2.0
        return 0.5 + 0.5 * avg

    def _compute_stability_confidence(
        self,
        curvature: float,
        acceleration: float,
        symmetry: float,
        slope: float,
    ) -> float:
        """Confidence that cluster is in a stable state.

        Higher when all features are near zero, meaning no significant
        geometric deformation is occurring.
        """
        # How far each feature is from "stable" (near zero)
        curv_dist = min(1.0, abs(curvature) / 0.10)
        accel_dist = min(1.0, abs(acceleration) / 0.10)
        sym_dist = min(1.0, abs(symmetry) / 0.20)
        slope_dist = min(1.0, abs(slope) / 0.10)

        avg_dist = (curv_dist + accel_dist + sym_dist + slope_dist) / 4.0
        stability = 1.0 - avg_dist
        return 0.5 + 0.5 * stability


# ---------------------------------------------------------------------------
# Lead Time Estimation
# ---------------------------------------------------------------------------


def estimate_lead_cycles(features: Dict[str, float], pre_regime: str) -> int:
    """Estimate how many cycles until geometry deformations affect RFE pressure.

    Parameters
    ----------
    features : dict
        Feature dict with at least 'coherence_curvature' for PRE_COLLAPSE
        and FAST_INSTABILITY estimation.
    pre_regime : str
        PreRegimeType constant.

    Returns
    -------
    int
        Estimated lead cycles until pressure impact.

    Rules
    -----
    PRE_COLLAPSE:
    - Mild curvature (-0.05 to -0.10): 8-12 cycles → return 10
    - Moderate curvature (-0.10 to -0.20): 4-7 cycles → return 5
    - Severe curvature (< -0.20): 2-3 cycles → return 2

    FAST_INSTABILITY:
    - Shock: 1-2 cycles → return 2

    EARLY_EXPANSION / STRUCTURAL_STABILITY:
    - Lead time: 0 (no impending pressure impact)
    """
    if pre_regime == PreRegimeType.PRE_COLLAPSE:
        curv = features.get("coherence_curvature", 0.0)
        if curv <= -0.20:
            return 2  # severe
        elif curv <= -0.10:
            return 5  # moderate
        else:
            return 10  # mild

    if pre_regime == PreRegimeType.FAST_INSTABILITY:
        return 2  # imminent impact

    # EARLY_EXPANSION and STRUCTURAL_STABILITY: no impending pressure
    return 0


# ---------------------------------------------------------------------------
# ClusterGeometryForecaster (Main Class)
# ---------------------------------------------------------------------------


class ClusterGeometryForecaster:
    """Orchestrates geometry tracking + classification.

    Evaluates each cluster's geometry trajectory and produces
    pre-pressure regime forecasts.

    Usage
    -----
        forecaster = ClusterGeometryForecaster()
        result = forecaster.evaluate(cluster_states)
        # Returns per-cluster regime + aggregate portfolio forecast
    """

    def __init__(self) -> None:
        self.tracker = GeometryTracker()
        self.classifier = GeometryClassifier()
        self._forecast_history: Dict[str, List[str]] = defaultdict(list)

    def evaluate(
        self, cluster_states: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Evaluate cluster geometry and produce pre-pressure forecasts.

        Parameters
        ----------
        cluster_states : dict
            Mapping of cluster name -> cluster state dict, same format as
            SignalManifoldProjector output. Expected keys per cluster:
            - coherence (float)
            - divergence (float)
            - net_direction (float)

        Returns
        -------
        dict
            - forecasts (dict): per-cluster regime predictions
            - portfolio_forecast (dict): aggregate portfolio summary
            - lead_time_map (dict): per-cluster lead cycle estimates
        """
        if not cluster_states:
            return {
                "forecasts": {},
                "portfolio_forecast": {
                    "dominant_regime": "NONE",
                    "regime_diversity": 0.0,
                    "collapse_clusters": [],
                    "fast_clusters": [],
                    "expansion_clusters": [],
                    "warning": False,
                },
                "lead_time_map": {},
            }

        # Step 1: Update tracker with current snapshot
        self.tracker.update(cluster_states)

        # Step 2: Classify each cluster
        forecasts: Dict[str, Dict[str, Any]] = {}
        for cluster in cluster_states:
            features = {
                "coherence_curvature": self.tracker.get_curvature(
                    cluster, "coherence"
                ),
                "divergence_acceleration": self.tracker.get_acceleration(
                    cluster, "divergence"
                ),
                "symmetry_imbalance": self.tracker.get_symmetry_imbalance(
                    cluster
                ),
                "coherence_slope": self.tracker.get_coherence_slope(
                    cluster
                ),
            }

            classification = self.classifier.classify(features)
            pre_regime = classification["pre_regime"]
            lead = estimate_lead_cycles(features, pre_regime)

            # Track forecast history
            self._forecast_history[cluster].append(pre_regime)

            forecasts[cluster] = {
                "pre_regime": pre_regime,
                "confidence": classification["confidence"],
                "features": features,
                "lead_estimate": lead,
                "history": list(self._forecast_history[cluster]),
            }

        # Step 3: Build portfolio forecast
        collapse_clusters: List[str] = []
        fast_clusters: List[str] = []
        expansion_clusters: List[str] = []
        stable_clusters: List[str] = []

        regime_counts: Dict[str, int] = defaultdict(int)
        for cluster, fc in forecasts.items():
            regime_counts[fc["pre_regime"]] += 1
            if fc["pre_regime"] == PreRegimeType.PRE_COLLAPSE:
                collapse_clusters.append(cluster)
            elif fc["pre_regime"] == PreRegimeType.FAST_INSTABILITY:
                fast_clusters.append(cluster)
            elif fc["pre_regime"] == PreRegimeType.EARLY_EXPANSION:
                expansion_clusters.append(cluster)
            else:
                stable_clusters.append(cluster)

        dominant_regime = (
            max(regime_counts, key=regime_counts.get)
            if regime_counts
            else PreRegimeType.STRUCTURAL_STABILITY
        )
        total = len(cluster_states)
        regime_diversity = len(collapse_clusters) / total if total > 0 else 0.0

        portfolio_forecast = {
            "dominant_regime": dominant_regime,
            "regime_diversity": round(regime_diversity, 4),
            "collapse_clusters": collapse_clusters,
            "fast_clusters": fast_clusters,
            "expansion_clusters": expansion_clusters,
            "stable_clusters": stable_clusters,
            "warning": len(collapse_clusters) > 0 or len(fast_clusters) > 0,
        }

        lead_time_map: Dict[str, int] = {
            cluster: fc["lead_estimate"] for cluster, fc in forecasts.items()
        }

        return {
            "forecasts": forecasts,
            "portfolio_forecast": portfolio_forecast,
            "lead_time_map": lead_time_map,
        }

    def get_symbol_forecast(self, symbol: str) -> Dict[str, Any]:
        """Get the latest forecast for a specific symbol's cluster.

        Parameters
        ----------
        symbol : str
            Trading symbol (e.g. 'CHFJPY').

        Returns
        -------
        dict
            Forecast for the cluster containing this symbol, or empty dict
            if no forecast available.
        """
        cluster = symbol_to_primary_cluster(symbol)
        # We don't store last forecast directly, so we reconstruct from
        # tracker + classifier. But since evaluate() already ran, the
        # latest entry in the forecast history is the current regime.
        history = self._forecast_history.get(cluster, [])
        if not history:
            return {
                "pre_regime": PreRegimeType.STRUCTURAL_STABILITY,
                "confidence": 0.5,
                "features": {
                    "coherence_curvature": 0.0,
                    "divergence_acceleration": 0.0,
                    "symmetry_imbalance": 0.0,
                    "coherence_slope": 0.0,
                },
                "lead_estimate": 0,
                "history": [],
            }

        # Use the tracker to compute current features
        features = {
            "coherence_curvature": self.tracker.get_curvature(
                cluster, "coherence"
            ),
            "divergence_acceleration": self.tracker.get_acceleration(
                cluster, "divergence"
            ),
            "symmetry_imbalance": self.tracker.get_symmetry_imbalance(
                cluster
            ),
            "coherence_slope": self.tracker.get_coherence_slope(cluster),
        }

        pre_regime = history[-1]
        lead = estimate_lead_cycles(features, pre_regime)

        return {
            "pre_regime": pre_regime,
            "confidence": 0.0,  # Would need classifier re-run for full conf
            "features": features,
            "lead_estimate": lead,
            "history": history,
        }

    def get_forecast_history(self, cluster: str) -> List[str]:
        """Return the forecast history for a given cluster."""
        return list(self._forecast_history.get(cluster, []))

    def reset(self) -> None:
        """Clear all internal state."""
        self.tracker.reset()
        self._forecast_history.clear()


# ---------------------------------------------------------------------------
# Dashboard Formatting
# ---------------------------------------------------------------------------


def format_geometry_dashboard(result: Dict[str, Any]) -> str:
    """Render the Cluster Geometry Forecaster dashboard.

    Parameters
    ----------
    result : dict
        Output from ``ClusterGeometryForecaster.evaluate()``.

    Returns
    -------
    str
        Formatted dashboard string.
    """
    lines: list[str] = []
    lines.append("")
    lines.append("CLUSTER GEOMETRY FORECASTER \u2014 PRE-RFE REGIME PREDICTION")
    lines.append("=" * 78)

    forecasts = result.get("forecasts", {})
    portfolio = result.get("portfolio_forecast", {})
    lead_time_map = result.get("lead_time_map", {})

    if not forecasts:
        lines.append("  (no cluster states evaluated)")
        lines.append("")
        lines.append("=" * 78)
        return "\n".join(lines)

    # Table header
    header = (
        f"{'Cluster':<15s} {'Pre-Regime':<22s} {'Conf.':<7s} "
        f"{'Curvature':<10s} {'Accel.':<8s} {'Sym.':<8s} {'Lead':<5s}"
    )
    lines.append(header)
    lines.append("-" * 78)

    for cluster in sorted(forecasts.keys()):
        fc = forecasts[cluster]
        pre = fc["pre_regime"]
        conf = fc["confidence"]
        feats = fc["features"]
        lead = lead_time_map.get(cluster, 0)

        curv = feats.get("coherence_curvature", 0.0)
        accel = feats.get("divergence_acceleration", 0.0)
        sym = feats.get("symmetry_imbalance", 0.0)

        lines.append(
            f"{cluster:<15s} {pre:<22s} {conf:<7.2f} "
            f"{curv:<10.4f} {accel:<8.4f} {sym:<8.4f} {lead:<5d}"
        )

    lines.append("")

    # Portfolio warning
    if portfolio.get("warning", False):
        collapse = portfolio.get("collapse_clusters", [])
        fast = portfolio.get("fast_clusters", [])
        if collapse:
            lines.append(
                f"PORTFOLIO WARNING: {', '.join(collapse)} cluster(s) "
                f"in PRE_COLLAPSE"
            )
        if fast:
            lines.append(
                f"PORTFOLIO WARNING: {', '.join(fast)} cluster(s) "
                f"in FAST_INSTABILITY"
            )

        # Show lead time for the first warned cluster
        warned = (collapse + fast)[0]
        lt = lead_time_map.get(warned, 0)
        lines.append(f"Lead time: ~{lt} cycle(s) before pressure impact")
        lines.append("")

    # Regime history for each cluster
    lines.append("GEOMETRY HISTORY:")
    for cluster in sorted(forecasts.keys()):
        fc = forecasts[cluster]
        hist = fc.get("history", [])
        if hist:
            # Summarize: count consecutive runs
            summary_parts: list[str] = []
            current = hist[0]
            count = 1
            for h in hist[1:]:
                if h == current:
                    count += 1
                else:
                    summary_parts.append(f"{current}[{count}]")
                    current = h
                    count = 1
            summary_parts.append(f"{current}[{count}] (current)")
            summary = " \u2192 ".join(summary_parts)
            lines.append(f"  {cluster}: {summary}")
        else:
            lines.append(f"  {cluster}: (no history)")

    lines.append("")

    # Summary counts
    collapse_count = len(portfolio.get("collapse_clusters", []))
    fast_count = len(portfolio.get("fast_clusters", []))
    expansion_count = len(portfolio.get("expansion_clusters", []))
    stable_count = len(portfolio.get("stable_clusters", []))

    lines.append(
        f"EXPANSION ({expansion_count}) | STABLE ({stable_count}) "
        f"| COLLAPSE ({collapse_count}) | FAST ({fast_count})"
    )

    lines.append("")
    lines.append("=" * 78)

    return "\n".join(lines)
