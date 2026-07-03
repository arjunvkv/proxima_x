"""Trade Field Coherence Map — models trades as perturbation nodes in a cluster field.

Instead of treating trades as independent observations, each trade is a perturbation
in a shared cluster field. This module tracks:

- Field distortion: does the trade amplify or dampen its cluster's pressure?
- Resonance coupling: do trades in the same cluster phase-lock?
- Interference: do trades in different clusters cancel or reinforce?

Dependencies
------------
- ``signal_manifold.py`` for cluster definitions and symbol→cluster mapping.
- ``cluster_risk_oscillator.py`` for cluster correlation matrix.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .signal_manifold import (
    CLUSTERS,
    symbol_to_primary_cluster,
)
from .cluster_risk_oscillator import CLUSTER_CORRELATION

logger = logging.getLogger("proxima_ops.risk.trade_field_coherence")

# ---------------------------------------------------------------------------
# Cluster Field Parameters — field-theoretic overlay on the 7-cluster system
# ---------------------------------------------------------------------------

CLUSTER_FIELD_PARAMS: Dict[str, Dict[str, Any]] = {
    "AUD_NZD": {
        "core_pairs": ["AUDNZD"],
        "risk_sensitivity": 0.85,
        "commodity_correlation": 0.75,
        "typical_correlation": {
            "EUR": 0.35, "USD": -0.55, "JPY": 0.30,
            "CHF": -0.40, "GBP": 0.25, "CAD": 0.45,
        },
    },
    "EUR": {
        "core_pairs": ["EURGBP"],
        "risk_sensitivity": 0.40,
        "commodity_correlation": 0.30,
        "typical_correlation": {
            "USD": -0.70, "CHF": 0.65, "GBP": 0.50,
            "JPY": 0.20, "AUD_NZD": 0.35, "CAD": 0.20,
        },
    },
    "USD": {
        "core_pairs": [],
        "risk_sensitivity": 0.60,
        "commodity_correlation": 0.40,
        "typical_correlation": {
            "EUR": -0.70, "CHF": 0.30, "JPY": 0.30,
            "AUD_NZD": -0.55, "GBP": 0.30, "CAD": 0.40,
        },
    },
    "JPY": {
        "core_pairs": ["CHFJPY"],
        "risk_sensitivity": 0.55,
        "commodity_correlation": 0.25,
        "typical_correlation": {
            "EUR": 0.20, "USD": 0.30, "CHF": 0.55,
            "AUD_NZD": 0.30, "GBP": 0.20, "CAD": 0.20,
        },
    },
    "CHF": {
        "core_pairs": ["CHFJPY"],
        "risk_sensitivity": 0.45,
        "commodity_correlation": 0.20,
        "typical_correlation": {
            "EUR": 0.65, "USD": 0.30, "JPY": 0.55,
            "AUD_NZD": -0.40, "GBP": 0.30, "CAD": 0.20,
        },
    },
    "GBP": {
        "core_pairs": ["EURGBP", "GBPAUD", "GBPNZD"],
        "risk_sensitivity": 0.50,
        "commodity_correlation": 0.35,
        "typical_correlation": {
            "EUR": 0.50, "USD": 0.30, "JPY": 0.20,
            "AUD_NZD": 0.25, "CHF": 0.30, "CAD": 0.20,
        },
    },
    "CAD": {
        "core_pairs": [],
        "risk_sensitivity": 0.65,
        "commodity_correlation": 0.80,
        "typical_correlation": {
            "EUR": 0.20, "USD": 0.40, "JPY": 0.20,
            "AUD_NZD": 0.45, "CHF": 0.20, "GBP": 0.20,
        },
    },
}


# ---------------------------------------------------------------------------
# Trade Field Coherence Map
# ---------------------------------------------------------------------------

class TradeFieldCoherenceMap:
    """
    Models each open trade as a perturbation node in a cluster field.

    Tracks:
    - Field distortion: does the trade amplify or dampen its cluster's pressure?
    - Resonance coupling: do trades in the same cluster phase-lock?
    - Interference: do trades in different clusters cancel or reinforce?
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_field_distortion(
        self,
        trade: Dict[str, Any],
        cluster_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Measure how a single trade distorts its cluster's field.

        Parameters
        ----------
        trade : dict
            Must contain ``symbol``, ``type`` ("BUY"/"SELL"), ``volume``.
        cluster_state : dict
            The cluster's manifold state (must have ``net_pressure``, ``net_direction``).

        Returns
        -------
        dict with keys ``alignment``, ``distortion_magnitude``, ``amplifying``,
        ``cluster_pressure_before``, ``cluster_pressure_after``.
        """
        pos_type = str(trade.get("type", "")).upper()
        trade_dir = 1 if pos_type == "BUY" else (-1 if pos_type == "SELL" else 0)
        volume = float(trade.get("volume", 0.0))

        cluster_dir = cluster_state.get("net_direction", 0.0)
        cluster_pressure = cluster_state.get("net_pressure", "NEUTRAL")

        # -- Alignment: does trade direction match cluster net_pressure? --
        if cluster_dir > 0.15:  # BULLISH
            alignment = 1 if trade_dir > 0 else (-1 if trade_dir < 0 else 0)
        elif cluster_dir < -0.15:  # BEARISH
            alignment = 1 if trade_dir < 0 else (-1 if trade_dir > 0 else 0)
        else:  # NEUTRAL cluster
            alignment = 0

        # -- Distortion magnitude --
        #   estimate total cluster exposure from all signals in manifold state
        member_sigs = cluster_state.get("member_signals", [])
        total_cluster_volume = sum(
            float(s.get("confidence", 0.5)) for s in member_sigs
        ) + 1e-10
        distortion_magnitude = min(1.0, (volume * abs(alignment)) / total_cluster_volume)

        # -- Amplifying or dampening? --
        amplifying = alignment > 0
        dampening = alignment < 0

        # -- Cluster pressure before / after --
        #   before: the cluster's net_direction without this trade's influence
        #   after:  net_direction adjusted by the trade's directional contribution
        cluster_pressure_before = cluster_dir
        #   Normalise contribution: BUY pushes +, SELL pushes -
        trade_contribution = trade_dir * volume * 0.01  # scale to comparable units
        opposing_exposure = abs(cluster_dir) * 0.1 + 1e-10
        cluster_pressure_after = cluster_dir + trade_contribution / opposing_exposure
        cluster_pressure_after = max(-1.0, min(1.0, cluster_pressure_after))

        return {
            "alignment": alignment,
            "distortion_magnitude": round(distortion_magnitude, 4),
            "amplifying": amplifying,
            "dampening": dampening,
            "cluster_pressure_before": round(cluster_pressure_before, 4),
            "cluster_pressure_after": round(cluster_pressure_after, 4),
        }

    def compute_resonance(
        self,
        trade_a: Dict[str, Any],
        trade_b: Dict[str, Any],
        cluster: str,
    ) -> Dict[str, Any]:
        """
        Measure resonance (phase-locking) between two trades in the same cluster.

        Parameters
        ----------
        trade_a, trade_b : dict
            Each must contain ``type`` ("BUY"/"SELL") and ``volume``.
        cluster : str
            Cluster name they both belong to.

        Returns
        -------
        dict with keys ``in_phase``, ``out_of_phase``, ``resonance_strength``,
        ``constructive``.
        """
        dir_a = 1 if str(trade_a.get("type", "")).upper() == "BUY" else -1
        dir_b = 1 if str(trade_b.get("type", "")).upper() == "BUY" else -1
        vol_a = float(trade_a.get("volume", 0.0))
        vol_b = float(trade_b.get("volume", 0.0))

        in_phase = dir_a == dir_b
        out_of_phase = dir_a != dir_b

        # -- Resonance strength: 0-1 based on volume similarity and same direction --
        if in_phase:
            vol_similarity = 1.0 - abs(vol_a - vol_b) / max(vol_a, vol_b, 1e-10)
            resonance_strength = 0.5 + 0.5 * vol_similarity
            constructive = True
        else:
            # Opposite directions cancel → resonance is high but destructive
            vol_similarity = 1.0 - abs(vol_a - vol_b) / max(vol_a, vol_b, 1e-10)
            resonance_strength = 0.5 * vol_similarity
            constructive = False

        return {
            "in_phase": in_phase,
            "out_of_phase": out_of_phase,
            "resonance_strength": round(resonance_strength, 4),
            "constructive": constructive,
        }

    def compute_interference(
        self,
        trade_a: Dict[str, Any],
        trade_b: Dict[str, Any],
        cluster_a: str,
        cluster_b: str,
    ) -> Dict[str, Any]:
        """
        Measure interference between two trades in different clusters.

        Parameters
        ----------
        trade_a, trade_b : dict
            Each must contain ``type`` ("BUY"/"SELL") and ``symbol``.
        cluster_a, cluster_b : str
            Cluster names.

        Returns
        -------
        dict with keys ``cross_cluster_coupling``, ``interference_pattern``,
        ``net_system_effect``.
        """
        # -- Cross-cluster coupling from correlation matrix --
        coupling = CLUSTER_CORRELATION.get(cluster_a, {}).get(cluster_b, 0.0)

        # -- Determine macro direction of each trade --
        #   BUY = bullish on base currency = risk-on
        #   SELL = bearish = risk-off (simplified)
        dir_a = 1 if str(trade_a.get("type", "")).upper() == "BUY" else -1
        dir_b = 1 if str(trade_b.get("type", "")).upper() == "BUY" else -1

        # Check cluster field params for risk_sensitivity
        params_a = CLUSTER_FIELD_PARAMS.get(cluster_a, {})
        params_b = CLUSTER_FIELD_PARAMS.get(cluster_b, {})
        risk_sense_a = params_a.get("risk_sensitivity", 0.5)
        risk_sense_b = params_b.get("risk_sensitivity", 0.5)

        # -- Effective macro direction (weighted by risk_sensitivity) --
        macro_a = dir_a * risk_sense_a
        macro_b = dir_b * risk_sense_b

        # -- Interference pattern --
        if macro_a * macro_b > 0 and coupling > 0:
            pattern = "CONSTRUCTIVE"
            net_effect = (
                f"{cluster_a} {'bullish' if macro_a > 0 else 'bearish'} + "
                f"{cluster_b} {'bullish' if macro_b > 0 else 'bearish'} = "
                f"mutually reinforcing"
            )
        elif macro_a * macro_b > 0 and coupling < 0:
            pattern = "DESTRUCTIVE"
            net_effect = (
                f"same directional bias but negatively correlated clusters = "
                f"fields oppose each other"
            )
        elif macro_a * macro_b < 0 and coupling < 0:
            pattern = "CONSTRUCTIVE"
            net_effect = (
                f"opposite directional bias + negative cluster correlation = "
                f"natural hedge"
            )
        elif macro_a * macro_b < 0 and coupling > 0:
            pattern = "DESTRUCTIVE"
            net_effect = (
                f"opposing fields in positively correlated clusters = "
                f"cancellation risk"
            )
        else:
            pattern = "NEUTRAL"
            net_effect = f"minimal interaction between {cluster_a} and {cluster_b}"

        return {
            "cross_cluster_coupling": round(coupling, 4),
            "interference_pattern": pattern,
            "net_system_effect": net_effect,
        }

    def build_field_map(
        self,
        open_positions: List[Dict[str, Any]],
        cluster_states: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Full field coherence analysis for a portfolio of open positions.

        Parameters
        ----------
        open_positions : list of dict
            Each position has at least ``symbol``, ``type`` ("BUY"/"SELL"),
            ``volume``. May also have ``ticket``.
        cluster_states : dict
            Output from ``SignalManifoldProjector.project()["clusters"]``.

        Returns
        -------
        dict with keys ``field_map``, ``cross_cluster_interference``, ``system_health``.
        """
        # Group positions by cluster
        by_cluster: Dict[str, List[Dict[str, Any]]] = {}
        for pos in open_positions:
            sym = pos.get("symbol", "")
            cluster = symbol_to_primary_cluster(sym)
            by_cluster.setdefault(cluster, []).append(pos)

        field_map: Dict[str, Dict[str, Any]] = {}
        total_risk_exposure = 0.0
        cluster_risk_exposure: Dict[str, float] = {}

        for cname in CLUSTERS:
            trades = by_cluster.get(cname, [])
            cluster_state = cluster_states.get(cname, {})
            cluster_dir = cluster_state.get("net_direction", 0.0)

            if not trades:
                field_map[cname] = {
                    "trades": [],
                    "net_alignment": "NONE",
                    "distortion": 0.0,
                    "resonance": None,
                    "threat_level": "LOW",
                }
                cluster_risk_exposure[cname] = 0.0
                continue

            # Compute distortion for each trade
            distortions = [
                self.compute_field_distortion(t, cluster_state) for t in trades
            ]

            # Net alignment
            avg_alignment = sum(d["alignment"] for d in distortions) / len(trades)
            if avg_alignment > 0.3:
                net_alignment = "ALIGNED"
            elif avg_alignment < -0.3:
                net_alignment = "COUNTER"
            else:
                net_alignment = "NEUTRAL"

            # Aggregate distortion (average or max)
            distortion = sum(d["distortion_magnitude"] for d in distortions)

            # Resonance (if >= 2 trades in same cluster)
            resonance = None
            if len(trades) >= 2:
                res_pairs = []
                for i in range(len(trades)):
                    for j in range(i + 1, len(trades)):
                        r = self.compute_resonance(trades[i], trades[j], cname)
                        res_pairs.append(r)

                # Aggregate resonance
                if res_pairs:
                    avg_strength = sum(r["resonance_strength"] for r in res_pairs) / len(res_pairs)
                    all_constructive = all(r["constructive"] for r in res_pairs)
                    resonance = {
                        "in_phase": all(r["in_phase"] for r in res_pairs),
                        "out_of_phase": any(r["out_of_phase"] for r in res_pairs),
                        "strength": round(avg_strength, 4),
                        "constructive": all_constructive,
                    }

            # Threat level
            threat = self._threat_level(net_alignment, distortion, resonance)

            # Track exposure
            cluster_vol = sum(float(t.get("volume", 0.0)) for t in trades)
            cluster_risk_exposure[cname] = cluster_vol
            total_risk_exposure += cluster_vol

            field_map[cname] = {
                "trades": trades,
                "net_alignment": net_alignment,
                "distortion": round(distortion, 4),
                "resonance": resonance,
                "threat_level": threat,
            }

        # -- Cross-cluster interference --
        cross_cluster_interference: List[Dict[str, Any]] = []
        cluster_list = [c for c in CLUSTERS if by_cluster.get(c)]
        for i in range(len(cluster_list)):
            for j in range(i + 1, len(cluster_list)):
                ca, cb = cluster_list[i], cluster_list[j]
                ta = by_cluster[ca][0]  # representative trade
                tb = by_cluster[cb][0]
                interference = self.compute_interference(ta, tb, ca, cb)
                cross_cluster_interference.append({
                    "pair": (ca, cb),
                    "interference": interference["interference_pattern"],
                    "coupling": interference["cross_cluster_coupling"],
                    "net_effect": interference["net_system_effect"],
                })

        # -- System health --
        system_health = self._compute_system_health(
            field_map, by_cluster, cluster_risk_exposure, total_risk_exposure,
        )

        return {
            "field_map": field_map,
            "cross_cluster_interference": cross_cluster_interference,
            "system_health": system_health,
        }

    @staticmethod
    def _threat_level(
        net_alignment: str,
        distortion: float,
        resonance: Optional[Dict[str, Any]],
    ) -> str:
        """
        Determine threat level for a cluster field.

        Rules
        -----
        - 'HIGH': counter-aligned OR distortion > 0.5 OR destructive resonance
        - 'MEDIUM': neutral alignment OR moderate distortion (0.25–0.5)
        - 'LOW': aligned, low distortion (< 0.25), constructive resonance
        """
        if net_alignment == "COUNTER":
            return "HIGH"
        if distortion > 0.5:
            return "HIGH"
        if resonance and not resonance.get("constructive", True):
            return "HIGH"

        if net_alignment == "NEUTRAL":
            return "MEDIUM"
        if 0.25 < distortion <= 0.5:
            return "MEDIUM"

        return "LOW"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_system_health(
        field_map: Dict[str, Dict[str, Any]],
        by_cluster: Dict[str, List[Dict[str, Any]]],
        cluster_risk_exposure: Dict[str, float],
        total_risk_exposure: float,
    ) -> Dict[str, Any]:
        """Compute overall system health metrics."""
        # Overconcentrated: any cluster has >50% of total risk
        overconcentrated = False
        dominant_cluster = "NONE"
        max_exposure = 0.0
        for cname, exposure in cluster_risk_exposure.items():
            if total_risk_exposure > 0:
                share = exposure / total_risk_exposure
                if share > 0.5:
                    overconcentrated = True
                if exposure > max_exposure:
                    max_exposure = exposure
                    dominant_cluster = cname

        # Highest resonance
        highest_resonance: Optional[Tuple[str, float]] = None
        for cname, fm in field_map.items():
            res = fm.get("resonance")
            if res:
                strength = res.get("strength", 0.0)
                if highest_resonance is None or strength > highest_resonance[1]:
                    highest_resonance = (cname, strength)

        # Hedge effectiveness
        #   Simple heuristic: count pairs of trades in different clusters
        #   with opposite alignment → natural hedge
        trades_in_clusters = sum(1 for c in by_cluster.values() for _ in c)
        hedge_pairs = 0
        cluster_list = [c for c in CLUSTERS if by_cluster.get(c)]
        for i in range(len(cluster_list)):
            for j in range(i + 1, len(cluster_list)):
                ca, cb = cluster_list[i], cluster_list[j]
                align_a = field_map.get(ca, {}).get("net_alignment", "NONE")
                align_b = field_map.get(cb, {}).get("net_alignment", "NONE")
                if (align_a == "ALIGNED" and align_b == "COUNTER") or \
                   (align_a == "COUNTER" and align_b == "ALIGNED"):
                    hedge_pairs += 1

        total_pairs = len(cluster_list) * (len(cluster_list) - 1) / 2 if len(cluster_list) > 1 else 1
        hedge_effectiveness = hedge_pairs / total_pairs if total_pairs > 0 else 0.0

        # Overall verdict
        high_threats = sum(
            1 for fm in field_map.values() if fm["threat_level"] == "HIGH"
        )
        medium_threats = sum(
            1 for fm in field_map.values() if fm["threat_level"] == "MEDIUM"
        )

        if high_threats > 0:
            verdict = "CRITICAL"
        elif medium_threats > 1 or overconcentrated:
            verdict = "WARNING"
        elif medium_threats == 1:
            verdict = "WARNING"
        else:
            verdict = "STABLE"

        return {
            "overconcentrated": overconcentrated,
            "dominant_cluster": dominant_cluster,
            "highest_resonance": highest_resonance,
            "hedge_effectiveness": round(hedge_effectiveness, 4),
            "high_threat_clusters": high_threats,
            "medium_threat_clusters": medium_threats,
            "verdict": verdict,
        }


# ======================================================================
# Dashboard formatting
# ======================================================================

def format_field_coherence_dashboard(result: Dict[str, Any]) -> str:
    """
    Render the full Trade Field Coherence dashboard as a formatted string.
    """
    lines: List[str] = []
    lines.append("")
    lines.append("TRADE FIELD COHERENCE MAP")
    lines.append("=" * 67)

    field_map = result.get("field_map", {})

    # Header
    lines.append(
        f"{'Cluster':<12s} {'Trades':<7s} {'Alignment':<10s} {'Distort':<8s} "
        f"{'Resonance':<18s} {'Threat':<8s}"
    )
    lines.append("-" * 67)

    for cname in sorted(field_map.keys()):
        fm = field_map[cname]
        trades = fm["trades"]
        n_trades = len(trades)
        alignment = fm["net_alignment"]
        distortion = fm["distortion"]
        threat = fm["threat_level"]
        resonance = fm["resonance"]

        if n_trades == 0:
            lines.append(
                f"{cname:<12s} {'0':<7s} {'-':<10s} {'-':<8s} {'-':<18s} {'-':<8s}"
            )
            continue

        # Resonance display
        if resonance:
            icon = "✅" if resonance.get("constructive") else "⚠️"
            res_label = f"{icon} {'constr' if resonance['constructive'] else 'destr'}"
            res_str = f"{res_label:<18s}"
        else:
            res_str = f"{'N/A':<18s}"

        # Threat icon
        if threat == "HIGH":
            threat_str = "🔴 HIGH"
        elif threat == "MEDIUM":
            threat_str = "🟡 MEDIUM"
        else:
            threat_str = "✅ LOW"

        lines.append(
            f"{cname:<12s} {n_trades:<7d} {alignment:<10s} {distortion:<8.2f} "
            f"{res_str} {threat_str}"
        )

        # Show trade details for this cluster
        for t in trades:
            sym = t.get("symbol", "?")
            ptype = str(t.get("type", "")).upper()
            vol = t.get("volume", 0.0)
            lines.append(f"  {'':12s} ├─ {sym} {ptype} vol={vol}")

    # Cross-cluster interference
    lines.append("")
    lines.append("Cross-Cluster Interference:")
    interference_list = result.get("cross_cluster_interference", [])
    if interference_list:
        for item in interference_list:
            pair = item["pair"]
            pattern = item["interference"]
            coupling = item["coupling"]
            effect = item["net_effect"]
            if pattern == "CONSTRUCTIVE":
                icon = "✅"
            elif pattern == "DESTRUCTIVE":
                icon = "⚠️"
            else:
                icon = "➖"
            lines.append(
                f"  {pair[0]:<12s} ↔ {pair[1]:<12s} {pattern:<14s} "
                f"(ρ={coupling:<6.2f}) {icon}"
            )
            # Shortened effect on second line
            lines.append(f"  {'':26s} {effect}")
    else:
        lines.append("  (none — only one cluster has trades or no clusters)")

    # System health
    lines.append("")
    lines.append("System Health:")
    sys_h = result.get("system_health", {})
    over = sys_h.get("overconcentrated", False)
    lines.append(f"  Overconcentrated:       {'Yes ⚠️' if over else 'No ✅'}"
                 f" (max trades in {sys_h.get('dominant_cluster', 'NONE')})")
    lines.append(f"  Dominant Cluster:       {sys_h.get('dominant_cluster', 'NONE')}")
    hr = sys_h.get("highest_resonance")
    if hr:
        cname, strength = hr
        lines.append(f"  Highest Resonance:      {cname} ({strength:.2f} — "
                     f"{'constructive' if strength > 0.5 else 'destructive'})")
    else:
        lines.append(f"  Highest Resonance:      N/A (no cluster with ≥2 trades)")
    lines.append(f"  Hedge Effectiveness:    {sys_h.get('hedge_effectiveness', 0.0):.2f}")
    verdict = sys_h.get("verdict", "STABLE")
    if verdict == "STABLE":
        v_icon = "✅ STABLE"
    elif verdict == "WARNING":
        v_icon = "⚠️ WARNING"
    else:
        v_icon = "🔴 CRITICAL"
    lines.append(f"  Overall Health:         {v_icon}")

    lines.append("=" * 67)
    return "\n".join(lines)


# ======================================================================
# Sample data for demonstration
# ======================================================================

def get_sample_trades_3() -> List[Dict[str, Any]]:
    """
    Return a realistic 3-trade portfolio for demonstration.

    Uses symbols that map cleanly to their clusters:
      1. AUDNZD BUY 0.10  →  AUD_NZD cluster
      2. CADJPY BUY 0.05  →  JPY cluster  (JPY iterates before CAD)
      3. GBPCHF SELL 0.10 →  CHF cluster  (CHF iterates before GBP)
    """
    return [
        {"ticket": 1001, "symbol": "AUDNZD", "type": "BUY",  "volume": 0.10, "profit": 12.50},
        {"ticket": 1002, "symbol": "CADJPY", "type": "BUY",  "volume": 0.05, "profit": -3.20},
        {"ticket": 1003, "symbol": "GBPCHF", "type": "SELL", "volume": 0.10, "profit": 8.70},
    ]


def get_sample_trades_4() -> List[Dict[str, Any]]:
    """
    4-trade portfolio with 2 in AUD_NZD for resonance demonstration.

    Uses symbols that map cleanly:
      1. AUDNZD BUY 0.10  →  AUD_NZD
      2. GBPNZD BUY 0.05  →  AUD_NZD (AUD_NZD iterates before GBP)
      3. CADJPY BUY 0.05  →  JPY     (JPY iterates before CAD)
      4. GBPCHF SELL 0.10 →  CHF     (CHF iterates before GBP)
    """
    return [
        {"ticket": 2001, "symbol": "AUDNZD", "type": "BUY",  "volume": 0.10, "profit": 12.50},
        {"ticket": 2002, "symbol": "GBPNZD", "type": "BUY",  "volume": 0.05, "profit": 3.20},
        {"ticket": 2003, "symbol": "CADJPY", "type": "BUY",  "volume": 0.05, "profit": -3.20},
        {"ticket": 2004, "symbol": "GBPCHF", "type": "SELL", "volume": 0.10, "profit": 8.70},
    ]


# ======================================================================
# Signal generation for main block
# ======================================================================

def _generate_sample_signals(
    num_signals: int = 387,
    cluster_bias: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """Generate sample OSS signals for demonstration."""
    import random
    all_symbols = list({sym for members in CLUSTERS.values() for sym in members})
    timeframes = ["1H", "4H", "1D"]
    horizons = [3, 10, 20]
    rng = random.Random(42)

    combos = [(sym, tf, h) for sym in all_symbols for tf in timeframes for h in horizons]
    rng.shuffle(combos)

    sym_bias: Dict[str, float] = {}
    if cluster_bias:
        for sym in all_symbols:
            for cname, members in CLUSTERS.items():
                if sym in members:
                    sym_bias[sym] = cluster_bias.get(cname, 0.0)
                    break

    signals: List[Dict[str, Any]] = []
    idx = 0
    while len(signals) < num_signals:
        sym, tf, h = combos[idx % len(combos)]
        n_sigs = 1 if rng.random() < 0.6 else 2
        for _ in range(n_sigs):
            bias = sym_bias.get(sym, 0.0)
            if abs(bias) > 0.05:
                buy_prob = 0.5 + 0.4 * bias
                direction = 1 if rng.random() < buy_prob else -1
            else:
                direction = rng.choice([-1, 1])
            signals.append({
                "symbol": sym,
                "tf": tf,
                "horizon": h,
                "direction": direction,
                "confidence": round(rng.uniform(0.50, 0.95), 3),
                "ecdf": round(rng.uniform(0.20, 0.70), 3),
                "drift": rng.choice([-1, 0, 1]),
                "p_cont": round(rng.uniform(0.40, 0.80), 3),
                "bucket": f"{round(rng.uniform(0.20, 0.70), 1)}|{rng.choice([-1, 0, 1])}",
                "price": round(rng.uniform(1.0, 150.0), 5),
            })
        idx += 1
    return signals[:num_signals]


# ======================================================================
# Main block
# ======================================================================

def main() -> None:
    """Entry point: generate cluster state, compute field coherence, display dashboard."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )

    print("\n" + "#" * 67)
    print("# Trade Field Coherence Map — Analysis Module")
    print("#" * 67)

    # 1. Generate signals with realistic cluster biases
    print("\n[1/3] Generating cluster state from signals ...")
    bias: Dict[str, float] = {
        "EUR": 0.35,
        "AUD_NZD": 0.70,
        "USD": -0.40,
        "CHF": -0.30,
        "JPY": 0.10,
        "GBP": 0.15,
        "CAD": -0.10,
    }
    signals = _generate_sample_signals(num_signals=387, cluster_bias=bias)

    from .signal_manifold import SignalManifoldProjector
    projector = SignalManifoldProjector()
    projection = projector.project(signals)
    cluster_states = projection["clusters"]
    meta = projection["meta"]
    print(f"      dominant_regime={meta['dominant_regime']}  "
          f"net_market_direction={meta['net_market_direction']:+0.4f}")
    for cname, cstate in sorted(cluster_states.items()):
        print(f"      {cname:<10s} pressure={cstate['net_pressure']:<7s}  "
              f"net_dir={cstate['net_direction']:+0.4f}  "
              f"active={cstate['active_symbols']}/{cstate['total_symbols']}")

    # 2. Load sample portfolio
    print("\n[2/3] Loading portfolio ...")
    open_positions = get_sample_trades_3()
    print(f"      {len(open_positions)} open positions:")
    for pos in open_positions:
        print(f"        ticket={pos.get('ticket','?')}  {pos['symbol']} {pos['type']} "
              f"vol={pos['volume']}  PnL=${pos.get('profit',0.0):+.2f}")

    # 3. Compute field coherence
    print("\n[3/3] Computing Trade Field Coherence Map ...")
    mapper = TradeFieldCoherenceMap()
    field_result = mapper.build_field_map(open_positions, cluster_states)

    # 4. Dashboard
    print(format_field_coherence_dashboard(field_result))

    # 5. Summary
    print("\n" + "=" * 67)
    print("FIELD COHERENCE ASSESSMENT")
    print("=" * 67)
    sys_h = field_result["system_health"]
    fmap = field_result["field_map"]

    # Per-cluster alignment
    print("\nCluster Alignment Assessment:")
    for cname in sorted(fmap.keys()):
        fm = fmap[cname]
        if not fm["trades"]:
            continue
        print(f"  {cname}:")
        for t in fm["trades"]:
            sym = t.get("symbol", "?")
            ptype = t.get("type", "?")
            alignment = "ALIGNED" if fm["net_alignment"] == "ALIGNED" else \
                        "COUNTER" if fm["net_alignment"] == "COUNTER" else "NEUTRAL"
            print(f"    {sym} {ptype} → {alignment} (cluster alignment)")
        res = fm.get("resonance")
        if res:
            print(f"    Resonance: strength={res['strength']:.2f}, "
                  f"{'✅ constructive' if res['constructive'] else '⚠️ destructive'}")

    # Interference patterns
    print("\nInterference Patterns Between Clusters:")
    for item in field_result.get("cross_cluster_interference", []):
        pair = item["pair"]
        pat = item["interference"]
        coupling = item["coupling"]
        print(f"  {pair[0]} ↔ {pair[1]}: {pat} (ρ={coupling:.2f})")

    # System verdict
    print(f"\nSystem Health Verdict: {sys_h['verdict']}")
    print(f"  Dominant cluster: {sys_h['dominant_cluster']}")
    print(f"  Overconcentrated: {sys_h['overconcentrated']}")
    print(f"  High-threat clusters: {sys_h['high_threat_clusters']}")
    print(f"  Medium-threat clusters: {sys_h['medium_threat_clusters']}")
    print(f"  Hedge effectiveness: {sys_h['hedge_effectiveness']:.2f}")
    print()


if __name__ == "__main__":
    main()
