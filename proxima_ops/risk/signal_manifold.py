"""Signal Manifold Projection Layer — compresses raw OSS signals into cluster vectors.

Projects 387 raw per-symbol signals into 5–7 cluster manifold vectors with
net_direction, coherence, divergence, drift_alignment, and regime classification.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECTION_VERSION = "1.0.0"

CLUSTERS: Dict[str, List[str]] = {
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
    for cname, members in CLUSTERS.items():
        if symbol in members:
            return cname
    return "USD"


# ---------------------------------------------------------------------------
# Signal Manifold Projector
# ---------------------------------------------------------------------------


class SignalManifoldProjector:
    """
    Compresses N raw per-symbol OSS signals into 5–7 cluster manifold vectors.

    For each cluster:
    - net_direction: confidence-weighted directional bias in [-1, +1]
    - net_pressure:  string label BULLISH / BEARISH / NEUTRAL
    - coherence:     fraction of signals aligning with dominant direction
    - divergence:    1 - coherence
    - dominant_ecdf: average ECDF of the dominant-direction signals
    - drift_alignment: fraction of signals where drift matches direction
    - active_symbols / total_symbols
    - member_signals: original signal dicts
    """

    def __init__(self) -> None:
        self._last_result: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def project(
        self, signals: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Project a list of OSS signals onto the cluster manifold.

        Parameters
        ----------
        signals : list of dict
            Each dict must contain at least:
                - symbol (str)
                - direction (int): -1, 0, or +1
                - confidence (float): 0..1
                - ecdf (float): 0..1
                - drift (int): -1, 0, or +1

        Returns
        -------
        dict with keys ``clusters`` and ``meta``.
        """
        # Group signals by symbol
        by_symbol: Dict[str, List[Dict[str, Any]]] = {}
        for s in signals:
            sym = s.get("symbol", "UNKNOWN")
            by_symbol.setdefault(sym, []).append(s)

        clusters: Dict[str, Dict[str, Any]] = {}
        for cname, members in CLUSTERS.items():
            member_sigs: List[Dict[str, Any]] = []
            for sym in members:
                member_sigs.extend(by_symbol.get(sym, []))
            clusters[cname] = self._compute_cluster(member_sigs, len(members))

        meta = {
            "total_raw_signals": len(signals),
            "compression_ratio": (
                round(len(signals) / max(len(CLUSTERS), 1), 4)
                if signals
                else 0.0
            ),
            "dominant_regime": self.dominant_regime(clusters),
            "net_market_direction": self._net_market_direction(clusters),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        result: Dict[str, Any] = {"clusters": clusters, "meta": meta}
        self._last_result = result
        return result

    @staticmethod
    def dominant_regime(
        clusters: Dict[str, Dict[str, Any]],
    ) -> str:
        """
        Classify overall market regime from cluster projections.

        Returns ``RISK_ON``, ``RISK_OFF``, or ``MIXED``.
        """
        if not clusters:
            return "MIXED"

        def sign(cname: str) -> int:
            c = clusters.get(cname)
            if c is None:
                return 0
            nd = c.get("net_direction", 0.0)
            return 1 if nd > 0.05 else (-1 if nd < -0.05 else 0)

        aud_nd = sign("AUD_NZD")
        eur = sign("EUR")
        usd = sign("USD")
        chf = sign("CHF")

        if aud_nd > 0 and eur > 0 and usd < 0 and chf < 0:
            return "RISK_ON"
        if aud_nd < 0 and eur < 0 and usd > 0 and chf > 0:
            return "RISK_OFF"
        return "MIXED"

    def get_last_result(self) -> Optional[Dict[str, Any]]:
        """Return the most recent ``project()`` output."""
        return self._last_result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_cluster(
        signals: List[Dict[str, Any]], total_symbols: int
    ) -> Dict[str, Any]:
        """Compute manifold state for one cluster."""
        if not signals:
            return {
                "net_direction": 0.0,
                "net_pressure": "NEUTRAL",
                "confidence": 0.0,
                "coherence": 0.0,
                "divergence": 0.0,
                "active_symbols": 0,
                "total_symbols": total_symbols,
                "dominant_ecdf": 0.0,
                "drift_alignment": 0.0,
                "member_signals": [],
            }

        dirs = np.array([float(s.get("direction", 0)) for s in signals], dtype=float)
        confs = np.array([float(s.get("confidence", 0.5)) for s in signals], dtype=float)
        ecdfs = np.array([float(s.get("ecdf", 0.5)) for s in signals], dtype=float)
        drifts = np.array([float(s.get("drift", 0)) for s in signals], dtype=float)

        total_w = float(np.sum(confs)) + 1e-10

        # -- net_direction (confidence-weighted) --
        raw_sum = float(np.sum(dirs * confs))
        max_possible = float(np.sum(np.abs(dirs) * confs)) + 1e-10
        net_direction = raw_sum / max_possible
        net_direction = max(-1.0, min(1.0, net_direction))

        # -- net_pressure label --
        if net_direction > 0.15:
            pressure = "BULLISH"
        elif net_direction < -0.15:
            pressure = "BEARISH"
        else:
            pressure = "NEUTRAL"

        # -- Dominant direction (by signal count) --
        buy_count = int(np.sum(dirs > 0))
        sell_count = int(np.sum(dirs < 0))

        if buy_count == 0 and sell_count == 0:
            dominant_dir = 0
        elif buy_count >= sell_count:
            dominant_dir = 1
        else:
            dominant_dir = -1

        # -- Coherence & Divergence --
        if dominant_dir == 0:
            coherence = 0.0
        else:
            dominant_count = buy_count if dominant_dir > 0 else sell_count
            non_neutral = buy_count + sell_count
            coherence = dominant_count / non_neutral if non_neutral > 0 else 0.0
        divergence = 1.0 - coherence

        # -- Cluster confidence --
        directional_confs = confs[dirs != 0]
        mean_conf = float(np.mean(directional_confs)) if len(directional_confs) > 0 else 0.0
        cluster_confidence = mean_conf * coherence

        # -- Active symbols --
        active_symbols = len({
            s.get("symbol")
            for s in signals
            if s.get("direction", 0) != 0
        })

        # -- Dominant ECDF (simple average of signals matching dominant direction) --
        if dominant_dir != 0:
            dom_mask = dirs * dominant_dir > 1e-6
        else:
            dom_mask = np.abs(dirs) < 1e-6
        if dom_mask.sum() > 0:
            dom_ecdf = float(np.mean(ecdfs[dom_mask]))
        else:
            dom_ecdf = 0.5
        dom_ecdf = max(0.0, min(1.0, dom_ecdf))

        # -- Drift alignment (fraction where drift matches direction) --
        total_directional = int(np.sum(dirs != 0))
        if total_directional > 0:
            aligned = int(np.sum((dirs * drifts) > 0))
            drift_alignment = aligned / total_directional
        else:
            drift_alignment = 0.0

        return {
            "net_direction": round(net_direction, 4),
            "net_pressure": pressure,
            "confidence": round(cluster_confidence, 4),
            "coherence": round(coherence, 4),
            "divergence": round(divergence, 4),
            "active_symbols": active_symbols,
            "total_symbols": total_symbols,
            "dominant_ecdf": round(dom_ecdf, 4),
            "drift_alignment": round(drift_alignment, 4),
            "member_signals": signals,
        }

    @staticmethod
    def _net_market_direction(
        clusters: Dict[str, Dict[str, Any]]
    ) -> float:
        """Average net_direction across clusters with non-zero values."""
        vals = [
            c["net_direction"]
            for c in clusters.values()
            if c["net_direction"] != 0.0
        ]
        if not vals:
            return 0.0
        return round(float(np.mean(vals)), 4)
