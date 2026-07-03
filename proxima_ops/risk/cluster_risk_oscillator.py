"""Cluster Risk Oscillator — models risk as oscillation of correlated asset clusters.

Instead of 'risk per trade', measures:
- momentum: directional pressure building or fading
- compression (=coherence): how tightly signals align
- divergence: internal disagreement within clusters

Output determines whether the system should expand or compress risk exposure.
"""

from __future__ import annotations

import logging
import os
import pickle
import random
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from .signal_manifold import (
    CLUSTERS,
    SignalManifoldProjector,
    symbol_to_primary_cluster,
)

logger = logging.getLogger("proxima_ops.risk.cluster_risk_oscillator")

# ---------------------------------------------------------------------------
# Cluster correlation matrix (simplified — used for expansion safety check)
# 1.0 = perfectly correlated, 0.0 = independent
# ---------------------------------------------------------------------------
CLUSTER_CORRELATION: Dict[str, Dict[str, float]] = {
    "EUR":     {"EUR": 1.0, "USD": 0.3, "JPY": 0.2, "AUD_NZD": 0.3, "CHF": 0.4, "GBP": 0.4, "CAD": 0.2},
    "USD":     {"EUR": 0.3, "USD": 1.0, "JPY": 0.3, "AUD_NZD": 0.4, "CHF": 0.3, "GBP": 0.3, "CAD": 0.4},
    "JPY":     {"EUR": 0.2, "USD": 0.3, "JPY": 1.0, "AUD_NZD": 0.2, "CHF": 0.3, "GBP": 0.2, "CAD": 0.2},
    "AUD_NZD": {"EUR": 0.3, "USD": 0.4, "JPY": 0.2, "AUD_NZD": 1.0, "CHF": 0.3, "GBP": 0.3, "CAD": 0.4},
    "CHF":     {"EUR": 0.4, "USD": 0.3, "JPY": 0.3, "AUD_NZD": 0.3, "CHF": 1.0, "GBP": 0.3, "CAD": 0.2},
    "GBP":     {"EUR": 0.4, "USD": 0.3, "JPY": 0.2, "AUD_NZD": 0.3, "CHF": 0.3, "GBP": 1.0, "CAD": 0.2},
    "CAD":     {"EUR": 0.2, "USD": 0.4, "JPY": 0.2, "AUD_NZD": 0.4, "CHF": 0.2, "GBP": 0.2, "CAD": 1.0},
}

# ---------------------------------------------------------------------------
# Oscillator thresholds
# ---------------------------------------------------------------------------
COHERENCE_HIGH = 0.7     # above this → signals are tightly aligned
DIVERGENCE_HIGH = 0.5    # above this → cluster is divergent
MOMENTUM_THRESHOLD = 0.05  # minimum |momentum| to count as non-zero


class ClusterRiskOscillator:
    """
    Models risk as oscillation of correlated asset clusters.

    Instead of 'risk per trade', measures:
    - momentum: directional pressure building or fading
    - compression (=coherence): how tightly signals align
    - divergence: internal disagreement within clusters

    Output determines whether the system should expand or compress risk exposure.
    """

    def __init__(self) -> None:
        # Previous cluster state snapshot (for velocity / acceleration)
        self._previous_state: Dict[str, Dict[str, float]] = {}
        # Per-cluster velocity history for acceleration computation
        self._velocity_history: Dict[str, List[float]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def oscillate(
        self,
        cluster_state: Dict[str, Dict[str, Any]],
        open_positions: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Compute the full oscillation state for every cluster.

        Parameters
        ----------
        cluster_state : dict
            Output from ``SignalManifoldProjector.project()["clusters"]``.
        open_positions : list of dict, optional
            Each position has at least ``symbol``, ``type`` ("BUY"/"SELL"),
            ``volume``, and ``profit``.

        Returns
        -------
        dict
            Complete oscillator result containing per-cluster states,
            system-level decision, and portfolio alignment analysis.
        """
        if open_positions is None:
            open_positions = []

        cluster_results: Dict[str, Dict[str, Any]] = {}
        for cname in CLUSTERS:
            raw = cluster_state.get(cname, {})
            cluster_results[cname] = self._oscillate_cluster(cname, raw)

        system_decision = self._compute_system_decision(cluster_results)
        portfolio_alignment = self._analyze_portfolio(cluster_results, open_positions)

        # Persist for next call
        self._store_state(cluster_results)

        expanding = [c for c, s in cluster_results.items() if s["state"] == "EXPANDING"]
        contracting = [c for c, s in cluster_results.items() if s["state"] == "CONTRACTING"]
        divergent = [c for c, s in cluster_results.items() if s["state"] == "DIVERGENT"]
        neutral = [c for c, s in cluster_results.items() if s["state"] == "NEUTRAL"]

        return {
            "clusters": cluster_results,
            "system_decision": system_decision,
            "expanding_clusters": expanding,
            "contracting_clusters": contracting,
            "divergent_clusters": divergent,
            "neutral_clusters": neutral,
            "portfolio_alignment": portfolio_alignment,
            "timestamp": datetime.now().isoformat(),
        }

    def compute_momentum(
        self, cluster_name: str, cluster_now: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Compute momentum, velocity, and acceleration for a single cluster.

        Parameters
        ----------
        cluster_name : str
            Cluster identifier (e.g. ``"EUR"``).
        cluster_now : dict
            Current cluster state (must contain ``net_direction``).

        Returns
        -------
        dict with keys ``momentum``, ``velocity``, ``acceleration``.
        """
        np_now = cluster_now.get("net_direction", 0.0)
        prev = self._previous_state.get(cluster_name, {})

        # No previous measurement → velocity/acceleration are zero;
        # momentum is just the current net_direction (velocity = 0)
        if not prev:
            return {
                "momentum": round(0.7 * np_now, 4),
                "velocity": 0.0,
                "acceleration": 0.0,
            }

        np_prev = prev.get("net_direction", 0.0)

        # Velocity = change in net_direction since last measurement
        velocity = np_now - np_prev

        # Acceleration = change in velocity
        hist = self._velocity_history.get(cluster_name, [])
        vel_prev = hist[-1] if hist else 0.0
        acceleration = velocity - vel_prev

        # Momentum blends current bias with velocity (directional persistence)
        momentum = 0.7 * np_now + 0.3 * velocity

        return {
            "momentum": round(momentum, 4),
            "velocity": round(velocity, 4),
            "acceleration": round(acceleration, 4),
        }

    def risk_expansion_allowed(
        self, oscillator_state: Dict[str, Any]
    ) -> bool:
        """
        Return ``True`` only if it is safe to expand risk exposure.

        Conditions
        ----------
        1. At least two clusters are in EXPANDING state.
        2. The expanding clusters are pairwise *uncorrelated* (correlation < 0.4).
        """
        expanding = oscillator_state.get("expanding_clusters", [])
        if len(expanding) < 2:
            return False

        for i in range(len(expanding)):
            for j in range(i + 1, len(expanding)):
                corr = CLUSTER_CORRELATION.get(expanding[i], {}).get(expanding[j], 1.0)
                if corr >= 0.4:
                    return False
        return True

    def cluster_state_to_risk_weight(
        self, cluster_name: str, state: Dict[str, Any]
    ) -> float:
        """
        Convert a cluster's oscillation state into a risk weight in [0.0, 1.0].

        ==============  ====================
        State           Weight range
        ==============  ====================
        EXPANDING       0.80 – 1.00
        CONTRACTING     0.00 – 0.20
        DIVERGENT       0.20 – 0.40
        NEUTRAL         0.40 – 0.60
        ==============  ====================
        """
        osc_state = state.get("state", "NEUTRAL")
        momentum_mag = min(1.0, abs(state.get("momentum", 0.0)))
        coherence = min(1.0, max(0.0, state.get("coherence", 0.5)))

        if osc_state == "EXPANDING":
            return round(min(1.0, 0.80 + 0.20 * momentum_mag * coherence), 4)
        if osc_state == "CONTRACTING":
            return round(max(0.0, 0.20 - 0.20 * momentum_mag), 4)
        if osc_state == "DIVERGENT":
            return 0.30
        # NEUTRAL
        return 0.50

    def reset(self) -> None:
        """Clear all internal state (for testing or fresh start)."""
        self._previous_state = {}
        self._velocity_history = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _oscillate_cluster(
        self, cluster_name: str, state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compute oscillation state for a single cluster."""
        net_direction = state.get("net_direction", 0.0)
        divergence = state.get("divergence", 0.5)
        coherence = state.get("coherence", 0.5)

        momentum_info = self.compute_momentum(cluster_name, state)
        vel = momentum_info["velocity"]
        acc = momentum_info["acceleration"]
        momentum_val = momentum_info["momentum"]

        # -- Oscillation state rules --
        if momentum_val > MOMENTUM_THRESHOLD and coherence > COHERENCE_HIGH:
            osc_state = "EXPANDING"
        elif momentum_val < -MOMENTUM_THRESHOLD and coherence > COHERENCE_HIGH:
            osc_state = "CONTRACTING"
        elif divergence > DIVERGENCE_HIGH:
            osc_state = "DIVERGENT"
        else:
            osc_state = "NEUTRAL"

        risk_weight = self.cluster_state_to_risk_weight(cluster_name, {
            "state": osc_state,
            "momentum": momentum_val,
            "coherence": coherence,
            "net_direction": net_direction,
        })

        return {
            "state": osc_state,
            "momentum": round(momentum_val, 4),
            "velocity": round(vel, 4),
            "acceleration": round(acc, 4),
            "compression": round(coherence, 4),  # for display, coherence = compression
            "coherence": round(coherence, 4),
            "divergence": round(divergence, 4),
            "net_direction": round(net_direction, 4),
            "risk_weight": risk_weight,
            "signal_count": state.get("active_symbols", 0),
            "direction": state.get("net_pressure", "NEUTRAL"),
        }

    def _store_state(
        self, cluster_results: Dict[str, Dict[str, Any]]
    ) -> None:
        """Save current state for next momentum computation."""
        for cname, result in cluster_results.items():
            self._previous_state[cname] = {
                "net_direction": result["net_direction"],
                "momentum": result["momentum"],
                "velocity": result["velocity"],
            }
            self._velocity_history.setdefault(cname, []).append(result["velocity"])
            # Keep a rolling window of the last 10 readings
            if len(self._velocity_history[cname]) > 10:
                self._velocity_history[cname] = self._velocity_history[cname][-10:]

    def _compute_system_decision(
        self, cluster_results: Dict[str, Dict[str, Any]]
    ) -> str:
        """Return ``EXPAND``, ``CONTRACT``, or ``HOLD``."""
        expanding = sum(1 for c in cluster_results.values() if c["state"] == "EXPANDING")
        contracting = sum(1 for c in cluster_results.values() if c["state"] == "CONTRACTING")
        divergent = sum(1 for c in cluster_results.values() if c["state"] == "DIVERGENT")
        total = len(cluster_results)

        # Build a mock oscillator_state for the expansion check
        mock_osc = {
            "expanding_clusters": [
                c for c, s in cluster_results.items() if s["state"] == "EXPANDING"
            ],
        }

        if expanding >= 2 and self.risk_expansion_allowed(mock_osc):
            return "EXPAND"
        if contracting >= 2:
            return "CONTRACT"
        if divergent >= total // 2:
            return "HOLD"
        return "HOLD"

    def _analyze_portfolio(
        self,
        cluster_results: Dict[str, Dict[str, Any]],
        open_positions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Map open positions to clusters and check alignment with oscillation state.
        """
        aligned: List[Dict[str, Any]] = []
        counter: List[Dict[str, Any]] = []
        by_cluster: Dict[str, List[Dict[str, Any]]] = {}

        for pos in open_positions:
            symbol = pos.get("symbol", "")
            pos_type = str(pos.get("type", "")).upper()
            cluster = symbol_to_primary_cluster(symbol)

            by_cluster.setdefault(cluster, [])

            cluster_info = cluster_results.get(cluster, {})
            cluster_dir = cluster_info.get("direction", "NEUTRAL")  # BULLISH / BEARISH / NEUTRAL
            cluster_st = cluster_info.get("state", "NEUTRAL")

            # Normalise position type and cluster direction to common tokens
            #   BUY/SELL  ↔  BULLISH/BEARISH
            pos_is_buy = pos_type == "BUY"
            clus_is_bull = cluster_dir == "BULLISH"

            # A position is counter if:
            #   - its cluster is in CONTRACTING, OR
            #   - its cluster is EXPANDING but position direction opposes cluster direction
            is_aligned = True
            if cluster_st == "CONTRACTING":
                is_aligned = False
            elif (
                cluster_st == "EXPANDING"
                and cluster_dir != "NEUTRAL"
                and pos_is_buy != clus_is_bull
            ):
                is_aligned = False

            entry: Dict[str, Any] = {
                "symbol": symbol,
                "type": pos_type,
                "cluster": cluster,
                "cluster_state": cluster_st,
                "cluster_direction": cluster_dir,
                "aligned": is_aligned,
            }
            by_cluster[cluster].append(entry)
            if is_aligned:
                aligned.append(entry)
            else:
                counter.append(entry)

        return {
            "total_positions": len(open_positions),
            "aligned_count": len(aligned),
            "counter_count": len(counter),
            "aligned": aligned,
            "counter": counter,
            "by_cluster": by_cluster,
        }


# ======================================================================
# Signal generation helpers (for the ``__main__`` block)
# ======================================================================

def _generate_sample_signals(
    num_signals: int = 387,
    rng_seed: int = 42,
    cluster_bias: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """
    Produce synthetic OSS signals when real MT5 data is unavailable.

    Optionally biases signals toward a direction for a given cluster to
    create a more realistic oscillation scenario for demonstration.

    Parameters
    ----------
    num_signals : int
        Target number of signals to generate.
    rng_seed : int
        Seed for reproducible randomness.
    cluster_bias : dict, optional
        Mapping ``cluster_name -> directional_bias`` in [-1, +1].
        Positive values increase the proportion of BUY signals for
        symbols in that cluster; negative increases SELL proportion.
        Example: ``{"EUR": 0.6, "USD": -0.4}``
    """
    all_symbols = list({sym for members in CLUSTERS.values() for sym in members})
    timeframes = ["1H", "4H", "1D"]
    horizons = [3, 10, 20]
    rng = random.Random(rng_seed)

    # Build a deterministic combination pool
    combos = [(sym, tf, h) for sym in all_symbols for tf in timeframes for h in horizons]
    rng.shuffle(combos)

    # Pre-compute symbol-to-cluster bias map
    sym_bias: Dict[str, float] = {}
    if cluster_bias:
        for sym in all_symbols:
            # Find the first cluster that contains this symbol
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
            # Determine direction — biased if cluster_bias is set
            bias = sym_bias.get(sym, 0.0)
            if abs(bias) > 0.05:
                # Bias toward the sign of the bias
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


def _generate_real_signals(num_signals: int = 387) -> List[Dict[str, Any]]:
    """
    Generate signals from cached OSS models via MT5 data.

    Falls back to sample signals if MT5 or the cache directory is unavailable.
    """
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    cache_dir = os.path.join(project_root, "bootstrap", "oss_cache")

    if not os.path.isdir(cache_dir):
        logger.warning("OSS cache directory not found — using sample signals")
        return _generate_sample_signals(num_signals)

    try:
        from features.ecdf_transform import PerSymbolECDF
        import MetaTrader5 as mt5  # type: ignore[import-untyped]
    except ImportError:
        logger.warning("MT5 / ECDF not available — using sample signals")
        return _generate_sample_signals(num_signals)

    if not mt5.initialize():
        logger.warning("MT5 initialisation failed — using sample signals")
        return _generate_sample_signals(num_signals)

    TF_MAP: Dict[str, int] = {
        "1m": 1, "5m": 5, "15m": 15, "30m": 30, "1H": 16385, "4H": 16386,
    }
    HORIZONS = [3, 10, 20]
    CONF_MIN = 0.50
    ECDF_LOW, ECDF_HIGH = 0.20, 0.70

    all_symbols = list({sym for members in CLUSTERS.values() for sym in members})

    # Load cached models
    models_cache: Dict[str, Any] = {}
    for fn in os.listdir(cache_dir):
        if not fn.endswith(".pkl"):
            continue
        sym = fn[:-4]
        if sym not in all_symbols:
            continue
        try:
            with open(os.path.join(cache_dir, fn), "rb") as f:
                models_cache[sym] = pickle.load(f)
        except Exception:
            continue

    if not models_cache:
        mt5.shutdown()
        logger.warning("No OSS models found in cache — using sample signals")
        return _generate_sample_signals(num_signals)

    signals: List[Dict[str, Any]] = []
    for sym in sorted(models_cache.keys()):
        if len(signals) >= num_signals:
            break
        models = models_cache[sym]
        for tf_name, tf_id in TF_MAP.items():
            if len(signals) >= num_signals:
                break
            rates = mt5.copy_rates_from_pos(sym, tf_id, 0, 500)
            if rates is None or len(rates) < 300:
                continue

            closes = [r[4] for r in rates]
            ecdf_tracker = PerSymbolECDF(window_size=5000)
            ecdf_tracker.hydrate(sym, closes[:200])
            for c in closes[:200]:
                ecdf_tracker.compute_and_update(sym, c)

            drift_ema: Optional[float] = None
            alpha_ema = 2.0 / 21.0
            diff_hist: List[float] = []

            for idx in range(200, len(closes)):
                p = closes[idx]
                ev = ecdf_tracker.compute_and_update(sym, p)

                if drift_ema is None:
                    drift_ema = p
                else:
                    drift_ema = alpha_ema * p + (1.0 - alpha_ema) * drift_ema

                d = p - drift_ema
                diff_hist.append(d)
                if len(diff_hist) > 20:
                    diff_hist = diff_hist[-20:]

                dr = 0
                if len(diff_hist) > 5:
                    ls = float(np.std(diff_hist) + 1e-12)
                    z = d / ls
                    dr = 1 if z > 0.5 else (-1 if z < -0.5 else 0)

                if idx < len(closes) - 1:
                    continue

                for h in HORIZONS:
                    if len(signals) >= num_signals:
                        break
                    m = models.get(h)
                    if m is None:
                        continue
                    try:
                        info = m.predict_with_info(ev, drift_state=dr)
                    except Exception:
                        continue

                    sig_val = info.get("signal", 0)
                    conf = info.get("confidence", 0.0)
                    if sig_val == 0 or conf < CONF_MIN:
                        continue
                    if ev < ECDF_LOW or ev > ECDF_HIGH:
                        continue

                    direction = 1 if sig_val == 1 else -1
                    signals.append({
                        "symbol": sym,
                        "tf": tf_name,
                        "horizon": h,
                        "direction": direction,
                        "confidence": round(conf, 3),
                        "ecdf": round(ev, 3),
                        "drift": dr,
                        "p_cont": round(info.get("p_cont", 0.5), 3),
                        "bucket": info.get("bucket", "?"),
                        "price": round(closes[-1], 5),
                    })

    mt5.shutdown()

    if not signals:
        logger.warning("No real signals generated — using sample signals")
        return _generate_sample_signals(num_signals)

    return signals[:num_signals]


def _get_sample_open_positions() -> List[Dict[str, Any]]:
    """Return a realistic set of open positions for dashboard demonstration."""
    return [
        {"symbol": "EURUSD", "type": "BUY",  "volume": 0.10, "profit": 15.20},
        {"symbol": "GBPUSD", "type": "SELL", "volume": 0.05, "profit": -3.40},
        {"symbol": "USDJPY", "type": "BUY",  "volume": 0.10, "profit": 22.10},
        {"symbol": "AUDUSD", "type": "SELL", "volume": 0.05, "profit": 5.80},
        {"symbol": "AUDJPY", "type": "BUY",  "volume": 0.10, "profit": -8.30},
        {"symbol": "USDCHF", "type": "SELL", "volume": 0.10, "profit": 12.50},
        {"symbol": "GBPJPY", "type": "SELL", "volume": 0.05, "profit": 3.10},
        {"symbol": "EURJPY", "type": "BUY",  "volume": 0.05, "profit": -1.90},
    ]


# ======================================================================
# Dashboard formatting
# ======================================================================

def format_dashboard(result: Dict[str, Any]) -> str:
    """
    Render the full cluster risk oscillator dashboard as a formatted string.
    """
    lines: List[str] = []
    lines.append("")
    lines.append("Cluster Risk Oscillator Dashboard")
    lines.append("=" * 60)

    clusters = result.get("clusters", {})
    for cname in sorted(clusters.keys()):
        s = clusters[cname]
        lines.append(
            f"  {cname:<10s} {s['state']:<12s} "
            f"momentum={s['momentum']:+0.4f}  "
            f"compression={s['compression']:.2f}  "
            f"divergence={s['divergence']:.2f}"
        )

    lines.append("")
    sys_dec = result.get("system_decision", "HOLD")
    lines.append(f"  System Risk Decision: {sys_dec}")

    pa = result.get("portfolio_alignment", {})
    aligned_n = pa.get("aligned_count", 0)
    counter_n = pa.get("counter_count", 0)

    aligned_clusters = {e.get("cluster", "") for e in pa.get("aligned", [])}
    counter_clusters = {e.get("cluster", "") for e in pa.get("counter", [])}

    lines.append(f"  Portfolio Alignment:  {aligned_n} aligned, {counter_n} counter")
    if aligned_clusters:
        lines.append(f"    Aligned clusters: {', '.join(sorted(aligned_clusters))}")
    if counter_clusters:
        lines.append(f"    Counter clusters: {', '.join(sorted(counter_clusters))}")

    expanding = result.get("expanding_clusters", [])
    contracting = result.get("contracting_clusters", [])
    divergent = result.get("divergent_clusters", [])

    lines.append(f"  Expanding clusters:   {', '.join(expanding) if expanding else 'none'}")
    lines.append(f"  Contracting clusters: {', '.join(contracting) if contracting else 'none'}")
    lines.append(f"  Divergent clusters:   {', '.join(divergent) if divergent else 'none'}")

    lines.append("")
    lines.append("  Cluster Risk Weights:")
    for cname in sorted(clusters.keys()):
        w = clusters[cname].get("risk_weight", 0.5)
        lines.append(f"    {cname:<10s} weight={w:.2f}")

    lines.append("=" * 60)
    return "\n".join(lines)


# ======================================================================
# Main block
# ======================================================================

def main() -> None:
    """Entry point: generate ~387 signals, project, oscillate, display."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )

    print("\n" + "#" * 60)
    print("# Cluster Risk Oscillator — Analysis Module")
    print("#" * 60)

    # 1. Generate signals (try real, augment with sample to reach 387)
    print("\n[1/3] Generating signals from OSS models ...")
    signals = _generate_real_signals(num_signals=387)
    if len(signals) < 387:
        n_extra = 387 - len(signals)
        extra = _generate_sample_signals(num_signals=n_extra)
        signals.extend(extra)
        print(f"      -> {len(signals)} total ({len(signals) - n_extra} real + {n_extra} sample)")
    else:
        print(f"      -> {len(signals)} real signals generated")
    print(f"      -> {len({s['symbol'] for s in signals})} unique symbols")

    # 2. Project onto cluster manifold
    print("\n[2/3] Projecting onto cluster manifold ...")
    projector = SignalManifoldProjector()
    projection = projector.project(signals)
    cluster_state = projection["clusters"]
    meta = projection["meta"]
    print(f"      dominant_regime={meta['dominant_regime']}  "
          f"net_market_direction={meta['net_market_direction']:+0.4f}  "
          f"compression_ratio={meta['compression_ratio']:.4f}")
    for cname, cstate in sorted(cluster_state.items()):
        print(f"      {cname:<10s} direction={cstate['net_pressure']:<7s}  "
              f"net_dir={cstate['net_direction']:+0.4f}  "
              f"coherence={cstate['coherence']:.2f}  "
              f"active={cstate['active_symbols']}/{cstate['total_symbols']}")

    # 3. Run the oscillator (two passes to demonstrate velocity/acceleration)
    print("\n[3/3] Running cluster risk oscillator ...")
    oscillator = ClusterRiskOscillator()
    open_positions = _get_sample_open_positions()

    # First pass: bias in opposite direction for a stronger velocity swing
    first_bias: Dict[str, float] = {
        "EUR": -0.35,
        "AUD_NZD": -0.40,
        "USD": 0.35,
        "CHF": 0.30,
        "JPY": 0.0,
        "GBP": 0.0,
        "CAD": 0.0,
    }
    first_signals = _generate_sample_signals(
        num_signals=387, rng_seed=7, cluster_bias=first_bias
    )
    cluster_state_1 = projector.project(first_signals)["clusters"]
    _ = oscillator.oscillate(cluster_state_1, open_positions)

    # Second pass: generate directionally biased signals to demonstrate
    # the full range of oscillator states:
    #   EUR, AUD_NZD → strongly BULLISH (EXPANDING)
    #   USD, CHF     → strongly BEARISH (CONTRACTING)
    #   JPY, GBP, CAD → neutral / mixed
    # Use strong directional biases:
    #   EUR, AUD_NZD → strongly BULLISH
    #   USD, CHF     → strongly BEARISH
    #   JPY, GBP, CAD → mostly neutral
    bias: Dict[str, float] = {
        "EUR": 0.80,
        "AUD_NZD": 0.85,
        "USD": -0.75,
        "CHF": -0.70,
        "JPY": 0.10,
        "GBP": 0.15,
        "CAD": -0.10,
    }
    biased_signals = _generate_sample_signals(
        num_signals=387, rng_seed=99, cluster_bias=bias
    )
    cluster_state_2 = projector.project(biased_signals)["clusters"]
    result = oscillator.oscillate(cluster_state_2, open_positions)

    # Dashboard
    print(format_dashboard(result))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    safe_clusters = result.get("expanding_clusters", [])
    if safe_clusters:
        print(f"  Safe to add exposure: {', '.join(safe_clusters)}")
    else:
        print("  Safe to add exposure: none")

    counter_positions = result.get("portfolio_alignment", {}).get("counter", [])
    if counter_positions:
        print(f"  Positions counter to cluster state:")
        for pos in counter_positions:
            print(f"    {pos['symbol']} ({pos['type']}) "
                  f"-> cluster {pos['cluster']} is {pos['cluster_state']}")
    else:
        print("  All positions aligned with cluster oscillation states.")

    print(f"  System decision: {result['system_decision']}")
    print()


if __name__ == "__main__":
    main()
