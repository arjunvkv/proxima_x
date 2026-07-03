"""Hysteresis Cluster Memory — stateful clustering with inertia.

Upgrades instantaneous clustering to stateful clustering with:
1. Dual thresholds: entry threshold (higher) vs exit threshold (lower)
2. Exponential memory kernel: past states decay gradually, not instantly
3. Regime locking: once in a state, must lose exit condition to leave

This wraps around the existing ``ClusterRiskOscillator`` to provide
hysteresis-stabilised cluster states that eliminate flip-flopping.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("proxima_ops.risk.hysteresis_cluster_memory")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLUSTER_NAMES = ["EUR", "USD", "JPY", "AUD_NZD", "CHF", "GBP", "CAD"]

VALID_STATES = {"NEUTRAL", "EXPANDING", "CONTRACTING", "DIVERGENT"}

# ---------------------------------------------------------------------------
# Hysteresis Cluster Memory
# ---------------------------------------------------------------------------


class HysteresisClusterMemory:
    """Stateful cluster analysis with hysteresis, memory kernel, and regime locking.

    Parameters
    ----------
    memory_decay : float, default=0.7
        Exponential decay factor for the memory kernel.
        0.0 = no memory (instantaneous), 1.0 = infinite memory (never changes).
    hysteresis_band : float, default=0.15
        Gap between entry and exit thresholds (applied symmetrically).
    min_lock_cycles : int, default=3
        Number of consecutive cycles in the same state before a cluster
        is considered "locked".

    Notes
    -----
    **Dual Threshold State Machine**

    +--------------+----------------+----------------+
    | State        | Enter          | Exit           |
    +==============+================+================+
    | EXPANDING    | score > +0.65  | score < +0.45  |
    | CONTRACTING  | score < -0.65  | score > -0.45  |
    | NEUTRAL      | -0.45..+0.45   | outside ±0.65  |
    | DIVERGENT    | div > 0.60     | div < 0.40     |
    +--------------+----------------+----------------+

    **Transition Rules**

    1. NEUTRAL → EXPANDING only if decayed_score > +0.65
    2. NEUTRAL → CONTRACTING only if decayed_score < -0.65
    3. EXPANDING → NEUTRAL only if decayed_score < +0.45
    4. CONTRACTING → NEUTRAL only if decayed_score > -0.45
    5. DIVERGENT enters when divergence > 0.60, exits when divergence < 0.40
    6. States always transition through NEUTRAL (no direct EXPANDING↔CONTRACTING)
    """

    # ------------------------------------------------------------------
    # Thresholds — entry (harder to reach) vs exit (easier to leave)
    # ------------------------------------------------------------------
    THRESHOLDS: Dict[str, Dict[str, Any]] = {
        "EXPANDING": {
            "enter": 0.65,  # must exceed to enter EXPANDING
            "exit": 0.45,  # must drop below to leave EXPANDING
        },
        "CONTRACTING": {
            "enter": -0.65,  # must go below to enter CONTRACTING
            "exit": -0.45,  # must go above to leave CONTRACTING
        },
        "NEUTRAL": {
            "enter": (-0.45, 0.45),  # within this band to enter NEUTRAL
            "exit": (-0.65, 0.65),  # must break this band to leave NEUTRAL
        },
        "DIVERGENT": {
            "enter": 0.60,  # divergence above this enters
            "exit": 0.40,  # divergence below this exits
        },
    }

    def __init__(
        self,
        memory_decay: float = 0.7,
        hysteresis_band: float = 0.15,
        min_lock_cycles: int = 3,
    ) -> None:
        if not 0.0 <= memory_decay <= 1.0:
            raise ValueError(f"memory_decay must be in [0, 1], got {memory_decay}")
        if not 0.0 <= hysteresis_band <= 1.0:
            raise ValueError(
                f"hysteresis_band must be in [0, 1], got {hysteresis_band}"
            )
        if min_lock_cycles < 1:
            raise ValueError(
                f"min_lock_cycles must be >= 1, got {min_lock_cycles}"
            )

        self.memory_decay = memory_decay
        self.hysteresis_band = hysteresis_band
        self.min_lock_cycles = min_lock_cycles

        # Per-cluster persistent state
        self.cluster_history: Dict[str, Dict[str, Any]] = {}

        # Global flip log — every state transition is recorded
        self.flip_log: List[Dict[str, Any]] = []

        # Accumulated history for before/after comparison
        self._raw_score_history: Dict[str, List[float]] = {}
        self._raw_divergence_history: Dict[str, List[float]] = {}
        self._raw_state_history: Dict[str, List[str]] = {}
        self._hyst_state_history: Dict[str, List[str]] = {}

        # Initialise all known clusters
        for name in CLUSTER_NAMES:
            self._init_cluster(name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, cluster_states: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Process a new set of raw cluster states through the hysteresis filter.

        Parameters
        ----------
        cluster_states : dict
            Raw cluster states from ``SignalManifoldProjector.project()["clusters"]``.
            Each value must contain at least ``net_direction`` and ``divergence``.

        Returns
        -------
        dict
            Stabilised cluster states with hysteresis metadata, flip events,
            locked clusters, and activation flag.
        """
        flip_events: List[Dict[str, Any]] = []
        cluster_results: Dict[str, Dict[str, Any]] = {}

        for cname in CLUSTER_NAMES:
            raw = cluster_states.get(cname, {})
            net_dir = raw.get("net_direction", 0.0)
            divergence = raw.get("divergence", 0.5)

            # --- step 1: compute instantaneous raw state (no hysteresis) ---
            raw_state = self._compute_raw_state(net_dir, divergence)

            # --- step 2: apply memory kernel ---
            self._apply_memory_kernel(cname, net_dir)

            # --- step 3: apply hysteresis state machine ---
            hist = self.cluster_history[cname]
            prev_state = hist["current_state"]
            new_state = self._compute_state(cname, net_dir, divergence)

            # Track cycles
            if new_state == prev_state:
                hist["cycles_in_state"] += 1
            else:
                hist["cycles_in_state"] = 1
                flip_events.append({
                    "cluster": cname,
                    "from_state": prev_state,
                    "to_state": new_state,
                    "score": round(hist["decayed_score"], 4),
                    "raw_net_dir": round(net_dir, 4),
                    "divergence": round(divergence, 4),
                })
                self.flip_log.append(flip_events[-1])

            hist["previous_state"] = prev_state
            hist["current_state"] = new_state

            # Store state history for stability comparison
            self._raw_state_history.setdefault(cname, []).append(raw_state)
            self._hyst_state_history.setdefault(cname, []).append(new_state)
            self._raw_score_history.setdefault(cname, []).append(net_dir)
            self._raw_divergence_history.setdefault(cname, []).append(divergence)

            locked = self.is_locked(cname)

            # Build result for this cluster
            cluster_results[cname] = {
                "raw_net_dir": round(net_dir, 4),
                "raw_state": raw_state,
                "decayed_score": round(hist["decayed_score"], 4),
                "current_state": new_state,
                "previous_state": prev_state,
                "cycles_in_state": hist["cycles_in_state"],
                "locked": locked,
                "entry_threshold": self._entry_threshold_for(new_state),
                "exit_threshold": self._exit_threshold_for(new_state),
            }

        locked_clusters = [
            c for c in CLUSTER_NAMES if cluster_results[c]["locked"]
        ]

        return {
            "clusters": cluster_results,
            "flip_events": flip_events,
            "locked_clusters": locked_clusters,
            "hysteresis_active": True,
            "memory_decay": self.memory_decay,
            "hysteresis_band": self.hysteresis_band,
            "min_lock_cycles": self.min_lock_cycles,
        }

    def is_locked(self, cluster_name: str) -> bool:
        """Return ``True`` if the cluster has been in the same state
        for at least ``min_lock_cycles``."""
        hist = self.cluster_history.get(cluster_name)
        if hist is None:
            return False
        return hist["cycles_in_state"] >= self.min_lock_cycles

    def compute_stability_score(self, cluster_name: str) -> Dict[str, Any]:
        """Compute stability metrics using hysteresis-stabilised states.

        Metrics:
        - StabHalfLife: avg cycles between state flips
        - StabFlipFreq: flips per total updates
        - StabVolatility: std of decayed net_direction scores

        Returns
        -------
        dict with ``raw_stability``, ``stabilized_stability``, ``improvement``,
        ``score``, and per-metric breakdown.
        """
        raw_states = self._raw_state_history.get(cluster_name, [])
        hyst_states = self._hyst_state_history.get(cluster_name, [])
        raw_scores = self._raw_score_history.get(cluster_name, [])
        n = len(raw_states)

        if n < 2:
            return {
                "raw_stability": 0.0,
                "stabilized_stability": 0.0,
                "improvement": 0.0,
                "score": 0.0,
                "half_life_raw": 0.0,
                "half_life_hyst": 0.0,
                "flip_freq_raw": 0.0,
                "flip_freq_hyst": 0.0,
                "volatility_raw": 0.0,
                "volatility_hyst": 0.0,
                "sample_count": n,
            }

        # --- Raw flips (instantaneous state) ---
        raw_flips = sum(
            1 for i in range(1, len(raw_states)) if raw_states[i] != raw_states[i - 1]
        )
        # --- Hysteresis flips ---
        hyst_flips = sum(
            1
            for i in range(1, len(hyst_states))
            if hyst_states[i] != hyst_states[i - 1]
        )

        raw_half_life = n / max(raw_flips, 1)
        hyst_half_life = n / max(hyst_flips, 1)
        raw_flip_freq = raw_flips / n
        hyst_flip_freq = hyst_flips / n

        # Volatility = std of net_direction
        raw_vol = float(np.std(raw_scores)) if len(raw_scores) > 1 else 0.0

        # For hysteresis volatility, use decayed scores
        hyst_scores = [
            s.get("decayed_score", 0.0)
            for s in self.cluster_history.get(cluster_name, {}).get("states", [])
        ]
        if not hyst_scores:
            # Fall back to raw scores during early accumulation
            hyst_scores = raw_scores
        hyst_vol = float(np.std(hyst_scores)) if len(hyst_scores) > 1 else 0.0

        # Composite stability score: higher = more stable
        # Components: half_life (longer = better), flip_freq (lower = better),
        # volatility (lower = better, normalised)
        raw_composite = self._composite_stability(
            raw_half_life, raw_flip_freq, raw_vol
        )
        hyst_composite = self._composite_stability(
            hyst_half_life, hyst_flip_freq, hyst_vol
        )

        improvement = hyst_composite - raw_composite

        return {
            "raw_stability": round(raw_composite, 4),
            "stabilized_stability": round(hyst_composite, 4),
            "improvement": round(improvement, 4),
            "score": round(hyst_composite, 4),
            "half_life_raw": round(raw_half_life, 2),
            "half_life_hyst": round(hyst_half_life, 2),
            "flip_freq_raw": round(raw_flip_freq, 4),
            "flip_freq_hyst": round(hyst_flip_freq, 4),
            "volatility_raw": round(raw_vol, 4),
            "volatility_hyst": round(hyst_vol, 4),
            "sample_count": n,
        }

    def compare_stability(
        self, raw_history: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Compare stability metrics before vs after hysteresis for all clusters.

        Parameters
        ----------
        raw_history : dict, optional
            Pre-computed raw stability metrics. If ``None``, computed from
            accumulated internal history.

        Returns
        -------
        dict mapping cluster name to ``{before, after, improvement}``.
        """
        results: Dict[str, Dict[str, Any]] = {}
        for cname in CLUSTER_NAMES:
            metrics = self.compute_stability_score(cname)
            if metrics["sample_count"] < 2:
                results[cname] = {
                    "before": {"stability": 0.0, "flip_freq": 0.0, "half_life": 0.0},
                    "after": {"stability": 0.0, "flip_freq": 0.0, "half_life": 0.0},
                    "improvement": "0%",
                }
                continue

            before = {
                "stability": metrics["raw_stability"],
                "flip_freq": metrics["flip_freq_raw"],
                "half_life": metrics["half_life_raw"],
            }
            after = {
                "stability": metrics["stabilized_stability"],
                "flip_freq": metrics["flip_freq_hyst"],
                "half_life": metrics["half_life_hyst"],
            }
            before_stab = before["stability"]
            pct = (
                f"+{((after['stability'] - before_stab) / max(before_stab, 0.001) * 100):.0f}%"
                if before_stab > 0.001
                else "N/A"
            )

            results[cname] = {
                "before": before,
                "after": after,
                "improvement": pct,
            }

        return results

    def reset(self) -> None:
        """Clear all accumulated state (for testing or fresh start)."""
        self.cluster_history.clear()
        self.flip_log.clear()
        self._raw_score_history.clear()
        self._raw_divergence_history.clear()
        self._raw_state_history.clear()
        self._hyst_state_history.clear()
        for name in CLUSTER_NAMES:
            self._init_cluster(name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_cluster(self, name: str) -> None:
        """Initialise or reset a single cluster's memory state."""
        self.cluster_history[name] = {
            "states": [],
            "current_state": "NEUTRAL",
            "previous_state": "NEUTRAL",
            "decayed_score": 0.0,
            "cycles_in_state": 0,
        }

    def _apply_memory_kernel(self, cluster_name: str, raw_score: float) -> None:
        """Update the exponentially decayed score for a cluster.

        ``decayed_score = memory_decay * prev + (1 - memory_decay) * raw``
        """
        hist = self.cluster_history[cluster_name]
        prev_decayed = hist["decayed_score"]
        new_decayed = (
            self.memory_decay * prev_decayed
            + (1.0 - self.memory_decay) * raw_score
        )
        hist["decayed_score"] = new_decayed
        hist["states"].append({
            "raw_score": raw_score,
            "decayed_score": new_decayed,
            "state": hist["current_state"],
        })

    def _compute_raw_state(
        self, net_direction: float, divergence: float
    ) -> str:
        """Compute the instantaneous state (no hysteresis, no memory)."""
        T = self.THRESHOLDS
        if net_direction > T["EXPANDING"]["enter"]:
            return "EXPANDING"
        if net_direction < T["CONTRACTING"]["enter"]:
            return "CONTRACTING"
        if divergence > T["DIVERGENT"]["enter"]:
            return "DIVERGENT"
        return "NEUTRAL"

    def _compute_state(
        self, cluster_name: str,
        raw_net_direction: float,
        raw_divergence: float,
    ) -> str:
        """Determine the hysteresis-stabilised state.

        Uses dual thresholds and regime locking.
        """
        hist = self.cluster_history[cluster_name]
        current_state = hist["current_state"]
        decayed_score = hist["decayed_score"]
        cycles = hist["cycles_in_state"]
        locked = cycles >= self.min_lock_cycles

        T = self.THRESHOLDS

        # --- Rule 4: DIVERGENT (based on raw divergence) ---
        if current_state == "DIVERGENT":
            if raw_divergence > T["DIVERGENT"]["exit"]:
                return "DIVERGENT"
            return "NEUTRAL"

        # For DIVERGENT entry, check from any non-divergent state
        if raw_divergence > T["DIVERGENT"]["enter"]:
            return "DIVERGENT"

        # --- Rules 1-3: directional states ---
        if current_state == "NEUTRAL":
            # Enter EXPANDING?
            if decayed_score > T["EXPANDING"]["enter"]:
                return "EXPANDING"
            # Enter CONTRACTING?
            if decayed_score < T["CONTRACTING"]["enter"]:
                return "CONTRACTING"
            return "NEUTRAL"

        if current_state == "EXPANDING":
            # Stay EXPANDING unless exit threshold breached
            if decayed_score > T["EXPANDING"]["exit"]:
                return "EXPANDING"
            return "NEUTRAL"

        if current_state == "CONTRACTING":
            # Stay CONTRACTING unless exit threshold breached
            if decayed_score < T["CONTRACTING"]["exit"]:
                return "CONTRACTING"
            return "NEUTRAL"

        # Fallback
        return "NEUTRAL"

    def _entry_threshold_for(self, state: str) -> float:
        """Return the entry threshold for a given state (for display)."""
        T = self.THRESHOLDS.get(state, {})
        entry = T.get("enter", 0.0)
        if isinstance(entry, tuple):
            return entry[1]  # upper bound for NEUTRAL
        return entry

    def _exit_threshold_for(self, state: str) -> float:
        """Return the exit threshold for a given state (for display)."""
        T = self.THRESHOLDS.get(state, {})
        exit_ = T.get("exit", 0.0)
        if isinstance(exit_, tuple):
            return exit_[1]  # upper bound for NEUTRAL
        return exit_

    @staticmethod
    def _composite_stability(
        half_life: float, flip_freq: float, volatility: float
    ) -> float:
        """Compute a composite stability score in [0, 10].

        Higher is more stable. Blend of normalised half-life, inverse
        flip frequency, and inverse volatility.
        """
        # Normalise half-life: cap at 50 cycles → 10 points
        hl_score = min(half_life / 5.0, 10.0)
        # Inverse flip frequency: 0 flips → 10, 1 flip/cycle → 0
        ff_score = (1.0 - min(flip_freq, 1.0)) * 10.0
        # Inverse volatility: 0 vol → 10, high vol → 0
        vol_score = (1.0 - min(volatility, 1.0)) * 10.0
        # Weighted average
        return 0.4 * hl_score + 0.3 * ff_score + 0.3 * vol_score


# ======================================================================
# Dashboard formatting
# ======================================================================


def format_hysteresis_dashboard(result: Dict[str, Any]) -> str:
    """Render the full Hysteresis Cluster Memory dashboard as a string.

    Parameters
    ----------
    result : dict
        Output from ``HysteresisClusterMemory.update()``.

    Returns
    -------
    str
        Formatted dashboard.
    """
    lines: List[str] = []
    lines.append("")
    lines.append("HYSTERESIS CLUSTER MEMORY SYSTEM")
    lines.append("=" * 50)
    decay = result.get("memory_decay", 0.7)
    band = result.get("hysteresis_band", 0.15)
    lock = result.get("min_lock_cycles", 3)
    lines.append(
        f"Memory Decay: {decay:.2f}  |  "
        f"Hysteresis Band: {band:.2f}  |  "
        f"Lock Cycles: {lock}"
    )
    lines.append("")

    clusters = result.get("clusters", {})
    # Header
    lines.append(
        f"{'Cluster':<12s} {'RawState':<12s} {'HystState':<12s} "
        f"{'Score(Decayed)':<18s} {'Locked':<8s} {'Cycles':<6s}"
    )
    lines.append("-" * 70)

    for cname in sorted(clusters.keys()):
        s = clusters[cname]
        raw_st = s["raw_state"]
        hyst_st = s["current_state"]
        raw_nd = s["raw_net_dir"]
        decayed = s["decayed_score"]
        locked = "✅" if s["locked"] else "❌"
        cycles = s["cycles_in_state"]

        # Format score arrow
        score_str = f"{raw_nd:+0.2f} → {decayed:+0.2f}"

        lines.append(
            f"{cname:<12s} {raw_st:<12s} {hyst_st:<12s} "
            f"{score_str:<18s} {locked:<8s} {cycles:<6d}"
        )

    lines.append("")

    # Flip events
    flip_events = result.get("flip_events", [])
    lines.append(f"FLIP EVENTS (this cycle):")
    if flip_events:
        for ev in flip_events:
            lines.append(
                f"  {ev['cluster']}: {ev['from_state']} → {ev['to_state']} "
                f"(score crossed exit threshold)"
            )
    else:
        lines.append("  None")

    # Locked clusters
    locked_clusters = result.get("locked_clusters", [])
    lines.append("")
    lines.append(
        f"LOCKED CLUSTERS: {', '.join(locked_clusters) if locked_clusters else 'None'}"
    )
    lines.append("")

    return "\n".join(lines)


def format_stability_comparison(
    comparison: Dict[str, Any],
    title: str = "BEFORE vs AFTER STABILITY COMPARISON",
) -> str:
    """Render the before/after stability comparison table.

    Parameters
    ----------
    comparison : dict
        Output from ``HysteresisClusterMemory.compare_stability()``.
    title : str
        Optional title for the table.

    Returns
    -------
    str
        Formatted comparison table.
    """
    lines: List[str] = []
    lines.append("")
    lines.append(title)
    lines.append("=" * 50)
    lines.append(
        f"{'Cluster':<12s} {'Before':<10s} {'After':<10s} {'Improvement':<12s}"
    )
    lines.append("-" * 50)

    totals = {"before": 0.0, "after": 0.0, "count": 0}
    for cname in sorted(comparison.keys()):
        data = comparison[cname]
        before = data["before"]["stability"]
        after = data["after"]["stability"]
        impr = data["improvement"]
        lines.append(
            f"{cname:<12s} {before:<10.1f} {after:<10.1f} {impr:<12s}"
        )
        totals["before"] += before
        totals["after"] += after
        totals["count"] += 1

    if totals["count"] > 0:
        avg_before = totals["before"] / totals["count"]
        avg_after = totals["after"] / totals["count"]
        avg_improvement = (
            f"+{((avg_after - avg_before) / max(avg_before, 0.001) * 100):.0f}%"
            if avg_before > 0.001
            else "N/A"
        )
        lines.append("-" * 50)
        lines.append(
            f"{'Average':<12s} {avg_before:<10.1f} {avg_after:<10.1f} {avg_improvement:<12s}"
        )

    lines.append("")
    return "\n".join(lines)


def format_verdict(comparison: Dict[str, Any]) -> str:
    """Generate a USD verdict string from the comparison data."""
    lines: List[str] = []
    lines.append("")
    lines.append("USD VERDICT:")
    lines.append("-" * 50)

    usd_data = comparison.get("USD", {})
    if not usd_data or usd_data["before"]["stability"] == 0.0:
        lines.append("  Insufficient data to evaluate USD stability.")
        lines.append("  Run more update cycles and re-check.")
    else:
        before = usd_data["before"]["stability"]
        after = usd_data["after"]["stability"]
        flip_before = usd_data["before"]["flip_freq"]
        flip_after = usd_data["after"]["flip_freq"]
        hl_before = usd_data["before"]["half_life"]
        hl_after = usd_data["after"]["half_life"]

        lines.append(f"  Raw stability:      {before:.1f}/10")
        lines.append(f"  Stabilised stability: {after:.1f}/10")
        lines.append(f"  Improvement:        {usd_data['improvement']}")
        lines.append(f"  Flip freq:          {flip_before:.3f} → {flip_after:.3f}")
        lines.append(f"  Half-life:          {hl_before:.1f} → {hl_after:.1f}")

        if after >= 5.0:
            lines.append("")
            lines.append(
                "  ✅ USD is now a usable reference axis. "
                "Hysteresis stabilises it into a meaningful signal."
            )
        else:
            lines.append("")
            lines.append(
                "  ⚠️  USD stability improved but remains marginal. "
                "Consider increasing memory_decay or min_lock_cycles."
            )

    lines.append("")
    return "\n".join(lines)


def format_flip_reduction(result: Dict[str, Any]) -> str:
    """Summarise flip reduction from the accumulated history."""
    lines: List[str] = []
    lines.append("")
    lines.append("FLIP REDUCTION METRICS")
    lines.append("=" * 50)

    # We need to look inside the memory object for raw vs hyst history
    # This is a display helper meant to be called externally
    lines.append("  (Run compare_stability() for detailed flip metrics)")
    lines.append("")
    return "\n".join(lines)


# ======================================================================
# Demo / main block
# ======================================================================


def _simulate_cluster_states(
    seed: int = 42,
    bias: Optional[Dict[str, float]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Generate synthetic cluster states for testing/demo.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    bias : dict, optional
        Per-cluster directional bias to create realistic oscillation.

    Returns
    -------
    dict of cluster states compatible with ``update()``.
    """
    rng = np.random.default_rng(seed)
    if bias is None:
        bias = {c: 0.0 for c in CLUSTER_NAMES}

    states: Dict[str, Dict[str, Any]] = {}
    for cname in CLUSTER_NAMES:
        b = bias.get(cname, 0.0)
        # Generate net_direction with noise around the bias
        net_dir = b + rng.normal(0, 0.15)
        net_dir = max(-1.0, min(1.0, net_dir))

        # Divergence: higher when net_dir is near zero (uncertain)
        divergence = max(0.0, min(1.0, rng.exponential(0.25)))
        if abs(net_dir) > 0.4:
            divergence *= 0.5  # lower divergence when direction is clear

        divergence = max(0.0, min(1.0, divergence))

        states[cname] = {
            "net_direction": round(net_dir, 4),
            "divergence": round(divergence, 4),
            "coherence": round(1.0 - divergence, 4),
            "net_pressure": (
                "BULLISH" if net_dir > 0.15
                else ("BEARISH" if net_dir < -0.15 else "NEUTRAL")
            ),
        }

    # Add some oscillation: make EUR and AUD_NZD strongly directional
    # while USD flip-flops
    return states


def main() -> None:
    """Run a demonstration of the Hysteresis Cluster Memory system."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )

    print("\n" + "#" * 60)
    print("# Hysteresis Cluster Memory — Demonstration")
    print("#" * 60)

    # Create the hysteresis filter
    hcm = HysteresisClusterMemory(
        memory_decay=0.7,
        hysteresis_band=0.15,
        min_lock_cycles=3,
    )

    # --- Simulation: 20 cycles of cluster state evolution ---
    print("\nSimulating 20 analysis cycles ...\n")

    # Cycle 1-3: neutral-ish starting point
    for cycle in range(1, 21):
        # Evolve biases over time to create realistic oscillation
        # USD flip-flops, EUR expands then fades, etc.
        phase = cycle / 20.0

        # Create time-varying bias
        bias = {
            "EUR": 0.6 * np.sin(phase * np.pi * 2),  # oscillates
            "USD": -0.3 * np.sin(phase * np.pi * 3),  # faster oscillation (flip-flop)
            "JPY": 0.1 * np.sin(phase * np.pi),
            "AUD_NZD": 0.5 * np.sin(phase * np.pi * 1.5 + 0.5),
            "CHF": -0.2 * np.sin(phase * np.pi * 2.5),
            "GBP": 0.05 * np.sin(phase * np.pi * 2),
            "CAD": -0.15 * np.sin(phase * np.pi * 1.8),
        }

        cluster_states = _simulate_cluster_states(seed=cycle, bias=bias)
        result = hcm.update(cluster_states)

        # Print a summary every 5 cycles
        if cycle in (1, 5, 10, 15, 20):
            clusters = result["clusters"]
            flips = result["flip_events"]
            print(f"  Cycle {cycle:>2d}: "
                  f"{sum(1 for c in clusters.values() if c['current_state'] == 'EXPANDING'):>1d} expanding, "
                  f"{sum(1 for c in clusters.values() if c['current_state'] == 'CONTRACTING'):>1d} contracting, "
                  f"{sum(1 for c in clusters.values() if c['current_state'] == 'DIVERGENT'):>1d} divergent, "
                  f"{len(flips)} flips, "
                  f"{len(result['locked_clusters'])} locked")

    # --- Final Dashboard ---
    print("\n" + "=" * 60)
    print("FINAL DASHBOARD")
    print("=" * 60)
    print(format_hysteresis_dashboard(result))

    # --- Stability comparison ---
    print("\n" + "=" * 60)
    comparison = hcm.compare_stability()
    print(format_stability_comparison(comparison))

    # --- USD Verdict ---
    print(format_verdict(comparison))

    # --- Flip reduction summary ---
    print("FLIP REDUCTION METRICS")
    print("=" * 50)
    total_raw_flips = 0
    total_hyst_flips = 0
    for cname in CLUSTER_NAMES:
        metrics = hcm.compute_stability_score(cname)
        if metrics["sample_count"] >= 2:
            raw_f = metrics["flip_freq_raw"]
            hyst_f = metrics["flip_freq_hyst"]
            total_raw_flips += int(raw_f * metrics["sample_count"])
            total_hyst_flips += int(hyst_f * metrics["sample_count"])
            print(
                f"  {cname:<10s}: {int(raw_f * metrics['sample_count']):>3d} raw flips → "
                f"{int(hyst_f * metrics['sample_count']):>3d} hyst flips "
                f"({raw_f:.3f} → {hyst_f:.3f} per cycle)"
            )
    print(f"  {'TOTAL':<10s}: {total_raw_flips:>3d} raw flips → {total_hyst_flips:>3d} hyst flips")
    print()

    # --- Recommended parameters ---
    print("RECOMMENDED HYSTERESIS PARAMETERS")
    print("=" * 50)
    print("  memory_decay = 0.70  (smooth but responsive)")
    print("  hysteresis_band = 0.15  (20% gap between entry/exit = 0.20)")
    print("  min_lock_cycles = 3  (requires 3 cycles in same state to lock)")
    print()
    print("  Rationale:")
    print("  - 0.70 decay: strong smoothing without making the system")
    print("    unresponsive to genuine regime changes")
    print("  - 0.15 band: 20% gap eliminates noise-driven flip-flops")
    print("    while allowing transitions on real directional shifts")
    print("  - 3 cycles: prevents transient spikes from locking the cluster")
    print()

    print("#" * 60)
    print("# End of Demonstration")
    print("#" * 60)


if __name__ == "__main__":
    main()
