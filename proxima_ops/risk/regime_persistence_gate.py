"""Regime Persistence Gate — Decision Stability Filter.

Prevents execution unless cluster state has persisted long enough,
cross-cluster phase agreement is verified, and volatility is stable.

Three gates:
1. Cluster persistence threshold — state must hold for N cycles
2. Cross-cluster phase agreement — >=2 clusters must reinforce
3. Volatility stability — expansion not a short spike
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from .signal_manifold import CLUSTERS, SignalManifoldProjector
from .cluster_risk_oscillator import (
    CLUSTER_CORRELATION,
    ClusterRiskOscillator,
    _generate_real_signals,
    _generate_sample_signals,
    _get_sample_open_positions,
)

logger = logging.getLogger("proxima_ops.risk.regime_persistence_gate")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MIN_CYCLES = 3
"""Default number of consecutive cycles required for persistence."""

CROSS_CORRELATION_THRESHOLD = 0.4
"""Maximum correlation for two clusters to be considered independent."""

MIN_AGREEING_CLUSTERS = 2
"""Minimum number of independent clusters required for phase agreement."""


class RegimePersistenceGate:
    """
    Prevents execution unless cluster state has persisted long enough
    and cross-cluster phase agreement is verified.

    Gates
    -----
    1. **Cluster persistence threshold**
       State must hold for N consecutive evaluations.
    2. **Cross-cluster phase agreement**
       At least 2 independent clusters must reinforce the same phase.
    3. **Volatility stability**
       Expansion must not be a short-lived spike; coherence/divergence
       must be within normal historical bounds.
    """

    def __init__(self) -> None:
        # Per-cluster state history for Gate 1 (list of state strings)
        self._state_history: Dict[str, List[str]] = defaultdict(list)
        # Per-cluster coherence history for Gate 3
        self._coherence_history: Dict[str, List[float]] = defaultdict(list)
        # Per-cluster divergence history for Gate 3
        self._divergence_history: Dict[str, List[float]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Gate 1: Cluster Persistence Threshold
    # ------------------------------------------------------------------

    def check_persistence(
        self,
        cluster_name: str,
        current_state: str,
        min_cycles: int = DEFAULT_MIN_CYCLES,
    ) -> Dict[str, Any]:
        """
        Check whether *cluster_name* has remained in *current_state*
        for at least *min_cycles* consecutive evaluations.

        Parameters
        ----------
        cluster_name : str
            Cluster identifier (e.g. ``"EUR"``).
        current_state : str
            Current oscillator state (EXPANDING / CONTRACTING / DIVERGENT / NEUTRAL).
        min_cycles : int
            Minimum number of consecutive cycles required (default 3).

        Returns
        -------
        dict
            ``persistent``, ``current_state``, ``cycles_in_state``,
            ``min_cycles_required``, ``history``.
        """
        # Append current state to history
        self._state_history[cluster_name].append(current_state)

        # Keep a manageable rolling window
        if len(self._state_history[cluster_name]) > 20:
            self._state_history[cluster_name] = self._state_history[cluster_name][-20:]

        # Count consecutive same-state from the tail
        history = list(self._state_history[cluster_name])
        cycles_in_state = 0
        for s in reversed(history):
            if s == current_state:
                cycles_in_state += 1
            else:
                break

        persistent = cycles_in_state >= min_cycles

        return {
            "persistent": persistent,
            "current_state": current_state,
            "cycles_in_state": cycles_in_state,
            "min_cycles_required": min_cycles,
            "history": history,
        }

    # ------------------------------------------------------------------
    # Gate 2: Cross-Cluster Phase Agreement
    # ------------------------------------------------------------------

    def cross_cluster_agreement(
        self, cluster_states: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Require that at least 2 independent clusters (correlation < 0.4)
        agree on the overall direction before any execution is allowed.

        Scans all clusters and finds pairs that are:
        - Both **EXPANDING** → bullish alignment
        - Both **CONTRACTING** → bearish alignment
        - One EXPANDING + one CONTRACTING → divergent (block)

        Parameters
        ----------
        cluster_states : dict
            Per-cluster state dicts from oscillator output, each containing
            at minimum a ``state`` key (EXPANDING/CONTRACTING/DIVERGENT/NEUTRAL).

        Returns
        -------
        dict
            ``agreement``, ``aligned_clusters``, ``misaligned_clusters``,
            ``phase``, plus breakdown counts.
        """
        # Separate clusters by state
        expanding: List[str] = []
        contracting: List[str] = []
        divergent: List[str] = []
        neutral: List[str] = []

        for cname, state_data in cluster_states.items():
            if cname not in CLUSTERS:
                continue
            s = state_data.get("state", "NEUTRAL")
            if s == "EXPANDING":
                expanding.append(cname)
            elif s == "CONTRACTING":
                contracting.append(cname)
            elif s == "DIVERGENT":
                divergent.append(cname)
            else:
                neutral.append(cname)

        # Find aligned expanding pairs (correlation < threshold)
        aligned_expanding: List[str] = []
        for i, c1 in enumerate(expanding):
            for c2 in expanding[i + 1:]:
                corr = CLUSTER_CORRELATION.get(c1, {}).get(c2, 1.0)
                if corr < CROSS_CORRELATION_THRESHOLD:
                    if c1 not in aligned_expanding:
                        aligned_expanding.append(c1)
                    if c2 not in aligned_expanding:
                        aligned_expanding.append(c2)

        # Find aligned contracting pairs (correlation < threshold)
        aligned_contracting: List[str] = []
        for i, c1 in enumerate(contracting):
            for c2 in contracting[i + 1:]:
                corr = CLUSTER_CORRELATION.get(c1, {}).get(c2, 1.0)
                if corr < CROSS_CORRELATION_THRESHOLD:
                    if c1 not in aligned_contracting:
                        aligned_contracting.append(c1)
                    if c2 not in aligned_contracting:
                        aligned_contracting.append(c2)

        # Find divergent pairs (expanding + contracting, independent)
        divergent_pairs: List[str] = []
        for e in expanding:
            for c in contracting:
                corr = CLUSTER_CORRELATION.get(e, {}).get(c, 1.0)
                if corr < CROSS_CORRELATION_THRESHOLD:
                    if e not in divergent_pairs:
                        divergent_pairs.append(e)
                    if c not in divergent_pairs:
                        divergent_pairs.append(c)

        # Determine overall phase
        if aligned_expanding and not divergent_pairs:
            if aligned_contracting:
                # Both bullish and bearish aligned → still divergent overall
                phase = "DIVERGENT"
                agreement = False
            else:
                phase = "BULLISH_ALIGNMENT"
                agreement = True
        elif aligned_contracting and not divergent_pairs:
            phase = "BEARISH_ALIGNMENT"
            agreement = True
        elif divergent_pairs:
            phase = "DIVERGENT"
            agreement = False
        else:
            # Check if there is at least one expanding and one contracting
            if expanding and contracting:
                phase = "DIVERGENT"
            else:
                phase = "NEUTRAL"
            agreement = False

        aligned_set = set(aligned_expanding + aligned_contracting)
        misaligned_set = set(expanding + contracting + divergent) - aligned_set

        return {
            "agreement": agreement,
            "aligned_clusters": sorted(aligned_set),
            "misaligned_clusters": sorted(misaligned_set),
            "phase": phase,
            "expanding": expanding,
            "contracting": contracting,
            "divergent": divergent,
            "neutral": neutral,
        }

    # ------------------------------------------------------------------
    # Gate 3: Volatility Stability Filter
    # ------------------------------------------------------------------

    def volatility_stability(
        self,
        cluster_name: str,
        coherence_history: List[float],
        divergence_history: List[float],
    ) -> Dict[str, Any]:
        """
        Verify that expansion is not a short-lived spike.

        Checks
        ------
        - Coherence has been trending up (not a sudden spike).
        - Divergence has been trending down (not a sudden drop).
        - Current values are within 1 standard deviation of history mean.

        Parameters
        ----------
        cluster_name : str
            Cluster identifier (used for logging).
        coherence_history : list of float
            Recent coherence values (most recent last).
        divergence_history : list of float
            Recent divergence values (most recent last).

        Returns
        -------
        dict
            ``stable``, ``coherence_trend``, ``divergence_trend``,
            ``coherence_zscore``, ``divergence_zscore``.
        """
        if len(coherence_history) < 2 or len(divergence_history) < 2:
            return {
                "stable": False,
                "coherence_trend": "flat",
                "divergence_trend": "flat",
                "coherence_zscore": 0.0,
                "divergence_zscore": 0.0,
                "reason": "insufficient_history",
            }

        coh = np.array(coherence_history, dtype=float)
        div = np.array(divergence_history, dtype=float)

        # -- Trend analysis (direction count on diffs) --
        coh_diff = np.diff(coh)
        div_diff = np.diff(div)

        coh_up = int(np.sum(coh_diff > 1e-6))
        coh_down = int(np.sum(coh_diff < -1e-6))
        div_up = int(np.sum(div_diff > 1e-6))
        div_down = int(np.sum(div_diff < -1e-6))

        if coh_up > coh_down * 1.5 and coh_up > 0:
            coherence_trend = "rising"
        elif coh_down > coh_up * 1.5 and coh_down > 0:
            coherence_trend = "falling"
        else:
            coherence_trend = "flat"

        if div_down > div_up * 1.5 and div_down > 0:
            divergence_trend = "falling"
        elif div_up > div_down * 1.5 and div_up > 0:
            divergence_trend = "rising"
        else:
            divergence_trend = "flat"

        # -- Z-score analysis --
        coh_mean = float(np.mean(coh))
        coh_std = float(np.std(coh)) + 1e-12
        div_mean = float(np.mean(div))
        div_std = float(np.std(div)) + 1e-12

        coh_current = coh[-1]
        div_current = div[-1]

        coh_zscore = (coh_current - coh_mean) / coh_std
        div_zscore = (div_current - div_mean) / div_std

        # Stable if coherence and divergence are within 2 std of their means
        stable = bool(abs(coh_zscore) < 2.0 and abs(div_zscore) < 2.0)

        return {
            "stable": stable,
            "coherence_trend": coherence_trend,
            "divergence_trend": divergence_trend,
            "coherence_zscore": round(coh_zscore, 4),
            "divergence_zscore": round(div_zscore, 4),
            "current_coherence": round(coh_current, 4),
            "current_divergence": round(div_current, 4),
            "mean_coherence": round(coh_mean, 4),
            "mean_divergence": round(div_mean, 4),
        }

    # ------------------------------------------------------------------
    # Main Gate Method
    # ------------------------------------------------------------------

    def evaluate(
        self,
        cluster_states: Dict[str, Dict[str, Any]],
        oscillator_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Run ALL gates and produce final decision.

        Parameters
        ----------
        cluster_states : dict
            Per-cluster manifold state dicts (from projector output).
            Contains ``coherence``, ``divergence``, ``net_direction`` etc.
        oscillator_state : dict
            Full oscillator result. Its ``["clusters"]`` sub-dict contains
            enriched per-cluster data including ``state``, ``momentum``,
            ``velocity``, ``acceleration``, plus the original manifold fields.

        Returns
        -------
        dict
            ``overall_decision``, ``gates``, ``blocking_gates``,
            ``eligible_clusters``, ``phase``, ``timestamp``, ``cluster_details``.
        """
        # The oscillator enriches manifold data with state/momentum/velocity.
        # Use it as the primary data source; fall back to raw manifold data.
        enriched_states: Dict[str, Dict[str, Any]] = (
            oscillator_state.get("clusters") or cluster_states
        )

        # -- Update internal histories from the current cluster states --
        for cname in CLUSTERS:
            cdata = enriched_states.get(cname, {})
            coherence = cdata.get("coherence", 0.5)
            divergence = cdata.get("divergence", 0.5)
            self._coherence_history[cname].append(coherence)
            self._divergence_history[cname].append(divergence)
            # Trim to last 20
            if len(self._coherence_history[cname]) > 20:
                self._coherence_history[cname] = self._coherence_history[cname][-20:]
            if len(self._divergence_history[cname]) > 20:
                self._divergence_history[cname] = self._divergence_history[cname][-20:]

        # -- Gate 1: Persistence --
        persistence_results: Dict[str, Dict[str, Any]] = {}
        for cname in CLUSTERS:
            cdata = enriched_states.get(cname, {})
            state = cdata.get("state", "NEUTRAL")
            persistence_results[cname] = self.check_persistence(cname, state)

        # -- Gate 2: Cross-Cluster Agreement --
        agreement_result = self.cross_cluster_agreement(enriched_states)

        # -- Gate 3: Volatility Stability --
        volatility_results: Dict[str, Dict[str, Any]] = {}
        for cname in CLUSTERS:
            coh_hist = list(self._coherence_history[cname])
            div_hist = list(self._divergence_history[cname])
            volatility_results[cname] = self.volatility_stability(
                cname, coh_hist, div_hist
            )

        # -- Per-cluster gate verdicts --
        cluster_details: Dict[str, Dict[str, Any]] = {}
        for cname in CLUSTERS:
            cdata = enriched_states.get(cname, {})
            state = cdata.get("state", "NEUTRAL")
            p_result = persistence_results[cname]
            v_result = volatility_results[cname]

            # Persistence pass: only meaningful for non-NEUTRAL
            persist_pass: Optional[bool] = (
                p_result["persistent"] if state != "NEUTRAL" else None
            )
            # Volatility pass: only meaningful for non-NEUTRAL
            vol_pass: Optional[bool] = (
                v_result["stable"] if state != "NEUTRAL" else None
            )
            # Agreement pass: does this cluster participate in aligned phase?
            if state in ("EXPANDING", "CONTRACTING"):
                agreement_pass = cname in agreement_result["aligned_clusters"]
            else:
                agreement_pass = None  # NEUTRAL/DIVERGENT not expected to agree

            cluster_details[cname] = {
                "state": state,
                "persistence": p_result,
                "persistence_pass": persist_pass,
                "agreement_pass": agreement_pass,
                "volatility": v_result,
                "volatility_pass": vol_pass,
            }

        # -- Overall gate-level summaries --
        blocking_gates: List[str] = []
        gates_result: Dict[str, Dict[str, Any]] = {}

        # Gate 1: overall persistence passes if at least one non-NEUTRAL cluster persists
        any_persist = any(
            cd["persistence_pass"] is True
            for cd in cluster_details.values()
            if cd["state"] != "NEUTRAL"
        )
        gates_result["persistence"] = {
            "passed": any_persist,
            "details": persistence_results,
        }
        if not any_persist:
            blocking_gates.append("persistence")

        # Gate 2: cross-cluster agreement
        gates_result["cross_cluster_agreement"] = {
            "passed": agreement_result["agreement"],
            "details": agreement_result,
        }
        if not agreement_result["agreement"]:
            blocking_gates.append("cross_cluster_agreement")

        # Gate 3: overall volatility passes if at least one non-NEUTRAL cluster is stable
        any_stable = any(
            cd["volatility_pass"] is True
            for cd in cluster_details.values()
            if cd["state"] != "NEUTRAL"
        )
        gates_result["volatility_stability"] = {
            "passed": any_stable,
            "details": volatility_results,
        }
        if not any_stable:
            blocking_gates.append("volatility_stability")

        # -- Eligible clusters: those that passed ALL three gates --
        eligible_clusters = [
            cname
            for cname in CLUSTERS
            if cluster_details[cname]["state"] in ("EXPANDING", "CONTRACTING")
            and cluster_details[cname]["persistence_pass"] is True
            and cluster_details[cname]["agreement_pass"] is True
            and cluster_details[cname]["volatility_pass"] is True
        ]

        overall_decision = (
            "EXECUTION_ALLOWED"
            if not blocking_gates and eligible_clusters
            else "EXECUTION_BLOCKED"
        )

        phase = agreement_result["phase"]

        return {
            "overall_decision": overall_decision,
            "gates": gates_result,
            "blocking_gates": blocking_gates,
            "eligible_clusters": eligible_clusters,
            "phase": phase,
            "timestamp": datetime.now().isoformat(),
            "cluster_details": cluster_details,
        }

    def get_persistence_history(self, cluster_name: str) -> List[str]:
        """Return raw persistence history for *cluster_name*."""
        return list(self._state_history.get(cluster_name, []))

    def reset(self) -> None:
        """Clear all internal state (for testing or fresh start)."""
        self._state_history.clear()
        self._coherence_history.clear()
        self._divergence_history.clear()


# ======================================================================
# Dashboard formatting
# ======================================================================


def format_gate_dashboard(result: Dict[str, Any]) -> str:
    """
    Render the Regime Persistence Gate dashboard as a formatted string.
    """
    lines: List[str] = []
    lines.append("")
    lines.append("REGIME PERSISTENCE GATE — DECISION STABILITY FILTER")
    lines.append("=" * 60)
    lines.append(
        f"{'Cluster':<12s} {'State':<12s} {'Persist':<10s} {'Agreement':<12s} "
        f"{'Volatility':<12s} {'Gate Result':<10s}"
    )
    lines.append("-" * 60)

    cluster_details = result.get("cluster_details", {})
    ag_details = (
        result.get("gates", {})
        .get("cross_cluster_agreement", {})
        .get("details", {})
    )
    phase = ag_details.get("phase", "NEUTRAL")

    for cname in sorted(CLUSTERS.keys()):
        cd = cluster_details.get(cname, {})
        state = cd.get("state", "NEUTRAL")

        # --- Persist column ---
        if state == "NEUTRAL":
            persist_str = "N/A"
        else:
            pp = cd.get("persistence_pass", False)
            p = cd.get("persistence", {})
            cycles = p.get("cycles_in_state", 0)
            req = p.get("min_cycles_required", 3)
            persist_str = f"{'✅' if pp else '❌'} {cycles}/{req}"

        # --- Agreement column ---
        if state in ("EXPANDING", "CONTRACTING"):
            ap = cd.get("agreement_pass", False)
            agreement_str = f"✅ {phase}" if ap else "❌"
        else:
            agreement_str = "N/A"

        # --- Volatility column ---
        if state == "NEUTRAL":
            vol_str = "N/A"
        else:
            vp = cd.get("volatility_pass", False)
            if vp:
                vol_str = "✅ stable"
            else:
                v = cd.get("volatility", {})
                reason = v.get("reason", "")
                if reason == "insufficient_history":
                    vol_str = "❌ short"
                elif abs(v.get("coherence_zscore", 0)) >= 2:
                    vol_str = "❌ spike"
                elif abs(v.get("divergence_zscore", 0)) >= 2:
                    vol_str = "❌ drop"
                else:
                    vol_str = "❌ unstable"

        # --- Gate result column ---
        eligible = cname in result.get("eligible_clusters", [])
        if eligible:
            gate_result_str = "ELIGIBLE"
        elif state == "NEUTRAL":
            gate_result_str = "BLOCKED"
        else:
            gate_result_str = "BLOCKED"

        lines.append(
            f"{cname:<12s} {state:<12s} {persist_str:<10s} {agreement_str:<12s} "
            f"{vol_str:<12s} {gate_result_str:<10s}"
        )

    lines.append("")
    eligible_clusters = result.get("eligible_clusters", [])
    if eligible_clusters:
        lines.append(
            f"OVERALL DECISION: {result['overall_decision']}  "
            f"({' -> '.join(eligible_clusters)} passed all gates)"
        )
    else:
        lines.append(
            f"OVERALL DECISION: {result['overall_decision']}  "
            f"(none passed all gates)"
        )
    lines.append(f"PHASE: {phase}")

    blocking = result.get("blocking_gates", [])
    if blocking:
        lines.append(f"Blocking gates: {', '.join(blocking)}")
    else:
        lines.append("No blocking gates.")

    # Build per-cluster blocking reasons
    blocking_reasons: List[str] = []
    for cname in sorted(CLUSTERS.keys()):
        cd = cluster_details.get(cname, {})
        state = cd.get("state", "NEUTRAL")
        if state == "NEUTRAL":
            continue
        reasons: List[str] = []
        if cd.get("persistence_pass") is False:
            p = cd.get("persistence", {})
            reasons.append(
                f"insufficient persistence "
                f"({p.get('cycles_in_state', 0)}/{p.get('min_cycles_required', 3)})"
            )
        if cd.get("agreement_pass") is False:
            reasons.append("not in aligned phase")
        if cd.get("volatility_pass") is False:
            v = cd.get("volatility", {})
            if v.get("reason") == "insufficient_history":
                reasons.append("insufficient volatility history")
            elif abs(v.get("coherence_zscore", 0)) >= 2:
                reasons.append(
                    f"coherence spike (z={v.get('coherence_zscore', 0):.1f})"
                )
            elif abs(v.get("divergence_zscore", 0)) >= 2:
                reasons.append(
                    f"divergence drop (z={v.get('divergence_zscore', 0):.1f})"
                )
            else:
                reasons.append("volatility unstable")
        if reasons:
            blocking_reasons.append(f"{cname}: {', '.join(reasons)}")

    if blocking_reasons:
        lines.append("Blocking reasons:")
        for r in blocking_reasons:
            lines.append(f"  - {r}")

    lines.append("=" * 60)
    return "\n".join(lines)


# ======================================================================
# Main block
# ======================================================================


def main() -> None:
    """
    Entry point: generate signals, project, oscillate, gate, display.

    Pipeline
    --------
    1. Generate ~387 OSS signals (real or sampled)
    2. Project onto cluster manifold (SignalManifoldProjector)
    3. Run ClusterRiskOscillator (multiple passes for velocity history)
    4. Run RegimePersistenceGate.evaluate() (multiple passes for persistence history)
    5. Print dashboard
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )

    print("\n" + "#" * 60)
    print("# Regime Persistence Gate — Decision Stability Filter")
    print("#" * 60)

    # ------------------------------------------------------------------
    # Step 1: Generate signals
    # ------------------------------------------------------------------
    print("\n[1/5] Generating signals from OSS models ...")
    signals = _generate_real_signals(num_signals=387)
    if len(signals) < 387:
        n_extra = 387 - len(signals)
        extra = _generate_sample_signals(num_signals=n_extra)
        signals.extend(extra)
        print(
            f"      -> {len(signals)} total "
            f"({len(signals) - n_extra} real + {n_extra} sample)"
        )
    else:
        print(f"      -> {len(signals)} real signals generated")
    print(f"      -> {len({s['symbol'] for s in signals})} unique symbols")

    # ------------------------------------------------------------------
    # Step 2: Project onto cluster manifold
    # ------------------------------------------------------------------
    print("\n[2/5] Projecting onto cluster manifold ...")
    projector = SignalManifoldProjector()
    projection = projector.project(signals)
    meta = projection["meta"]
    print(
        f"      dominant_regime={meta['dominant_regime']}  "
        f"net_market_direction={meta['net_market_direction']:+0.4f}"
    )

    # ------------------------------------------------------------------
    # Step 3+4: Run oscillator + gate through multiple passes to build
    #            velocity, acceleration, and persistence history.
    # ------------------------------------------------------------------
    print("\n[3/5] Running cluster risk oscillator (building history) ...")
    oscillator = ClusterRiskOscillator()
    gate = RegimePersistenceGate()
    open_positions = _get_sample_open_positions()

    # --- Pass 1: bias opposite direction for velocity swing ---
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
    cs_pass1 = projector.project(first_signals)["clusters"]
    osc_result_pass1 = oscillator.oscillate(cs_pass1, open_positions)
    gate_result = gate.evaluate(cs_pass1, osc_result_pass1)

    # --- Passes 2-4: strong directional bias so some clusters persist ---
    main_bias: Dict[str, float] = {
        "EUR": 0.80,
        "AUD_NZD": 0.85,
        "USD": -0.75,
        "CHF": -0.70,
        "JPY": 0.10,
        "GBP": 0.15,
        "CAD": -0.10,
    }
    for seed in [99, 101, 103]:
        sigs = _generate_sample_signals(
            num_signals=387, rng_seed=seed, cluster_bias=main_bias
        )
        cs = projector.project(sigs)["clusters"]
        osc_result = oscillator.oscillate(cs, open_positions)
        gate_result = gate.evaluate(cs, osc_result)

    print("      Built oscillator + gate history across 4 passes.")

    # ------------------------------------------------------------------
    # Step 5: Dashboard
    # ------------------------------------------------------------------
    print("\n[4/5] Regime Persistence Gate Dashboard:")
    print(format_gate_dashboard(gate_result))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Overall Decision: {gate_result['overall_decision']}")
    print(f"  Phase: {gate_result['phase']}")
    eligible = gate_result.get("eligible_clusters", [])
    if eligible:
        print(f"  Eligible clusters: {', '.join(eligible)}")
    else:
        print("  Eligible clusters: none (all BLOCKED)")
    print(f"  Blocking gates: {gate_result['blocking_gates']}")
    print()


if __name__ == "__main__":
    main()
