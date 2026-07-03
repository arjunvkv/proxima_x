"""RFE Arbitration Layer — Exit Authority Bridge.

Transforms raw RFE (Regime Field Energy) detection into graded probabilistic
exit authority. RFE does NOT directly trigger exits. Instead, it produces an
EXIT PRESSURE SCORE (0-1) that increases over time as divergence persists.
Transient divergence decays automatically.

RFE States (graded escalation):
    INFO       — divergence observed, no action (score: 0.0-0.15)
    WATCH      — persistence building (score: 0.15-0.35)
    WARNING    — structural decay likely (score: 0.35-0.60)
    EXIT_PREP  — high probability reversal (score: 0.60-0.85)
    EXIT       — execution allowed (score: 0.85-1.00)

Dependencies
------------
- ``hysteresis_cluster_memory.py`` for hysteresis-stabilised cluster states.
- ``signal_manifold.py`` for symbol→cluster mapping.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .signal_manifold import symbol_to_primary_cluster

logger = logging.getLogger("proxima_ops.risk.rfe_arbitration")

# ---------------------------------------------------------------------------
# Constants — default parameters for the arbitration layer
# ---------------------------------------------------------------------------

DEFAULT_DECAY_FACTOR: float = 0.15
"""Pressure lost per resolution cycle."""

DEFAULT_ACCUMULATION_FACTOR: float = 0.10
"""Pressure gained per divergence cycle."""

DEFAULT_MAX_SINGLE_CYCLE_INCREASE: float = 0.15
"""Cap on per-cycle pressure increase (prevents overreaction)."""

DEFAULT_RESOLUTION_CYCLES: int = 3
"""Cycles of resolution required to fully clear pressure."""

MIN_CYCLES_IN_STATE: int = 1
"""Minimum cycles a state must be held before it can change (anti-flicker)."""

# Component weights for exit pressure score
WEIGHT_DIVERGENCE: float = 0.40
WEIGHT_PERSISTENCE: float = 0.25
WEIGHT_HYSTERESIS_DECAY: float = 0.20
WEIGHT_PNL_REGIEM: float = 0.15

# ---------------------------------------------------------------------------
# RFE State Machine
# ---------------------------------------------------------------------------


class RFEState:
    """RFE state constants and thresholds."""

    INFO = "INFO"
    WATCH = "WATCH"
    WARNING = "WARNING"
    EXIT_PREP = "EXIT_PREP"
    EXIT = "EXIT"

    # Ordered list for progression comparison
    ORDER = [INFO, WATCH, WARNING, EXIT_PREP, EXIT]

    THRESHOLDS: Dict[str, Tuple[float, float]] = {
        INFO: (0.00, 0.15),
        WATCH: (0.15, 0.35),
        WARNING: (0.35, 0.60),
        EXIT_PREP: (0.60, 0.85),
        EXIT: (0.85, 1.00),
    }

    @classmethod
    def from_score(cls, score: float) -> str:
        """Map a pressure score to its corresponding RFE state."""
        for state in cls.ORDER:
            lo, hi = cls.THRESHOLDS[state]
            if lo <= score < hi:
                return state
        return cls.EXIT  # score == 1.0

    @classmethod
    def index(cls, state: str) -> int:
        """Return the ordinal index of a state (0-4)."""
        try:
            return cls.ORDER.index(state)
        except ValueError:
            return 0

    @classmethod
    def escalation_allowed(cls, current: str, target: str) -> bool:
        """Check if we can skip intermediate states on escalation (increase only).
        
        On increase: skip states is allowed (INFO→WARNING OK).
        On decrease: must pass through each state (EXIT→EXIT_PREP→WARNING→...).
        """
        cur_idx = cls.index(current)
        tgt_idx = cls.index(target)

        if tgt_idx >= cur_idx:
            # Escalating — skip states allowed
            return True
        else:
            # De-escalating — must go step by step
            return (cur_idx - tgt_idx) == 1

    @classmethod
    def overall_risk(cls, max_pressure: float) -> str:
        """Classify overall portfolio risk level."""
        if max_pressure >= 0.85:
            return "CRITICAL"
        if max_pressure >= 0.60:
            return "HIGH"
        if max_pressure >= 0.35:
            return "MEDIUM"
        return "LOW"


# ---------------------------------------------------------------------------
# Temporal Decay Model
# ---------------------------------------------------------------------------


class TemporalDecayModel:
    """
    Ensures transient divergence decays automatically.

    Rules:
    1. Each cycle WITHOUT divergence decreases pressure by decay_factor
    2. Each cycle WITH continued divergence increases pressure by accumulation_factor
    3. Pressure can never reach 1.0 in a single cycle (prevents overreaction)
    4. Pressure decays to 0 if divergence resolves for N consecutive cycles

    Parameters
    ----------
    decay_factor : float, default=0.15
        Pressure lost per resolution cycle.
    accumulation_factor : float, default=0.10
        Pressure gained per divergence cycle.
    max_single_cycle_increase : float, default=0.15
        Cap on per-cycle pressure increase.
    resolution_cycles : int, default=3
        Cycles of resolution required to fully clear pressure.
    """

    def __init__(
        self,
        decay_factor: float = DEFAULT_DECAY_FACTOR,
        accumulation_factor: float = DEFAULT_ACCUMULATION_FACTOR,
        max_single_cycle_increase: float = DEFAULT_MAX_SINGLE_CYCLE_INCREASE,
        resolution_cycles: int = DEFAULT_RESOLUTION_CYCLES,
    ) -> None:
        if not 0.0 <= decay_factor <= 1.0:
            raise ValueError(f"decay_factor must be in [0, 1], got {decay_factor}")
        if not 0.0 <= accumulation_factor <= 1.0:
            raise ValueError(
                f"accumulation_factor must be in [0, 1], got {accumulation_factor}"
            )
        if not 0.0 <= max_single_cycle_increase <= 1.0:
            raise ValueError(
                f"max_single_cycle_increase must be in [0, 1], "
                f"got {max_single_cycle_increase}"
            )
        if resolution_cycles < 1:
            raise ValueError(
                f"resolution_cycles must be >= 1, got {resolution_cycles}"
            )

        self.decay_factor = decay_factor
        self.accumulation_factor = accumulation_factor
        self.max_single_cycle_increase = max_single_cycle_increase
        self.resolution_cycles = resolution_cycles

    def decay(self, current_pressure: float, cycles_since_divergence: int) -> float:
        """Reduce pressure when divergence resolves.

        Parameters
        ----------
        current_pressure : float
            Current exit pressure score (0-1).
        cycles_since_divergence : int
            Number of consecutive cycles without divergence.

        Returns
        -------
        float
            New pressure after decay.
        """
        if cycles_since_divergence <= 0:
            return current_pressure

        decay_amount = self.decay_factor * cycles_since_divergence
        new_pressure = max(0.0, current_pressure - decay_amount)

        # Rule 4: If resolved for resolution_cycles, pressure goes to 0
        if cycles_since_divergence >= self.resolution_cycles:
            new_pressure = 0.0

        return round(new_pressure, 4)

    def accumulate(
        self, current_pressure: float, cycles_in_divergence: int
    ) -> float:
        """Increase pressure when divergence persists.

        Parameters
        ----------
        current_pressure : float
            Current exit pressure score (0-1).
        cycles_in_divergence : int
            Number of consecutive cycles with divergence.

        Returns
        -------
        float
            New pressure after accumulation.
        """
        if cycles_in_divergence <= 0:
            return current_pressure

        accumulation = self.accumulation_factor * cycles_in_divergence
        # Cap per-cycle increase
        accumulation = min(accumulation, self.max_single_cycle_increase)

        new_pressure = min(1.0, current_pressure + accumulation)
        return round(new_pressure, 4)


# ---------------------------------------------------------------------------
# RFE Arbitration Layer
# ---------------------------------------------------------------------------


class RFEArbitrationLayer:
    """
    Transforms raw RFE detection into graded probabilistic exit authority.

    RFE does NOT directly trigger exits.
    Instead, it produces EXIT PRESSURE SCORE (0-1) that increases over time
    as divergence persists. Transient divergence decays automatically.

    Parameters
    ----------
    decay_factor : float, default=0.15
        Pressure lost per resolution cycle.
    accumulation_factor : float, default=0.10
        Pressure gained per divergence cycle.
    max_single_cycle_increase : float, default=0.15
        Cap on per-cycle pressure increase.
    resolution_cycles : int, default=3
        Cycles of resolution required to fully clear pressure.
    """

    def __init__(
        self,
        decay_factor: float = DEFAULT_DECAY_FACTOR,
        accumulation_factor: float = DEFAULT_ACCUMULATION_FACTOR,
        max_single_cycle_increase: float = DEFAULT_MAX_SINGLE_CYCLE_INCREASE,
        resolution_cycles: int = DEFAULT_RESOLUTION_CYCLES,
    ) -> None:
        # Temporal decay model
        self.temporal = TemporalDecayModel(
            decay_factor=decay_factor,
            accumulation_factor=accumulation_factor,
            max_single_cycle_increase=max_single_cycle_increase,
            resolution_cycles=resolution_cycles,
        )

        # Per-trade persistent state
        self._trade_states: Dict[str, Dict[str, Any]] = {}

        # History for dashboard
        self._pressure_history: Dict[str, List[float]] = {}
        self._state_history: Dict[str, List[str]] = {}
        self._transition_log: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        open_positions: List[Dict[str, Any]],
        cluster_states: Dict[str, Dict[str, Any]],
        hysteresis_state: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Evaluate ALL open positions through the arbitration layer.

        Parameters
        ----------
        open_positions : list of dict
            Each trade dict must contain at least:
            - symbol (str)
            - direction (str): "BUY" or "SELL"
            - current_pnl (float): current unrealised PnL
            - entry_price (float): trade entry price
            - sl (float): stop loss level
            - tp (float): take profit level
        cluster_states : dict
            Current cluster manifold output (from ``SignalManifoldProjector``).
        hysteresis_state : dict
            Current hysteresis-stabilised states (from ``HysteresisClusterMemory``).

        Returns
        -------
        dict
            Complete evaluation result with per-trade breakdown, summary,
            state transitions, and temporal decay info.
        """
        evaluations: Dict[str, Dict[str, Any]] = {}
        trades_at_risk: List[str] = []

        for trade in open_positions:
            symbol = trade.get("symbol", "UNKNOWN")
            trade_id = f"{symbol}_{trade.get('direction', 'BUY')}"

            eval_result = self.compute_exit_pressure(
                trade, cluster_states, hysteresis_state
            )

            evaluations[symbol] = eval_result

            # Track state history
            self._pressure_history.setdefault(symbol, []).append(eval_result["score"])
            self._state_history.setdefault(symbol, []).append(eval_result["state"])

            if eval_result["state"] in (RFEState.WATCH, RFEState.WARNING, RFEState.EXIT_PREP, RFEState.EXIT):
                trades_at_risk.append(symbol)

        # Build summary
        pressures = [e["score"] for e in evaluations.values()]
        states = [e["state"] for e in evaluations.values()]
        max_pressure = max(pressures) if pressures else 0.0
        max_state = max(states, key=lambda s: RFEState.index(s)) if states else RFEState.INFO
        any_exit = any(e["exit_allowed"] for e in evaluations.values())

        summary = {
            "max_pressure": round(max_pressure, 4),
            "max_state": max_state,
            "any_exit_allowed": any_exit,
            "trades_at_risk": trades_at_risk,
            "overall_risk": RFEState.overall_risk(max_pressure),
        }

        # Build transition summary (last 10 per trade)
        transitions = self._format_transitions()

        # Temporal decay info
        decay_info = self._format_temporal_info()

        # Threshold breaches
        breaches = self._detect_threshold_breaches()

        return {
            "evaluations": evaluations,
            "summary": summary,
            "transitions": transitions,
            "temporal": decay_info,
            "breaches": breaches,
            "timestamp": datetime.now().isoformat(),
        }

    def compute_exit_pressure(
        self,
        trade: Dict[str, Any],
        cluster_states: Dict[str, Dict[str, Any]],
        hysteresis_state: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compute exit pressure score (0-1) for a single trade.

        Inputs
        ------
        trade : dict
            Trade data with symbol, direction, current_pnl, entry_price, sl, tp.
        cluster_states : dict
            Current cluster manifold output.
        hysteresis_state : dict
            Current hysteresis-stabilised states.

        Returns
        -------
        dict
            Evaluation result with score, state, components, and transition info.
        """
        symbol = trade.get("symbol", "UNKNOWN")
        trade_dir = str(trade.get("direction", "BUY")).upper()
        trade_id = f"{symbol}_{trade_dir}"
        cluster = symbol_to_primary_cluster(symbol)

        # Get cluster information
        cluster_manifold = cluster_states.get(cluster, {})
        cluster_net_dir = cluster_manifold.get("net_direction", 0.0)
        cluster_divergence = cluster_manifold.get("divergence", 0.0)

        # Get hysteresis-stabilised information
        hyst_clusters = hysteresis_state.get("clusters", {}) if isinstance(hysteresis_state, dict) else {}
        hyst_info = hyst_clusters.get(cluster, {})
        hyst_state = hyst_info.get("current_state", "NEUTRAL")
        decayed_score = hyst_info.get("decayed_score", 0.0)

        # Load or initialise per-trade persistent state
        trade_state = self._trade_states.setdefault(trade_id, {
            "current_pressure": 0.0,
            "previous_state": RFEState.INFO,
            "current_state": RFEState.INFO,
            "cycles_in_state": 0,
            "cycles_in_divergence": 0,
            "cycles_since_divergence": 0,
            "peak_pnl": trade.get("current_pnl", 0.0),
            "previous_pressure": 0.0,
            "breach_threshold": False,
        })

        # Determine if trade direction diverges from cluster direction
        # cluster_net_dir > 0.15 → BULLISH, < -0.15 → BEARISH
        cluster_direction = 1 if cluster_net_dir > 0.15 else (-1 if cluster_net_dir < -0.15 else 0)
        trade_is_buy = 1 if trade_dir == "BUY" else -1
        is_divergent = (cluster_direction != 0 and trade_is_buy != cluster_direction)

        # Track divergence cycles
        if is_divergent or cluster_divergence > 0.5:
            trade_state["cycles_in_divergence"] += 1
            trade_state["cycles_since_divergence"] = 0
        else:
            trade_state["cycles_in_divergence"] = 0
            trade_state["cycles_since_divergence"] += 1

        # --- Component A: Cluster Divergence (weight: 0.40) ---
        div_cycles = trade_state["cycles_in_divergence"]
        if is_divergent:
            divergence_score = min(1.0, cluster_divergence + 0.1 * div_cycles)
        else:
            divergence_score = cluster_divergence * 0.3  # reduced when aligned
        divergence_score = min(1.0, max(0.0, divergence_score))

        # --- Component B: Persistence Duration (weight: 0.25) ---
        persistence_score = min(1.0, div_cycles * 0.15)
        if div_cycles == 0:
            persistence_score = 0.0

        # --- Component C: Hysteresis Decay Rate (weight: 0.20) ---
        # How fast is the cluster's decayed_score changing?
        # Use difference from raw to decayed as a proxy for decay velocity
        hyst_decay_score = abs(cluster_net_dir - decayed_score)
        hyst_decay_score = min(1.0, hyst_decay_score * 2.0)

        # If cluster net direction is weakening against the decayed score
        if cluster_net_dir < decayed_score and trade_is_buy == 1:
            # For BUY trades: cluster turning bearish → pressure up
            hyst_decay_score = min(1.0, hyst_decay_score * 1.5)
        elif cluster_net_dir > decayed_score and trade_is_buy == -1:
            # For SELL trades: cluster turning bullish → pressure up
            hyst_decay_score = min(1.0, hyst_decay_score * 1.5)

        # --- Component D: PnL Regime (weight: 0.15) ---
        current_pnl = trade.get("current_pnl", 0.0)
        peak_pnl = trade_state.get("peak_pnl", current_pnl)
        if current_pnl > peak_pnl:
            peak_pnl = current_pnl
        trade_state["peak_pnl"] = peak_pnl

        if peak_pnl > 0:
            # Drawdown from peak
            drawdown = peak_pnl - current_pnl
            pnl_regime_score = min(1.0, max(0.0, drawdown / max(peak_pnl, 1.0)))
        elif current_pnl < 0:
            # Already in loss: severity based on how deep
            # Use 3% of entry price as a normalisation
            entry_price = trade.get("entry_price", 1.0)
            pnl_regime_score = min(1.0, abs(current_pnl) / (0.03 * entry_price * 10000))
        else:
            pnl_regime_score = 0.0

        # --- Composite exit pressure score ---
        raw_score = (
            WEIGHT_DIVERGENCE * divergence_score
            + WEIGHT_PERSISTENCE * persistence_score
            + WEIGHT_HYSTERESIS_DECAY * hyst_decay_score
            + WEIGHT_PNL_REGIEM * pnl_regime_score
        )
        raw_score = min(1.0, max(0.0, raw_score))

        # --- Apply temporal decay/accumulation ---
        prev_pressure = trade_state["current_pressure"]

        if div_cycles > 0:
            # Accumulate
            new_pressure = self.temporal.accumulate(prev_pressure, div_cycles)
        else:
            # Decay
            new_pressure = self.temporal.decay(
                prev_pressure, trade_state["cycles_since_divergence"]
            )

        # Blend raw score with temporal pressure (60% temporal, 40% raw)
        blended = 0.6 * new_pressure + 0.4 * raw_score
        blended = round(min(1.0, max(0.0, blended)), 4)

        # --- Escalate ---
        new_state = RFEState.from_score(blended)
        prev_state = trade_state["current_state"]

        breach = self._check_threshold_breach(prev_state, new_state)

        # Update trade state tracking
        if new_state == prev_state:
            trade_state["cycles_in_state"] += 1
        else:
            trade_state["cycles_in_state"] = 1
            # Log transition
            self._transition_log.append({
                "symbol": symbol,
                "trade_id": trade_id,
                "from_state": prev_state,
                "to_state": new_state,
                "score": blended,
                "timestamp": datetime.now().isoformat(),
            })

        trade_state["previous_state"] = prev_state
        trade_state["current_state"] = new_state
        trade_state["current_pressure"] = blended
        trade_state["previous_pressure"] = prev_pressure
        trade_state["breach_threshold"] = breach

        # Determine if exit is allowed
        exit_allowed = RFEState.index(new_state) >= RFEState.index(RFEState.EXIT)

        components = {
            "divergence": round(divergence_score, 4),
            "persistence": round(persistence_score, 4),
            "hysteresis_decay": round(hyst_decay_score, 4),
            "pnl_regime": round(pnl_regime_score, 4),
        }

        return {
            "score": blended,
            "state": new_state,
            "components": components,
            "breach_threshold": breach,
            "cycles_in_state": trade_state["cycles_in_state"],
            "exit_allowed": exit_allowed,
            "divergence_cycles": div_cycles,
        }

    def escalate(self, trade_id: str, new_pressure: float) -> Dict[str, Any]:
        """Handle state transitions based on exit pressure score.

        Parameters
        ----------
        trade_id : str
            Unique trade identifier (e.g. "AUDUSD_BUY").
        new_pressure : float
            New computed exit pressure score.

        Returns
        -------
        dict
            Escalation result with state, transition info, and exit permission.
        """
        new_pressure = min(1.0, max(0.0, new_pressure))
        new_state = RFEState.from_score(new_pressure)

        trade_state = self._trade_states.setdefault(trade_id, {
            "current_pressure": 0.0,
            "previous_state": RFEState.INFO,
            "current_state": RFEState.INFO,
            "cycles_in_state": 0,
            "cycles_in_divergence": 0,
            "cycles_since_divergence": 0,
            "peak_pnl": 0.0,
            "previous_pressure": 0.0,
            "breach_threshold": False,
        })

        prev_state = trade_state["current_state"]
        prev_pressure = trade_state["current_pressure"]

        # Check if escalation is allowed by transition rules
        transition_allowed = RFEState.escalation_allowed(prev_state, new_state)
        exit_allowed = RFEState.index(new_state) >= RFEState.index(RFEState.EXIT)

        # Minimum cycle lock: can't change state if only 1 cycle in current
        can_transition = (
            trade_state["cycles_in_state"] >= MIN_CYCLES_IN_STATE
            or new_state == prev_state
            or RFEState.index(new_state) > RFEState.index(prev_state)  # always allow escalation
        )

        if not can_transition:
            # Keep current state
            final_state = prev_state
            trade_state["cycles_in_state"] += 1
        elif transition_allowed:
            final_state = new_state
            if new_state != prev_state:
                trade_state["cycles_in_state"] = 1
                trade_state["previous_state"] = prev_state
                self._transition_log.append({
                    "trade_id": trade_id,
                    "from_state": prev_state,
                    "to_state": new_state,
                    "score": new_pressure,
                    "timestamp": datetime.now().isoformat(),
                })
            else:
                trade_state["cycles_in_state"] += 1
        else:
            # De-escalation must go step by step
            prev_idx = RFEState.index(prev_state)
            new_idx = RFEState.index(new_state)
            if new_idx < prev_idx:
                # Must go through intermediate state
                intermediate_idx = prev_idx - 1
                final_state = RFEState.ORDER[intermediate_idx]
            else:
                final_state = new_state
            trade_state["cycles_in_state"] = 1
            if final_state != prev_state:
                trade_state["previous_state"] = prev_state
                self._transition_log.append({
                    "trade_id": trade_id,
                    "from_state": prev_state,
                    "to_state": final_state,
                    "score": new_pressure,
                    "timestamp": datetime.now().isoformat(),
                })

        trade_state["current_pressure"] = new_pressure
        trade_state["current_state"] = final_state

        return {
            "trade_id": trade_id,
            "previous_state": prev_state,
            "current_state": final_state,
            "previous_pressure": prev_pressure,
            "current_pressure": new_pressure,
            "exit_allowed": exit_allowed,
            "transition_allowed": transition_allowed,
        }

    def reset(self) -> None:
        """Clear all internal state (for testing or fresh start)."""
        self._trade_states.clear()
        self._pressure_history.clear()
        self._state_history.clear()
        self._transition_log.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_threshold_breach(self, prev_state: str, new_state: str) -> bool:
        """Return True if the score crossed a state boundary."""
        return RFEState.index(new_state) != RFEState.index(prev_state)

    def _format_transitions(self) -> Dict[str, List[str]]:
        """Build per-symbol transition summaries (last 10)."""
        transitions: Dict[str, List[str]] = {}
        for entry in self._transition_log:
            symbol = entry.get("symbol", entry.get("trade_id", "?"))
            transitions.setdefault(symbol, [])
            state_entry = (
                f"{entry['from_state']}[?] → {entry['to_state']}[?] (current)"
            )
            transitions[symbol].append(state_entry)

        # Simplify to match dashboard format: collect unique sequences
        result: Dict[str, List[str]] = {}
        for entry in self._transition_log:
            sym = entry.get("symbol", entry.get("trade_id", "?"))
            result.setdefault(sym, [])
            # Track cycles in each state
            seq = f"{entry['from_state']} → {entry['to_state']}"
            if not result[sym] or result[sym][-1] != seq:
                result[sym].append(seq)

        # Keep last 10 per symbol
        for sym in result:
            result[sym] = result[sym][-10:]

        return result

    def _format_temporal_info(self) -> Dict[str, Any]:
        """Build human-readable temporal decay info."""
        info: Dict[str, Any] = {
            "parameters": {
                "decay": self.temporal.decay_factor,
                "accumulate": self.temporal.accumulation_factor,
                "max_rise": self.temporal.max_single_cycle_increase,
            },
            "symbols": {},
        }

        for trade_id, state in self._trade_states.items():
            symbol = trade_id.rsplit("_", 1)[0] if "_" in trade_id else trade_id
            div_cycles = state["cycles_in_divergence"]
            res_cycles = state["cycles_since_divergence"]
            if div_cycles > 0:
                status = f"accumulating ({div_cycles} cycle{'s' if div_cycles != 1 else ''} in divergence)"
            elif res_cycles > 0:
                status = f"decaying ({res_cycles} cycle{'s' if res_cycles != 1 else ''} since divergence)"
            else:
                status = "stable (0 cycles in divergence)"
            info["symbols"][symbol] = status

        return info

    def _detect_threshold_breaches(self) -> List[Dict[str, Any]]:
        """Detect threshold boundary crossings this cycle."""
        breaches: List[Dict[str, Any]] = []
        for entry in reversed(self._transition_log[-50:]):
            if entry["from_state"] != entry["to_state"]:
                breaches.append({
                    "trade_id": entry.get("trade_id", "?"),
                    "symbol": entry.get("symbol", "?"),
                    "from": entry["from_state"],
                    "to": entry["to_state"],
                    "score": entry["score"],
                })
        # Deduplicate by (trade_id, from, to)
        seen = set()
        unique: List[Dict[str, Any]] = []
        for b in breaches:
            key = (b["trade_id"], b["from"], b["to"])
            if key not in seen:
                seen.add(key)
                unique.append(b)
        return unique[-10:]  # last 10 unique breaches


# ======================================================================
# Dashboard formatting
# ======================================================================


def format_arbitration_dashboard(result: Dict[str, Any]) -> str:
    """Render the full RFE Arbitration dashboard as a formatted string.

    Parameters
    ----------
    result : dict
        Output from ``RFEArbitrationLayer.evaluate()``.

    Returns
    -------
    str
        Formatted dashboard.
    """
    lines: list[str] = []
    lines.append("")
    lines.append("RFE ARBITRATION LAYER — EXIT AUTHORITY BRIDGE")
    lines.append("=" * 78)

    evaluations = result.get("evaluations", {})
    summary = result.get("summary", {})

    # Main table header
    header = (
        f"{'Trade':<14s} {'State':<12s} {'Pressure':<9s} "
        f"{'Cycles':<7s} {'Diverg':<7s} {'Persist':<8s} "
        f"{'HystDec':<8s} {'PnLReg':<7s} {'Exit?':<6s}"
    )
    lines.append(header)
    lines.append("-" * 78)

    for symbol in sorted(evaluations.keys()):
        e = evaluations[symbol]
        trade_label = symbol
        state = e["state"]
        score = e["score"]
        cycles = e.get("divergence_cycles", e.get("cycles_in_state", 0))
        comp = e.get("components", {})
        div = comp.get("divergence", 0.0)
        pers = comp.get("persistence", 0.0)
        hd = comp.get("hysteresis_decay", 0.0)
        pnl = comp.get("pnl_regime", 0.0)
        exit_flag = "YES" if e.get("exit_allowed", False) else "NO"

        lines.append(
            f"{trade_label:<14s} {state:<12s} {score:<9.2f} "
            f"{cycles:<7d} {div:<7.2f} {pers:<8.2f} "
            f"{hd:<8.2f} {pnl:<7.2f} {exit_flag:<6s}"
        )

    lines.append("")

    # Overall risk
    max_pressure = summary.get("max_pressure", 0.0)
    max_state = summary.get("max_state", "INFO")
    overall_risk = summary.get("overall_risk", "LOW")
    trades_at_risk = summary.get("trades_at_risk", [])
    any_exit = summary.get("any_exit_allowed", False)

    lines.append(f"OVERALL RISK: {overall_risk} (max pressure {max_pressure:.2f})")
    if trades_at_risk:
        risk_details = ", ".join(
            f"{s} ({result['evaluations'].get(s, {}).get('state', '?')})"
            for s in trades_at_risk
        )
        lines.append(f"Trades at risk: {risk_details}")
    else:
        lines.append("Trades at risk: none")
    lines.append(f"Exit allowed: {'YES - ' + ', '.join(t for t in trades_at_risk if result['evaluations'].get(t, {}).get('exit_allowed', False)) if any_exit else 'NONE'}")

    lines.append("")

    # State transitions
    transitions = result.get("transitions", {})
    lines.append("RFE STATE TRANSITIONS (last 10 cycles):")
    if transitions:
        for symbol in sorted(transitions.keys()):
            seq = " → ".join(transitions[symbol])
            lines.append(f"  {symbol}: {seq}")
    else:
        lines.append("  No transitions recorded yet.")

    lines.append("")

    # Temporal decay info
    temporal = result.get("temporal", {})
    lines.append("TEMPORAL DECAY:")
    params = temporal.get("parameters", {})
    lines.append(
        f"  Parameters: decay={params.get('decay', 0.15):.2f}/cycle, "
        f"accumulate={params.get('accumulate', 0.10):.2f}/cycle, "
        f"max_rise={params.get('max_rise', 0.15):.2f}/cycle"
    )
    symbols_temporal = temporal.get("symbols", {})
    for symbol in sorted(symbols_temporal.keys()):
        lines.append(f"  {symbol}: {symbols_temporal[symbol]}")

    lines.append("")

    # Threshold breaches
    breaches = result.get("breaches", [])
    lines.append("THRESHOLD BREACHES:")
    if breaches:
        for b in breaches:
            tid = b.get("symbol", b.get("trade_id", "?"))
            lines.append(
                f"  {tid}: {b['from']} → {b['to']} "
                f"(score={b['score']:.2f})"
            )
    else:
        lines.append("  None (no state boundaries crossed this cycle)")

    lines.append("")
    lines.append("=" * 78)

    return "\n".join(lines)
