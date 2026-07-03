"""Execution Governor Layer — Final Safety Boundary.

Sits between RFE Arbitration and MT5 execution. Converts probabilistic exit
pressure into action permission by applying three filters:

1. **Temporal Persistence** — state must persist N consecutive cycles before
   action is taken (prevents "flicker exits").
2. **Price-Context Weighting** — amplify exit pressure near local maxima,
   suppress during recovery.
3. **Reversal Filter** — if pressure drops after a spike, cancel the exit
   signal (prevents CHFJPY-style false signals).

Dependencies
------------
- ``rfe_arbitration.py`` for RFE state constants and thresholds.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .rfe_arbitration import RFEState

logger = logging.getLogger("proxima_ops.risk.execution_governor")

# ---------------------------------------------------------------------------
# Governor State Constants
# ---------------------------------------------------------------------------


class GovernorState:
    """Governor action-permission states — more conservative than RFE states.

    Thresholds differ from RFE — the governor requires higher pressure to
    reach equivalent action states, adding an extra safety margin.
    """

    HOLD = "HOLD"
    PREPARE = "PREPARE"
    CONDITIONAL_EXIT = "CONDITIONAL_EXIT"
    EXIT = "EXIT"

    ORDER = [HOLD, PREPARE, CONDITIONAL_EXIT, EXIT]

    PRESSURE_THRESHOLDS: Dict[str, Tuple[float, float]] = {
        HOLD: (0.0, 0.35),
        PREPARE: (0.35, 0.60),
        CONDITIONAL_EXIT: (0.60, 0.85),
        EXIT: (0.85, 1.0),
    }

    @classmethod
    def from_pressure(cls, pressure: float) -> str:
        """Map a pressure score to governor state (more conservative than RFE)."""
        for state in cls.ORDER:
            lo, hi = cls.PRESSURE_THRESHOLDS[state]
            if lo <= pressure < hi:
                return state
        return cls.EXIT

    @classmethod
    def index(cls, state: str) -> int:
        """Return ordinal index of a governor state."""
        try:
            return cls.ORDER.index(state)
        except ValueError:
            return 0

    @classmethod
    def risk_label(cls, state: str) -> str:
        """Classify governor state into a human-readable risk label."""
        idx = cls.index(state)
        if idx >= cls.index(cls.EXIT):
            return "CRITICAL"
        if idx >= cls.index(cls.CONDITIONAL_EXIT):
            return "HIGH"
        if idx >= cls.index(cls.PREPARE):
            return "MEDIUM"
        return "LOW"


# ---------------------------------------------------------------------------
# Temporal Persistence Gate
# ---------------------------------------------------------------------------


class TemporalPersistenceGate:
    """RFE state must persist N consecutive cycles before action is taken.

    This prevents "flicker exits" — the exact failure the Brain warned about:
    *"EXIT signal was correct early but system would have been wrong at least
    once before final correctness."*

    Parameters
    ----------
    min_persist_cycles : dict, optional
        Minimum consecutive cycles required per RFE state.
        Default: WARNING=2, EXIT_PREP=2, EXIT=1
    reset_on_downgrade : bool, default=True
        If True, counter resets when RFE state downgrades.
    """

    def __init__(
        self,
        min_persist_cycles: Optional[Dict[str, int]] = None,
        reset_on_downgrade: bool = True,
    ) -> None:
        self.min_persist_cycles = min_persist_cycles or {
            "WARNING": 2,
            "EXIT_PREP": 2,
            "EXIT": 1,
        }
        self.reset_on_downgrade = reset_on_downgrade

        # Per-trade state: trade_id -> {current_rfe_state, cycles_confirmed, ...}
        self._state: Dict[str, Dict[str, Any]] = {}

    def check(self, trade_id: str, current_rfe_state: str, current_pressure: float) -> Dict[str, Any]:
        """Check if the persistence requirement is satisfied.

        Returns
        -------
        dict
            - persisted (bool): True if requirement met
            - cycles_confirmed (int): consecutive cycles in current RFE state
            - cycles_required (int): what is needed
            - cycles_remaining (int): 0 if satisfied
            - gate_open (bool): alias for persisted
        """
        ts = self._state.setdefault(
            trade_id,
            {
                "current_rfe_state": "<INIT>",
                "previous_rfe_state": None,
                "cycles_confirmed": 0,
            },
        )

        prev_state = ts["current_rfe_state"]

        if current_rfe_state != prev_state:
            # State changed
            prev_idx = RFEState.index(prev_state)
            curr_idx = RFEState.index(current_rfe_state)

            if curr_idx < prev_idx and self.reset_on_downgrade:
                # Downgrade with reset enabled -> reset counter
                ts["cycles_confirmed"] = 0
            elif curr_idx > prev_idx:
                # Upgrade (any state skip) -> reset counter for new state
                ts["cycles_confirmed"] = 0
            # else: downgrade with reset_on_downgrade=False -> keep counter

            ts["previous_rfe_state"] = prev_state
            ts["current_rfe_state"] = current_rfe_state
        else:
            # Same state: increment counter (capped at a large number)
            ts["cycles_confirmed"] += 1

        cycles_confirmed = ts["cycles_confirmed"]
        cycles_required = self.min_persist_cycles.get(current_rfe_state, 1)
        cycles_remaining = max(0, cycles_required - cycles_confirmed)
        gate_open = cycles_confirmed >= cycles_required

        return {
            "persisted": gate_open,
            "cycles_confirmed": cycles_confirmed,
            "cycles_required": cycles_required,
            "cycles_remaining": cycles_remaining,
            "gate_open": gate_open,
        }

    def reset(self) -> None:
        """Clear all internal state."""
        self._state.clear()


# ---------------------------------------------------------------------------
# Price-Context Weighting
# ---------------------------------------------------------------------------


class PriceContextWeighting:
    """Amplifies or suppresses exit pressure based on price position.

    Key insight from CHFJPY data:
    - Near local maxima -> exit pressure should count MORE
    - During recovery phase -> pressure should be DISCOUNTED

    Parameters
    ----------
    lookback_period : int, default=20
        Bars to look back when finding local max/min.
    max_amplification : float, default=1.5
        Maximum multiplier when near a peak.
    recovery_discount : float, default=0.3
        Discount factor during recovery phase.
    """

    def __init__(
        self,
        lookback_period: int = 20,
        max_amplification: float = 1.5,
        recovery_discount: float = 0.3,
    ) -> None:
        self.lookback_period = lookback_period
        self.max_amplification = max_amplification
        self.recovery_discount = recovery_discount

    def evaluate(
        self,
        trade: Dict[str, Any],
        price_history: Dict[str, List[float]],
        pressure_history: List[float],
    ) -> Dict[str, Any]:
        """Full evaluation: returns weight and context metadata.

        Returns
        -------
        dict
            - weight (float): multiplier [0.3, 1.5]
            - proximity_to_peak (float): 0.0 (trough) to 1.0 (peak)
            - is_recovering (bool): price recovering from recent low
            - pressure_dropping (bool): exit pressure decreasing
            - amplified (bool): weight > 1.0
            - suppressed (bool): weight < 1.0
            - neutral (bool): weight == 1.0
        """
        symbol = trade.get("symbol", "UNKNOWN")
        current_price = trade.get("current_price", 0.0)
        current_pressure = pressure_history[-1] if pressure_history else 0.0

        # Determine pressure trend
        pressure_dropping = False
        if len(pressure_history) >= 2:
            pressure_dropping = pressure_history[-1] < pressure_history[-2] - 0.01
        elif len(pressure_history) == 1:
            pressure_dropping = pressure_history[0] < 0.35

        prices = price_history.get(symbol, [])
        recent_prices = prices[-self.lookback_period:] if prices else []

        # Type guard: current_price may be a dict (e.g., {"bid": 1.234, "ask": 1.236})
        if isinstance(current_price, dict):
            current_price = current_price.get("close", 0.0)

        if not recent_prices or not isinstance(current_price, (int, float)) or current_price <= 0.0:
            return {
                "weight": 1.0,
                "proximity_to_peak": 0.5,
                "is_recovering": False,
                "pressure_dropping": pressure_dropping,
                "amplified": False,
                "suppressed": False,
                "neutral": True,
            }

        # Type guard: ensure all recent_prices entries are numeric (not dicts)
        recent_prices = [
            p if isinstance(p, (int, float))
            else p.get("close", 0.0) if isinstance(p, dict)
            else 0.0
            for p in recent_prices
        ]

        highest = max(recent_prices)
        lowest = min(recent_prices)
        price_range = highest - lowest

        if price_range <= 0.0:
            return {
                "weight": 1.0,
                "proximity_to_peak": 0.5,
                "is_recovering": False,
                "pressure_dropping": pressure_dropping,
                "amplified": False,
                "suppressed": False,
                "neutral": True,
            }

        # Proximity to peak: 1.0 = at peak, 0.0 = at trough
        proximity_to_peak = (current_price - lowest) / price_range

        # Detect recovery: price was near low and is moving up
        is_recovering = self._detect_recovery(recent_prices, current_price)

        # Determine weight based on context
        if proximity_to_peak >= 0.7 and current_pressure > 0.6:
            # Near peak AND pressure elevated -> amplify
            amp_range = self.max_amplification - 1.0
            t = (proximity_to_peak - 0.7) / 0.3  # 0.0 at 0.7, 1.0 at 1.0
            weight = 1.0 + t * amp_range
            weight = round(min(self.max_amplification, weight), 4)
            return {
                "weight": weight,
                "proximity_to_peak": round(proximity_to_peak, 4),
                "is_recovering": is_recovering,
                "pressure_dropping": pressure_dropping,
                "amplified": weight > 1.0,
                "suppressed": False,
                "neutral": False,
            }

        if is_recovering and pressure_dropping:
            # Recovering AND pressure dropping -> strong suppression
            weight = round(self.recovery_discount, 4)
            return {
                "weight": weight,
                "proximity_to_peak": round(proximity_to_peak, 4),
                "is_recovering": is_recovering,
                "pressure_dropping": pressure_dropping,
                "amplified": False,
                "suppressed": True,
                "neutral": False,
            }

        if is_recovering and not pressure_dropping:
            # Recovering AND pressure rising -> neutral (don't interfere)
            return {
                "weight": 1.0,
                "proximity_to_peak": round(proximity_to_peak, 4),
                "is_recovering": is_recovering,
                "pressure_dropping": pressure_dropping,
                "amplified": False,
                "suppressed": False,
                "neutral": True,
            }

        if proximity_to_peak <= 0.3:
            # Near trough -> slight suppression (0.7 at trough, 1.0 at 0.3)
            t = proximity_to_peak / 0.3
            weight = 0.7 + t * 0.3
            weight = round(min(1.0, weight), 4)
            return {
                "weight": weight,
                "proximity_to_peak": round(proximity_to_peak, 4),
                "is_recovering": is_recovering,
                "pressure_dropping": pressure_dropping,
                "amplified": False,
                "suppressed": weight < 1.0,
                "neutral": weight == 1.0,
            }

        return {
            "weight": 1.0,
            "proximity_to_peak": round(proximity_to_peak, 4),
            "is_recovering": is_recovering,
            "pressure_dropping": pressure_dropping,
            "amplified": False,
            "suppressed": False,
            "neutral": True,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_recovery(self, prices: List[float], current_price: float) -> bool:
        """Detect if price is in a recovery phase (up from recent low)."""
        if len(prices) < 3:
            return False

        recent_low = min(prices[-3:])
        # Current price above recent low AND last two bars trending up
        return current_price > recent_low and prices[-1] > prices[-2] > recent_low

    def reset(self) -> None:
        """No persistent state to clear (all inputs are parameterised)."""
        pass


# ---------------------------------------------------------------------------
# Reversal Filter
# ---------------------------------------------------------------------------


class ReversalFilter:
    """Cancels exit signal if pressure decreases after a spike.

    Prevents the exact CHFJPY pattern:
    - Pressure spikes (e.g., 0.72), then drops (e.g., 0.38)
    - Without this filter, the system would have entered CONDITIONAL_EXIT
      during the spike, but by the time execution happened the signal reversed.

    State machine per trade::

        NORMAL -> SPIKE_DETECTED -> REVERSAL -> LOCKED -> NORMAL
                                 -> SUSTAINED -> (may detect later)

    Parameters
    ----------
    spike_window : int, default=3
        Number of recent cycles to compare for spike detection.
    drop_threshold : float, default=0.15
        Absolute pressure drop required to confirm reversal.
    recovery_window : int, default=3
        Lockout cycles after reversal before returning to NORMAL.
    """

    PHASE_NORMAL = "NORMAL"
    PHASE_SPIKE_DETECTED = "SPIKE_DETECTED"
    PHASE_REVERSAL = "REVERSAL"
    PHASE_SUSTAINED = "SUSTAINED"
    PHASE_LOCKED = "LOCKED"

    def __init__(
        self,
        spike_window: int = 3,
        drop_threshold: float = 0.15,
        recovery_window: int = 3,
    ) -> None:
        self.spike_window = spike_window
        self.drop_threshold = drop_threshold
        self.recovery_window = recovery_window

        # Per-trade state machine
        self._states: Dict[str, Dict[str, Any]] = {}

    def evaluate(self, trade_id: str, pressure_history: List[float]) -> Dict[str, Any]:
        """Evaluate whether a reversal has been detected.

        Returns
        -------
        dict
            - reversal_detected (bool): True if a reversal was just confirmed
            - signal_cancelled (bool): True if exit signal should be blocked
            - lockout_cycles_remaining (int): cycles left in lockout
            - phase (str): current state machine phase
        """
        ts = self._states.setdefault(
            trade_id,
            {
                "phase": self.PHASE_NORMAL,
                "peak_pressure": 0.0,
                "lockout_cycles": 0,
                "spike_cycle": 0,
            },
        )

        if not pressure_history:
            return {
                "reversal_detected": False,
                "signal_cancelled": False,
                "lockout_cycles_remaining": 0,
                "phase": ts["phase"],
            }

        current_pressure = pressure_history[-1]

        # Update peak pressure
        if current_pressure > ts["peak_pressure"]:
            ts["peak_pressure"] = current_pressure
            ts["spike_cycle"] = len(pressure_history) - 1

        # Countdown lockout
        if ts["lockout_cycles"] > 0:
            ts["lockout_cycles"] -= 1

        # --- State machine transitions ---
        phase = ts["phase"]

        if phase == self.PHASE_LOCKED:
            if ts["lockout_cycles"] <= 0:
                ts["phase"] = self.PHASE_NORMAL
                ts["peak_pressure"] = current_pressure

        elif phase == self.PHASE_REVERSAL:
            # One cycle in REVERSAL, then enter LOCKED
            ts["phase"] = self.PHASE_LOCKED
            ts["lockout_cycles"] = self.recovery_window

        elif phase == self.PHASE_SPIKE_DETECTED:
            drop = ts["peak_pressure"] - current_pressure
            if drop >= self.drop_threshold:
                ts["phase"] = self.PHASE_REVERSAL
            else:
                ts["phase"] = self.PHASE_SUSTAINED

        elif phase == self.PHASE_SUSTAINED:
            drop = ts["peak_pressure"] - current_pressure
            if drop >= self.drop_threshold:
                ts["phase"] = self.PHASE_REVERSAL

        elif phase == self.PHASE_NORMAL:
            # Check for new spike
            if len(pressure_history) >= self.spike_window:
                recent = pressure_history[-self.spike_window:]
                avg_recent = sum(recent) / len(recent)
                latest = recent[-1]
                # Spike: latest > 0.6 pressure AND significantly above average
                if latest > 0.6 and latest > avg_recent * 1.15:
                    ts["phase"] = self.PHASE_SPIKE_DETECTED
                    ts["peak_pressure"] = latest

        # --- Build result ---
        phase = ts["phase"]
        lockout_remaining = max(0, ts["lockout_cycles"])
        is_cancelled = phase in (self.PHASE_REVERSAL, self.PHASE_LOCKED)
        is_reversal = phase == self.PHASE_REVERSAL

        return {
            "reversal_detected": is_reversal,
            "signal_cancelled": is_cancelled,
            "lockout_cycles_remaining": lockout_remaining,
            "phase": phase,
        }

    def reset(self) -> None:
        """Clear all internal state."""
        self._states.clear()


# ---------------------------------------------------------------------------
# Execution Governor
# ---------------------------------------------------------------------------


class ExecutionGovernor:
    """Final safety boundary before any execution action.

    Converts probabilistic exit pressure into permission states:

    - **HOLD** — no action (score < 0.35 or transient)
    - **PREPARE** — monitoring, no action yet (score 0.35-0.60, persisting)
    - **CONDITIONAL_EXIT** — exit allowed only with price-context confirmation
    - **EXIT** — unconditional exit (score 0.85+ with full confirmation)

    Applies three sequential gates:

    1. **Temporal Persistence** — RFE state must persist N consecutive cycles.
    2. **Price-Context Weighting** — amplify/suppress based on price position.
    3. **Reversal Filter** — block exit if pressure dropped after a spike.

    Parameters
    ----------
    persistence_gate : TemporalPersistenceGate, optional
    price_weighting : PriceContextWeighting, optional
    reversal_filter : ReversalFilter, optional
    """

    def __init__(
        self,
        persistence_gate: Optional[TemporalPersistenceGate] = None,
        price_weighting: Optional[PriceContextWeighting] = None,
        reversal_filter: Optional[ReversalFilter] = None,
    ) -> None:
        self.persistence_gate = persistence_gate or TemporalPersistenceGate()
        self.price_weighting = price_weighting or PriceContextWeighting()
        self.reversal_filter = reversal_filter or ReversalFilter()

        # History tracking for dashboard and persistence
        self._governor_state_history: Dict[str, List[str]] = defaultdict(list)
        self._pressure_history: Dict[str, List[float]] = defaultdict(list)
        self._governor_cycles: Dict[str, int] = defaultdict(int)
        self._last_governor_state: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply_regime_params(self, params: dict) -> None:
        """Formal interface for regime-based parameter updates.

        Replaces cross-layer attribute mutation via GovernorParameterMapper.
        Only known keys are accepted; unknown keys are silently ignored.

        Accepted keys:
            temporal_persistence_window
            reversal_spike_window
            reversal_drop_threshold
            price_lookback_period
            max_amplification
            recovery_discount
        """
        pg = self.persistence_gate
        if pg and "temporal_persistence_window" in params:
            win = params["temporal_persistence_window"]
            for state in pg.min_persist_cycles:
                pg.min_persist_cycles[state] = max(1, win // 2)

        rf = self.reversal_filter
        if rf:
            if "reversal_spike_window" in params:
                rf.spike_window = params["reversal_spike_window"]
            if "reversal_drop_threshold" in params:
                rf.drop_threshold = params["reversal_drop_threshold"]

        pw = self.price_weighting
        if pw:
            if "price_lookback_period" in params:
                pw.lookback_period = params["price_lookback_period"]
            if "max_amplification" in params:
                pw.max_amplification = params["max_amplification"]
            if "recovery_discount" in params:
                pw.recovery_discount = params["recovery_discount"]

    def evaluate(
        self,
        rfe_output: Dict[str, Any],
        price_history: Optional[Dict[str, List[float]]] = None,
    ) -> Dict[str, Any]:
        """Evaluate RFE output through all 3 gates.

        Parameters
        ----------
        rfe_output : dict
            Output from ``RFEArbitrationLayer.evaluate()``.
            Must contain ``evaluations`` dict mapping symbol -> eval dict.
        price_history : dict, optional
            Mapping of symbol -> list of historical prices.
            If None, price-context weighting returns neutral (1.0).

        Returns
        -------
        dict
            - decisions (dict): per-symbol governor decisions
            - summary (dict): aggregate safety summary
            - timestamp (str): ISO timestamp
        """
        evaluations = rfe_output.get("evaluations", {})
        price_history = price_history or {}

        decisions: Dict[str, Dict[str, Any]] = {}
        all_states: List[str] = []
        trades_pending_exit: List[str] = []

        for symbol in sorted(evaluations.keys()):
            ev = evaluations[symbol]
            trade_id = symbol
            rfe_state = ev.get("state", RFEState.INFO)
            rfe_pressure = ev.get("score", 0.0)

            # Track pressure history
            self._pressure_history[trade_id].append(rfe_pressure)

            # --- Gate 1: Governor State (derived from RFE pressure) ---
            governor_state = GovernorState.from_pressure(rfe_pressure)

            # Track cycles in governor state
            if self._last_governor_state.get(trade_id) != governor_state:
                self._governor_cycles[trade_id] = 0
            self._governor_cycles[trade_id] += 1
            self._last_governor_state[trade_id] = governor_state

            self._governor_state_history[trade_id].append(governor_state)

            # --- Gate 2: Temporal Persistence ---
            persistence = self.persistence_gate.check(trade_id, rfe_state, rfe_pressure)

            # --- Gate 3: Price-Context Weighting ---
            # Extract current_price from flattened rfe context if available
            current_price = 0.0
            for ctx_key in ("current_price", "entry_price", "price"):
                if ctx_key in ev:
                    current_price = ev[ctx_key]
                    break
            # Type guard: current_price may be a dict (e.g., {"bid": 1.234, "ask": 1.236})
            if isinstance(current_price, dict):
                current_price = current_price.get("close", 0.0)
            trade = {
                "symbol": symbol,
                "current_price": current_price,
            }
            price_ctx = self.price_weighting.evaluate(
                trade, price_history, self._pressure_history[trade_id]
            )
            price_weight = price_ctx["weight"]
            weighted_pressure = round(rfe_pressure * price_weight, 4)

            # --- Gate 4: Reversal Filter ---
            reversal = self.reversal_filter.evaluate(trade_id, self._pressure_history[trade_id])

            # --- Resolve action ---
            action = self._resolve_action(
                governor_state, persistence, reversal, price_ctx
            )

            decision: Dict[str, Any] = {
                "governor_state": governor_state,
                "governor_cycles": self._governor_cycles[trade_id],
                "rfe_state": rfe_state,
                "rfe_pressure": rfe_pressure,
                "price_weight": price_weight,
                "weighted_pressure": weighted_pressure,
                "persistence": persistence,
                "reversal": reversal,
                "price_context": price_ctx,
                "action": action,
            }
            decisions[symbol] = decision
            all_states.append(governor_state)

            if action["type"] in ("CLOSE", "CLOSE_PARTIAL"):
                trades_pending_exit.append(symbol)

        # --- Build summary ---
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

        summary = {
            "any_exit_allowed": max_index >= GovernorState.index(GovernorState.CONDITIONAL_EXIT),
            "max_governor_state": max_state,
            "trades_pending_exit": trades_pending_exit,
            "system_safety": system_safety,
        }

        return {
            "decisions": decisions,
            "summary": summary,
            "timestamp": datetime.now().isoformat(),
        }

    def resolve_actions(self, decisions: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert governor decisions into actionable command list.

        Returns
        -------
        list of dict
            Each dict: symbol, action (CLOSE/CLOSE_PARTIAL), fraction, reason,
            governor_state.
        """
        actions: List[Dict[str, Any]] = []
        for symbol in sorted(decisions.keys()):
            d = decisions[symbol]
            action = d.get("action", {})
            if action.get("type") in ("CLOSE", "CLOSE_PARTIAL"):
                actions.append(
                    {
                        "symbol": symbol,
                        "action": action["type"],
                        "fraction": action.get("fraction", 0.0),
                        "reason": action.get("reason", ""),
                        "governor_state": d.get("governor_state", ""),
                    }
                )
        return actions

    def reset(self) -> None:
        """Clear all internal state (for testing or fresh start)."""
        self.persistence_gate.reset()
        self.reversal_filter.reset()
        self._governor_state_history.clear()
        self._pressure_history.clear()
        self._governor_cycles.clear()
        self._last_governor_state.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_action(
        self,
        governor_state: str,
        persistence: Dict[str, Any],
        reversal: Dict[str, Any],
        price_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Convert governor decision into an actionable command.

        Rules
        -----
        - HOLD -> no action
        - PREPARE -> no action, only logging
        - CONDITIONAL_EXIT -> CLOSE_PARTIAL (50 %) if persistence OPEN
          AND no reversal detected
        - EXIT -> CLOSE (100 %) if persistence OPEN AND no reversal detected
        """
        gate_open = persistence.get("gate_open", False)
        signal_cancelled = reversal.get("signal_cancelled", False)
        reversal_detected = reversal.get("reversal_detected", False)
        cycles_confirmed = persistence.get("cycles_confirmed", 0)
        cycles_remaining = persistence.get("cycles_remaining", 0)

        if governor_state == GovernorState.HOLD:
            return {
                "type": "NONE",
                "fraction": 0.0,
                "reason": "HOLD: pressure below 0.35 threshold",
            }

        if governor_state == GovernorState.PREPARE:
            return {
                "type": "NONE",
                "fraction": 0.0,
                "reason": f"PREPARE: monitoring, pressure in 0.35-0.60 range",
            }

        if governor_state == GovernorState.CONDITIONAL_EXIT:
            if not gate_open:
                return {
                    "type": "NONE",
                    "fraction": 0.0,
                    "reason": (
                        f"CONDITIONAL_EXIT: persistence gate not yet open "
                        f"({cycles_confirmed}/{cycles_confirmed + cycles_remaining} cycles)"
                    ),
                }
            if signal_cancelled or reversal_detected:
                return {
                    "type": "NONE",
                    "fraction": 0.0,
                    "reason": "CONDITIONAL_EXIT: signal cancelled by reversal filter",
                }

            parts = ["CONDITIONAL_EXIT"]
            if persistence.get("persisted"):
                parts.append(f"persisted {cycles_confirmed} cycles")
            if price_context.get("amplified"):
                parts.append("price near peak (amplified)")
            if price_context.get("suppressed"):
                parts.append("price suppressed")

            return {
                "type": "CLOSE_PARTIAL",
                "fraction": 0.5,
                "reason": ": ".join(parts),
            }

        if governor_state == GovernorState.EXIT:
            if not gate_open:
                return {
                    "type": "NONE",
                    "fraction": 0.0,
                    "reason": (
                        f"EXIT: persistence gate not yet open "
                        f"({cycles_confirmed}/{cycles_confirmed + cycles_remaining} cycles)"
                    ),
                }
            if signal_cancelled or reversal_detected:
                return {
                    "type": "NONE",
                    "fraction": 0.0,
                    "reason": "EXIT: signal cancelled by reversal filter",
                }

            return {
                "type": "CLOSE",
                "fraction": 1.0,
                "reason": f"EXIT: all gates open, persisted {cycles_confirmed} cycles",
            }

        # Fallback (shouldn't happen)
        return {
            "type": "NONE",
            "fraction": 0.0,
            "reason": f"Unknown governor state: {governor_state}",
        }


# ---------------------------------------------------------------------------
# Dashboard Formatting
# ---------------------------------------------------------------------------


def format_governor_dashboard(result: Dict[str, Any]) -> str:
    """Render the Execution Governor dashboard as a formatted string.

    Parameters
    ----------
    result : dict
        Output from ``ExecutionGovernor.evaluate()``.

    Returns
    -------
    str
        Formatted dashboard.
    """
    lines: List[str] = []
    lines.append("")
    lines.append("EXECUTION GOVERNOR LAYER — SAFETY BOUNDARY")
    lines.append("=" * 78)

    decisions = result.get("decisions", {})
    summary = result.get("summary", {})

    if not decisions:
        lines.append("  (no trades evaluated)")
        lines.append("")
        lines.append("=" * 78)
        return "\n".join(lines)

    # Header
    header = (
        f"{'Trade':<16s} {'GState':<20s} {'RFE-State':<12s} "
        f"{'Pressure':<9s} {'WtPress':<9s} {'Exit?':<6s} Reason"
    )
    lines.append(header)
    lines.append("-" * 78)

    for symbol in sorted(decisions.keys()):
        d = decisions[symbol]
        gs = d["governor_state"]
        rfe_state = d["rfe_state"]
        pressure = d["rfe_pressure"]
        wt_pressure = d["weighted_pressure"]
        action = d.get("action", {})
        exit_flag = "YES" if action.get("type") in ("CLOSE", "CLOSE_PARTIAL") else "NO"
        reason = action.get("reason", "")
        # Truncate long reasons
        if len(reason) > 36:
            reason = reason[:33] + "..."

        lines.append(
            f"{symbol:<16s} {gs:<20s} {rfe_state:<12s} "
            f"{pressure:<9.2f} {wt_pressure:<9.2f} {exit_flag:<6s} {reason}"
        )

    lines.append("")

    # Summary
    safety = summary.get("system_safety", "SAFE")
    pending = summary.get("trades_pending_exit", [])
    lines.append(f"SYSTEM SAFETY: {safety}")
    if pending:
        lines.append(f"Trades pending exit: {', '.join(pending)}")
    else:
        lines.append("Trades pending exit: NONE")

    lines.append("")

    # Gate status
    lines.append("GATE STATUS:")
    pers_parts = []
    ctx_parts = []
    rev_parts = []
    for symbol in sorted(decisions.keys()):
        d = decisions[symbol]
        pers = d.get("persistence", {})
        gate_s = "OPEN" if pers.get("gate_open") else "CLOSED"
        pers_parts.append(f"{symbol}={gate_s}({pers.get('cycles_confirmed', 0)}cyc)")

        ctx = d.get("price_context", {})
        w = ctx.get("weight", 1.0)
        label = "(neutral)"
        if ctx.get("amplified"):
            label = "(amplified)"
        elif ctx.get("suppressed"):
            label = "(suppressed)"
        ctx_parts.append(f"{symbol}={w:.1f}x{label}")

        rev = d.get("reversal", {})
        rev_parts.append(f"{symbol}={rev.get('phase', 'NORMAL')}")

    lines.append(f"  Persistence Gate: {', '.join(pers_parts)}")
    lines.append(f"  Price Context: {', '.join(ctx_parts)}")
    lines.append(f"  Reversal Filter: {', '.join(rev_parts)}")

    lines.append("")

    # Governor state transitions
    lines.append("GOVERNOR STATE TRANSITIONS:")
    for symbol in sorted(decisions.keys()):
        d = decisions[symbol]
        gs = d["governor_state"]
        gc = d.get("governor_cycles", 0)
        # Determine if stable or transitioning
        rfe_state = d["rfe_state"]
        status = "stable" if d["action"]["type"] == "NONE" else "pending action"
        lines.append(f"  {symbol}: {gs}[{gc}] ({status})")

    lines.append("")
    lines.append("=" * 78)

    return "\n".join(lines)


# ======================================================================
# CHFJPY Replay Utility
# ======================================================================


def run_chfjpy_replay(governor: Optional[ExecutionGovernor] = None) -> Dict[str, Any]:
    """Run the canonical CHFJPY replay through the Execution Governor.

    Returns a dict with per-cycle governor decisions for analysis.
    """
    from .rfe_arbitration import RFEArbitrationLayer, format_arbitration_dashboard

    if governor is None:
        governor = ExecutionGovernor()

    rfe = RFEArbitrationLayer(
        decay_factor=0.15,
        accumulation_factor=0.08,
        max_single_cycle_increase=0.12,
    )
    rfe.reset()
    governor.reset()

    # CHFJPY lifecycle (same as TC10 in test_rfe_arbitration.py)
    lifecycle = [
        # (cycle, net_dir, divergence, pnl, price)
        (0, 0.05, 0.10, 0.0, 200.500),
        (1, -0.10, 0.20, 1.5, 200.480),
        (2, -0.30, 0.35, -2.0, 200.420),
        (3, -0.45, 0.50, -5.0, 200.380),
        (4, -0.50, 0.55, -8.0, 200.360),
        (5, -0.55, 0.60, -12.0, 200.340),
        (6, 0.20, 0.20, -8.0, 200.370),  # recovery
        (7, -0.35, 0.40, -9.0, 200.350),
        (8, -0.65, 0.70, -18.0, 200.310),
    ]

    from .signal_manifold import symbol_to_primary_cluster

    price_history: Dict[str, List[float]] = {"CHFJPY": []}
    results = []

    for cycle, net_dir, divergence, pnl, price in lifecycle:
        price_history["CHFJPY"].append(price)

        trade_dict = {
            "symbol": "CHFJPY",
            "direction": "BUY",
            "current_pnl": pnl,
            "entry_price": 200.500,
            "sl": 200.348,
            "tp": 201.200,
            "current_price": price,
        }

        cluster = symbol_to_primary_cluster("CHFJPY")

        cluster_states = {
            cluster: {
                "net_direction": net_dir,
                "net_pressure": "BEARISH" if net_dir < -0.15 else ("BULLISH" if net_dir > 0.15 else "NEUTRAL"),
                "coherence": 1.0 - abs(divergence),
                "divergence": divergence,
            }
        }

        hyst_state = {
            "clusters": {
                cluster: {
                    "current_state": "CONTRACTING" if net_dir < -0.15 else ("EXPANDING" if net_dir > 0.15 else "NEUTRAL"),
                    "decayed_score": net_dir * 0.85,
                    "raw_net_dir": net_dir,
                    "raw_state": "NEUTRAL",
                    "previous_state": "NEUTRAL",
                    "cycles_in_state": 1,
                    "locked": False,
                    "entry_threshold": 0.65,
                    "exit_threshold": 0.45,
                }
            },
            "flip_events": [],
            "locked_clusters": [],
            "hysteresis_active": True,
            "memory_decay": 0.7,
            "hysteresis_band": 0.15,
            "min_lock_cycles": 3,
        }

        # Run RFE arbitration
        rfe_result = rfe.compute_exit_pressure(trade_dict, cluster_states, hyst_state)

        # Run governor
        rfe_output = {
            "evaluations": {"CHFJPY": rfe_result},
            "summary": {
                "max_pressure": rfe_result["score"],
                "max_state": rfe_result["state"],
                "any_exit_allowed": rfe_result["exit_allowed"],
                "trades_at_risk": ["CHFJPY"] if rfe_result["state"] != "INFO" else [],
                "overall_risk": "LOW",
            },
            "transitions": {},
            "temporal": {},
            "breaches": [],
            "timestamp": "",
        }

        gov_result = governor.evaluate(rfe_output, price_history)
        decision = gov_result["decisions"]["CHFJPY"]

        results.append(
            {
                "cycle": cycle,
                "price": price,
                "pnl": pnl,
                "rfe_pressure": rfe_result["score"],
                "rfe_state": rfe_result["state"],
                "rfe_exit_allowed": rfe_result["exit_allowed"],
                "governor_state": decision["governor_state"],
                "price_weight": decision["price_weight"],
                "weighted_pressure": decision["weighted_pressure"],
                "persistence_gate": decision["persistence"]["gate_open"],
                "persistence_cycles": decision["persistence"]["cycles_confirmed"],
                "reversal_phase": decision["reversal"]["phase"],
                "reversal_cancelled": decision["reversal"]["signal_cancelled"],
                "action_type": decision["action"]["type"],
                "action_reason": decision["action"]["reason"],
            }
        )

    return {"results": results, "price_history": price_history["CHFJPY"]}


def format_chfjpy_comparison(replay_result: Dict[str, Any]) -> str:
    """Format CHFJPY replay as a with-vs-without comparison table."""
    results = replay_result.get("results", [])
    lines: List[str] = []
    lines.append("")
    lines.append("CHFJPY REPLAY: WITH GOVERNOR vs WITHOUT GOVERNOR")
    lines.append("=" * 78)
    lines.append("")
    lines.append(
        f"{'Cycle':<6s} {'Price':<9s} {'PnL':<8s} {'RFE_State':<12s} "
        f"{'RFE_Exit?':<10s} {'Gov_State':<20s} {'Gov_Exit?':<10s} {'Gate/Reason'}"
    )
    lines.append("-" * 78)

    for r in results:
        rfe_exit = "YES" if r["rfe_exit_allowed"] else "NO"
        gov_exit = "YES" if r["action_type"] in ("CLOSE", "CLOSE_PARTIAL") else "NO"

        # Show gate info
        gate_info = ""
        if r["governor_state"] == "HOLD":
            gate_info = "hold"
        elif r["governor_state"] == "PREPARE":
            gate_info = "preparing"
        elif r["reversal_cancelled"]:
            gate_info = f"REVERSAL({r['reversal_phase']})"
        elif not r["persistence_gate"]:
            gate_info = f"PERSIST({r['persistence_cycles']}cyc)"
        else:
            gate_info = r["action_reason"][:30]

        if len(gate_info) > 30:
            gate_info = gate_info[:27] + "..."

        lines.append(
            f"{r['cycle']:<6d} {r['price']:<9.3f} {r['pnl']:<8.1f} {r['rfe_state']:<12s} "
            f"{rfe_exit:<10s} {r['governor_state']:<20s} {gov_exit:<10s} {gate_info}"
        )

    lines.append("")
    lines.append("Key observations:")
    lines.append("  - Governor prevents false exit during early spikes")
    lines.append("  - Reversal filter catches pressure drops after spikes")
    lines.append("  - Allowed exit only when pressure is sustained")
    lines.append("")
    lines.append("=" * 78)

    return "\n".join(lines)
