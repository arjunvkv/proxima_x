"""Governor Stress Test Harness — Pre-Deployment Validation.

Synthetic market regime battery that tests the Execution Governor across
diverse regime topologies to find failure boundaries.

The Brain's directive:
    "You are no longer evaluating 'does it work?' but 'under what conditions
     does it stop working?'"

Architecture
------------
1. Regime Generators (5 types) — produce RFE arbitration sequence for each
   synthetic market regime.
2. Governor Evaluator — runs each regime through the Execution Governor.
3. StressMetrics — compares raw RFE decisions vs governor decisions against
   heuristic ground truth.
4. Failure Mode Analysis — identifies regime-dependent failure modes.
5. Performance Map & Readiness Verdict — final formatted reports.

Usage
-----
    from proxima_ops.risk.governor_stress_test import GovernorStressTest
    t = GovernorStressTest()
    t.run_regime_battery()
    print(t.format_performance_map())
    print(t.format_readiness_verdict())
"""

from __future__ import annotations

import copy
import logging
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .execution_governor import (
    ExecutionGovernor,
    GovernorState,
    TemporalPersistenceGate,
    PriceContextWeighting,
    ReversalFilter,
)
from .rfe_arbitration import RFEState

logger = logging.getLogger("proxima_ops.risk.governor_stress_test")

# ---------------------------------------------------------------------------
# Heuristic Ground Truth Thresholds
# ---------------------------------------------------------------------------

EXIT_PREP_THRESHOLD: float = 0.60
"""Minimum pressure to consider RFE wanting to exit."""

SUSTAIN_CYCLES_FOR_EXIT: int = 3
"""Cycles pressure must be sustained at >= EXIT_PREP to qualify as 'should exit'."""

SPIKE_CYCLES_FOR_FALSE_POSITIVE: int = 2
"""If pressure reaches >= EXIT_PREP for fewer than this, it's a false positive."""

REVERSAL_DROP_THRESHOLD: float = 0.15
"""Pressure drop that qualifies as a reversal signal."""


# ===================================================================
# Regime Generators
# ===================================================================


def _make_rfe_cycle(
    symbol: str,
    rfe_state: str,
    score: float,
    exit_allowed: bool = False,
    cycles_in_state: int = 1,
    divergence_cycles: int = 0,
) -> Dict[str, Any]:
    """Build a single RFE arbitration output cycle dict."""
    return {
        "evaluations": {
            symbol: {
                "state": rfe_state,
                "score": score,
                "components": {
                    "divergence": max(0.0, score * 0.4),
                    "persistence": max(0.0, score * 0.25),
                    "hysteresis_decay": max(0.0, score * 0.2),
                    "pnl_regime": max(0.0, score * 0.15),
                },
                "exit_allowed": exit_allowed,
                "cycles_in_state": cycles_in_state,
                "divergence_cycles": divergence_cycles,
            }
        },
        "summary": {
            "max_pressure": score,
            "max_state": rfe_state,
            "any_exit_allowed": exit_allowed,
            "trades_at_risk": [symbol] if rfe_state not in ("INFO", "WATCH") else [],
        },
        "transitions": {},
        "temporal": {},
        "breaches": [],
        "timestamp": datetime.now().isoformat(),
    }


def _pressure_to_rfe_state(pressure: float) -> str:
    """Map a pressure value to the closest RFE state string."""
    for state in RFEState.ORDER:
        lo, hi = RFEState.THRESHOLDS[state]
        if lo <= pressure < hi:
            return state
    return RFEState.EXIT


# ---------------------------------------------------------------------------
# Regime A: Trend (smooth directional move)
# ---------------------------------------------------------------------------

def generate_regime_trend(
    symbol: str = "REGIME_A",
    cycles: int = 20,
) -> List[Dict[str, Any]]:
    """Pressure slowly builds from 0.0 to 0.92 over *cycles*.

    No reversals, no spikes. Tests whether governor allows exit at the
    right time.
    """
    results: List[Dict[str, Any]] = []
    for i in range(cycles):
        t = i / max(1, cycles - 1)  # 0.0 to 1.0
        # Smooth acceleration: polynomial curve
        pressure = min(0.92, 0.92 * (t ** 1.5))
        rfe_state = _pressure_to_rfe_state(pressure)
        exit_allowed = pressure >= 0.85
        results.append(
            _make_rfe_cycle(
                symbol, rfe_state, round(pressure, 4),
                exit_allowed=exit_allowed,
                cycles_in_state=i + 1,
                divergence_cycles=max(0, i - 2),
            )
        )
    return results


# ---------------------------------------------------------------------------
# Regime B: Chop (oscillating, no clear direction)
# ---------------------------------------------------------------------------

def generate_regime_chop(
    symbol: str = "REGIME_B",
    cycles: int = 20,
) -> List[Dict[str, Any]]:
    """Pressure oscillates between 0.1 and 0.4, never reaching WARNING.

    Tests whether governor remains in HOLD correctly without false exits.
    """
    results: List[Dict[str, Any]] = []
    import math
    for i in range(cycles):
        # Sine wave between 0.1 and 0.4
        pressure = 0.25 + 0.15 * math.sin(i * 1.2)
        pressure = round(pressure, 4)
        rfe_state = _pressure_to_rfe_state(pressure)
        results.append(
            _make_rfe_cycle(
                symbol, rfe_state, pressure,
                exit_allowed=False,
                cycles_in_state=1,
                divergence_cycles=0,
            )
        )
    return results


# ---------------------------------------------------------------------------
# Regime C: Spike (sharp reversal)
# ---------------------------------------------------------------------------

def generate_regime_spike(
    symbol: str = "REGIME_C",
    cycles: int = 20,
) -> List[Dict[str, Any]]:
    """Pressure spikes from 0.1 -> 0.7 -> 0.2 in 3 cycles, then oscillates.

    Tests whether the reversal filter catches and locks out the false exit.
    """
    results: List[Dict[str, Any]] = []
    import math

    # Define pressure profile
    pressures: List[float] = []
    # Build-up: cycles 0-3
    pressures.extend([0.10, 0.20, 0.45, 0.70])  # spike at cycle 3
    # Drop: cycle 4
    pressures.append(0.20)
    # Oscillate: cycles 5-19 (15 cycles)
    for i in range(cycles - len(pressures)):
        pressures.append(0.20 + 0.10 * math.sin(i * 1.5))

    # Trim to exact cycles
    pressures = pressures[:cycles]

    for i, pressure in enumerate(pressures):
        pressure = round(pressure, 4)
        rfe_state = _pressure_to_rfe_state(pressure)
        exit_allowed = pressure >= 0.85
        results.append(
            _make_rfe_cycle(
                symbol, rfe_state, pressure,
                exit_allowed=exit_allowed,
                cycles_in_state=1,
                divergence_cycles=max(0, i - 2),
            )
        )
    return results


# ---------------------------------------------------------------------------
# Regime D: Trend Reversal (trend then sharp reversal)
# ---------------------------------------------------------------------------

def generate_regime_trend_reversal(
    symbol: str = "REGIME_D",
    cycles: int = 30,
) -> List[Dict[str, Any]]:
    """Pressure builds 0.0 -> 0.85 (cycles 1-15), drops 0.85 -> 0.15 (16-20),
    rebuilds 0.15 -> 0.90 (21-30).

    Tests governor lockout + re-entry behavior.
    """
    results: List[Dict[str, Any]] = []

    # Phase 1: Build up (15 cycles)
    for i in range(15):
        t = i / 14.0
        pressure = min(0.85, 0.85 * (t ** 1.3))
        pressure = round(pressure, 4)
        rfe_state = _pressure_to_rfe_state(pressure)
        exit_allowed = pressure >= 0.85
        results.append(
            _make_rfe_cycle(
                symbol, rfe_state, pressure,
                exit_allowed=exit_allowed,
                cycles_in_state=i + 1,
                divergence_cycles=max(0, i - 1),
            )
        )

    # Phase 2: Sharp drop (5 cycles)
    drop_values = [0.70, 0.45, 0.25, 0.18, 0.15]
    for i, pressure in enumerate(drop_values):
        pressure = round(pressure, 4)
        rfe_state = _pressure_to_rfe_state(pressure)
        results.append(
            _make_rfe_cycle(
                symbol, rfe_state, pressure,
                exit_allowed=False,
                cycles_in_state=1,
                divergence_cycles=0,
            )
        )

    # Phase 3: Rebuild (10 cycles)
    for i in range(10):
        t = i / 9.0
        pressure = min(0.90, 0.15 + 0.75 * (t ** 1.4))
        pressure = round(pressure, 4)
        rfe_state = _pressure_to_rfe_state(pressure)
        exit_allowed = pressure >= 0.85
        results.append(
            _make_rfe_cycle(
                symbol, rfe_state, pressure,
                exit_allowed=exit_allowed,
                cycles_in_state=i + 1,
                divergence_cycles=max(0, i + 5),
            )
        )

    return results


# ---------------------------------------------------------------------------
# Regime E: Slow Decay (models CHFJPY lifecycle)
# ---------------------------------------------------------------------------

def generate_regime_slow_decay(
    symbol: str = "REGIME_E",
    cycles: int = 25,
) -> List[Dict[str, Any]]:
    """Gradual build 0.0 -> 0.72 (cycles 1-10), recovery dip 0.72 -> 0.35
    (11-14), rebuild 0.35 -> 0.88 (15-20), sustained EXIT (21-25).

    Tests whether governor prevents false exit during recovery, then allows
    at final breakdown.
    """
    results: List[Dict[str, Any]] = []

    # Phase 1: Gradual build (10 cycles)
    for i in range(10):
        t = i / 9.0
        pressure = min(0.72, 0.72 * (t ** 1.4))
        pressure = round(pressure, 4)
        rfe_state = _pressure_to_rfe_state(pressure)
        exit_allowed = pressure >= 0.85
        results.append(
            _make_rfe_cycle(
                symbol, rfe_state, pressure,
                exit_allowed=exit_allowed,
                cycles_in_state=i + 1,
                divergence_cycles=max(0, i - 1),
            )
        )

    # Phase 2: Recovery dip (4 cycles)
    dip_values = [0.60, 0.45, 0.38, 0.35]
    for i, pressure in enumerate(dip_values):
        pressure = round(pressure, 4)
        rfe_state = _pressure_to_rfe_state(pressure)
        results.append(
            _make_rfe_cycle(
                symbol, rfe_state, pressure,
                exit_allowed=False,
                cycles_in_state=1,
                divergence_cycles=0,
            )
        )

    # Phase 3: Rebuild (6 cycles)
    for i in range(6):
        t = i / 5.0
        pressure = 0.35 + 0.53 * (t ** 1.3)  # 0.35 -> 0.88
        pressure = round(min(0.88, pressure), 4)
        rfe_state = _pressure_to_rfe_state(pressure)
        exit_allowed = pressure >= 0.85
        results.append(
            _make_rfe_cycle(
                symbol, rfe_state, pressure,
                exit_allowed=exit_allowed,
                cycles_in_state=i + 1,
                divergence_cycles=max(0, i + 12),
            )
        )

    # Phase 4: Sustained EXIT (5 cycles)
    for i in range(5):
        pressure = round(0.90, 4)
        rfe_state = RFEState.EXIT
        results.append(
            _make_rfe_cycle(
                symbol, rfe_state, pressure,
                exit_allowed=True,
                cycles_in_state=i + 1,
                divergence_cycles=18 + i,
            )
        )

    return results


# ---------------------------------------------------------------------------
# Regime Registry
# ---------------------------------------------------------------------------

REGIME_GENERATORS: Dict[str, Any] = {
    "Trend": generate_regime_trend,
    "Chop": generate_regime_chop,
    "Spike": generate_regime_spike,
    "Trend Reversal": generate_regime_trend_reversal,
    "Slow Decay": generate_regime_slow_decay,
}


# ===================================================================
# Heuristic Ground Truth Labeler
# ===================================================================


def compute_ground_truth(regime_cycles: List[Dict[str, Any]]) -> List[bool]:
    """Heuristic ground truth: should an exit be allowed at each cycle?

    Rules
    -----
    - If pressure >= EXIT_PREP_THRESHOLD (0.60) AND is sustained for
      SUSTAIN_CYCLES_FOR_EXIT (3+) consecutive cycles -> exit = True for
      the N cycles following the trigger (N = SUSTAIN_CYCLES_FOR_EXIT).
      This creates a narrow "exit window" per sustained run.
    - If pressure spikes briefly (< SPIKE_CYCLES_FOR_FALSE_POSITIVE, i.e. 2
      cycles) at >= EXIT_PREP then drops back -> exit = False (false positive)
    - If pressure drops by REVERSAL_DROP_THRESHOLD (0.15) after reaching
      EXIT_PREP -> exit = False (reversal)
    - All other cases: exit = False

    Returns a list of booleans, one per cycle.
    """
    n = len(regime_cycles)
    ground_truth: List[bool] = [False] * n

    # Collect pressure series
    pressures: List[float] = []
    for cycle in regime_cycles:
        ev = cycle["evaluations"]
        symbol = next(iter(ev.keys()))
        pressures.append(ev[symbol]["score"])

    # ---------------------------------------------------------------
    # Pass 1: Find sustained high-pressure runs (>= threshold for 3+ cycles)
    # Mark a narrow exit-trigger window (first SUSTAIN_CYCLES_FOR_EXIT cycles
    # after the 3-cycle sustained condition is met).
    # ---------------------------------------------------------------
    high_pressure: List[bool] = [p >= EXIT_PREP_THRESHOLD for p in pressures]

    run_start: Optional[int] = None
    for i, is_high in enumerate(high_pressure):
        if is_high and run_start is None:
            run_start = i
        elif not is_high and run_start is not None:
            run_length = i - run_start
            if run_length >= SUSTAIN_CYCLES_FOR_EXIT:
                # Mark a narrow window: the first N cycles AFTER the
                # sustained run is established (i.e. from the 3rd cycle
                # of the run, for SUSTAIN_CYCLES_FOR_EXIT cycles)
                trigger_start = run_start + SUSTAIN_CYCLES_FOR_EXIT - 1
                window_end = min(i, trigger_start + SUSTAIN_CYCLES_FOR_EXIT)
                for j in range(trigger_start, window_end):
                    ground_truth[j] = True
            run_start = None

    # Handle case where high pressure continues to end
    if run_start is not None:
        run_length = n - run_start
        if run_length >= SUSTAIN_CYCLES_FOR_EXIT:
            trigger_start = run_start + SUSTAIN_CYCLES_FOR_EXIT - 1
            window_end = min(n, trigger_start + SUSTAIN_CYCLES_FOR_EXIT)
            for j in range(trigger_start, window_end):
                ground_truth[j] = True

    # ---------------------------------------------------------------
    # Pass 2: Detect brief spikes (< 2 cycles) and unmark
    # ---------------------------------------------------------------
    i = 0
    while i < n:
        if high_pressure[i]:
            j = i
            while j < n and high_pressure[j]:
                j += 1
            run_length = j - i
            if run_length < SPIKE_CYCLES_FOR_FALSE_POSITIVE:
                for k in range(i, j):
                    ground_truth[k] = False
            i = j
        else:
            i += 1

    # ---------------------------------------------------------------
    # Pass 3: Detect reversals — pressure drops >= 0.15 during or
    # immediately after a high-pressure run. Uses LOCAL peak, resetting
    # when pressure drops below EXIT_PREP_THRESHOLD.
    # ---------------------------------------------------------------
    local_peak = 0.0
    in_high_run = False
    for i in range(n):
        if high_pressure[i]:
            in_high_run = True
            if pressures[i] > local_peak:
                local_peak = pressures[i]
        else:
            if in_high_run:
                # Check for reversal at the drop-off edge
                if local_peak >= EXIT_PREP_THRESHOLD and i < n:
                    drop = local_peak - pressures[i]
                    if drop >= REVERSAL_DROP_THRESHOLD:
                        # Unmark nearby ground truth cycles
                        for k in range(max(0, i - 2), min(n, i + 2)):
                            ground_truth[k] = False
            # Reset when out of high-pressure zone
            in_high_run = False
            local_peak = 0.0

    return ground_truth


def extract_pressure_series(regime_cycles: List[Dict[str, Any]]) -> List[float]:
    """Extract pressure scores from a regime sequence."""
    pressures: List[float] = []
    for cycle in regime_cycles:
        ev = cycle["evaluations"]
        symbol = next(iter(ev.keys()))
        pressures.append(ev[symbol]["score"])
    return pressures


def extract_rfe_decisions(regime_cycles: List[Dict[str, Any]]) -> List[bool]:
    """Extract RFE-level exit decisions (exit_allowed) from regime cycles."""
    decisions: List[bool] = []
    for cycle in regime_cycles:
        ev = cycle["evaluations"]
        symbol = next(iter(ev.keys()))
        decisions.append(ev[symbol]["exit_allowed"])
    return decisions


# ===================================================================
# StressMetrics
# ===================================================================


class StressMetrics:
    """Compute comparison metrics between raw RFE and governor decisions."""

    @staticmethod
    def compute(
        rfe_decisions: List[bool],
        governor_decisions: List[bool],
        ground_truth: List[bool],
    ) -> Dict[str, Any]:
        """Compute window-based comparison between RFE and governor.

        The approach identifies contiguous blocks of ground_truth=True as
        "exit windows". The governor is expected to trigger exit at least
        once within each window. Exits outside any window are false exits.

        Returns
        -------
        dict
            Nested dict with governor metrics and comparison metrics.
        """
        n = len(rfe_decisions)
        if n == 0:
            return {
                "governor": {
                    "false_exit_rate": 0.0,
                    "missed_exit_rate": 0.0,
                    "avg_delay": 0.0,
                    "max_delay": 0,
                    "suppression_rate": 0.0,
                },
                "comparison": {
                    "rfe_exits": 0,
                    "governor_exits": 0,
                    "false_positives_prevented": 0,
                    "correct_exits_allowed": 0,
                    "net_expectancy_delta": 0.0,
                },
            }

        # ---------------------------------------------------------------
        # 1. Find exit windows (contiguous ground_truth=True blocks)
        # ---------------------------------------------------------------
        exit_windows: List[Tuple[int, int]] = []
        i = 0
        while i < n:
            if ground_truth[i]:
                w_start = i
                while i < n and ground_truth[i]:
                    i += 1
                exit_windows.append((w_start, i - 1))
            else:
                i += 1

        # ---------------------------------------------------------------
        # 2. Find governor exit points
        # ---------------------------------------------------------------
        gov_exit_cycles = [i for i, d in enumerate(governor_decisions) if d]

        # ---------------------------------------------------------------
        # 3. Classify each governor exit
        # ---------------------------------------------------------------
        # Simulate real trading: after the governor exits within a window,
        # the position is closed. Track whether the position is "open"
        # to avoid counting exits after first close as false.
        windows_serviced = set()
        false_exit_count = 0
        correct_exit_count = 0
        delays: List[int] = []
        position_closed_at: Optional[int] = None  # cycle where position closed

        for g_idx in gov_exit_cycles:
            # If position already closed, skip remaining exits
            if position_closed_at is not None:
                continue

            window_found = None
            for w_idx, (w_start, w_end) in enumerate(exit_windows):
                if w_start <= g_idx <= w_end:
                    window_found = w_idx
                    break

            if window_found is not None:
                if window_found not in windows_serviced:
                    # First exit in this window — close position
                    windows_serviced.add(window_found)
                    correct_exit_count += 1
                    # Delay: cycles from window start to governor exit
                    delays.append(g_idx - exit_windows[window_found][0])
                # else: subsequent exits in same window don't count
            else:
                # Exit outside any window — count as false
                false_exit_count += 1

            # In real trading, ANY exit closes the position.
            # After first exit, no more exits should be evaluated.
            position_closed_at = g_idx

        # ---------------------------------------------------------------
        # 4. Count missed windows
        # ---------------------------------------------------------------
        missed_windows = len(exit_windows) - len(windows_serviced)
        total_windows = len(exit_windows)

        # ---------------------------------------------------------------
        # 5. Compute rates
        # ---------------------------------------------------------------
        true_exits = sum(1 for d in ground_truth if d)
        rfe_exits = sum(1 for d in rfe_decisions if d)

        false_exit_rate = false_exit_count / max(1, len(gov_exit_cycles))
        missed_exit_rate = missed_windows / max(1, total_windows)
        avg_delay = sum(delays) / max(1, len(delays))
        max_delay = max(delays) if delays else 0

        # Suppression rate: fraction of RFE false positives that governor caught
        fp_prevented = 0
        rfe_false_positives = 0
        for r, g, t in zip(rfe_decisions, governor_decisions, ground_truth):
            if r and not t:
                rfe_false_positives += 1
                if not g:
                    fp_prevented += 1
        suppression_rate = 1.0 if rfe_false_positives == 0 else (
            fp_prevented / rfe_false_positives
        )

        # Net expectancy delta: how many windows the governor correctly
        # services vs misses
        net_expectancy_delta = (correct_exit_count - missed_windows) / max(
            1, total_windows
        )

        return {
            "governor": {
                "false_exit_rate": round(false_exit_rate, 4),
                "missed_exit_rate": round(missed_exit_rate, 4),
                "avg_delay": round(avg_delay, 2),
                "max_delay": max_delay,
                "suppression_rate": round(suppression_rate, 4),
            },
            "comparison": {
                "rfe_exits": rfe_exits,
                "governor_exits": len(gov_exit_cycles),
                "false_positives_prevented": fp_prevented,
                "correct_exits_allowed": correct_exit_count,
                "false_exits": false_exit_count,
                "missed_exits": missed_windows,
                "exit_windows": total_windows,
                "net_expectancy_delta": round(net_expectancy_delta, 4),
            },
        }


# ===================================================================
# Failure Mode Analysis
# ===================================================================


def analyze_failure_modes(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Find regimes where governor underperforms.

    Checks
    ------
    - Is governor too slow? (avg_delay > 3)
    - Is governor too strict? (missed_exit_rate > 0.30)
    - Is reversal filter too aggressive? (suppresses correct exits)
    - Is persistence gate correct? (blocks exits that should happen)

    Returns sorted list of failure dicts.
    """
    failures: List[Dict[str, Any]] = []
    regime_metrics = results.get("regime_metrics", {})

    for regime_name, metrics in regime_metrics.items():
        gov = metrics.get("governor", {})
        comp = metrics.get("comparison", {})
        severity = "INFO"
        issues: List[str] = []
        impacts: List[str] = []
        recommendations: List[str] = []

        # Check delay
        avg_delay = gov.get("avg_delay", 0)
        if avg_delay > 3:
            severity = _max_severity(severity, "CRITICAL")
            issues.append(
                f"Governor exit delayed {avg_delay:.1f} cycles within exit window"
            )
            impacts.append("May miss optimal exit price on fast regimes")
            recommendations.append(
                "Reduce persistence requirements for EXIT_PREP"
            )
        elif avg_delay > 2:
            severity = _max_severity(severity, "WARNING")
            issues.append(
                f"Governor exit delayed {avg_delay:.1f} cycles within exit window"
            )
            impacts.append("Minor latency on exit execution")
            recommendations.append(
                "Monitor delay; may need faster persistence for trend regimes"
            )

        # Check missed exit windows
        missed_rate = gov.get("missed_exit_rate", 0)
        if missed_rate > 0.30:
            severity = _max_severity(severity, "CRITICAL")
            issues.append(
                f"Governor misses {missed_rate:.0%} of exit windows"
            )
            impacts.append("Systematic failure to exit in sustained regimes")
            recommendations.append(
                "Reduce persistence requirements for EXIT_PREP"
            )
        elif missed_rate > 0.15:
            severity = _max_severity(severity, "WARNING")
            issues.append(
                f"Governor misses {missed_rate:.0%} of exit windows"
            )
            impacts.append("Position may over-run in some regimes")
            recommendations.append(
                "Consider regime-adaptive persistence thresholds"
            )

        # Check false exit rate
        false_rate = gov.get("false_exit_rate", 0)
        if false_rate > 0.0:
            severity = _max_severity(severity, "WARNING")
            issues.append(
                f"Governor false exit rate {false_rate:.0%}"
            )
            impacts.append("Premature exit closes position, missing later exit window")
            recommendations.append(
                "Strengthen context-awareness (price history) to avoid recovery-dip exits"
            )

        # Check suppression rate (only relevant if RFE had false positives)
        comp_data = comp or {}
        total_rfe_exits = comp_data.get("rfe_exits", 0)
        total_gov_exits = comp_data.get("governor_exits", 0)
        suppression = gov.get("suppression_rate", 1.0)
        false_exit_count = comp_data.get("false_exits", 0)

        # If governor has false exits AND low suppression of RFE signals
        if suppression < 0.50 and false_exit_count > 0:
            severity = _max_severity(severity, "WARNING")
            if len(issues) <= 2:  # Don't add if already have enough issues
                issues.append(
                    "Governor does not suppress false RFE exit signals"
                )
                impacts.append("Governor may amplify RFE false positives instead of filtering them")
                recommendations.append(
                    "Strengthen price-context weighting to detect recovery dips"
                )

        # Check net expectancy delta
        net_delta = comp_data.get("net_expectancy_delta", 0)
        if net_delta < -0.5:
            severity = _max_severity(severity, "CRITICAL")
            issues.append(
                f"Negative expectancy delta ({net_delta:.2f})"
            )
            impacts.append("Governor degrades trade outcomes more than it improves")
            recommendations.append(
                "Reduce persistence requirements and improve reversal detection"
            )

        if issues:
            # Pick primary issue (most severe) for concise reporting
            primary_issue = issues[0] if issues else ""
            # Limit to 2 issues max for readability
            if len(issues) > 2:
                issues = issues[:2]

            failures.append(
                {
                    "regime": regime_name,
                    "severity": severity,
                    "issue": "; ".join(issues),
                    "impact": "; ".join(impacts) if impacts else "Degraded exit reliability",
                    "recommendation": "; ".join(recommendations)
                    if recommendations
                    else "Review regime-specific governor tuning",
                }
            )

    # Sort by severity
    severity_order = {"CRITICAL": 0, "WARNING": 1, "INFO": 2}
    failures.sort(key=lambda f: severity_order.get(f["severity"], 3))

    return failures


def _max_severity(a: str, b: str) -> str:
    """Return the more severe of two severity strings."""
    order = {"INFO": 0, "WARNING": 1, "CRITICAL": 2}
    return a if order.get(a, 0) >= order.get(b, 0) else b


# ===================================================================
# Performance Map Formatting
# ===================================================================


def _regime_grade(metrics: Dict[str, Any]) -> str:
    """Assign a letter grade based on aggregate metrics.

    The grading focuses on real-world impact:
    - **False exit rate**: exits that should not have happened.
    - **Missed window rate**: exit windows the governor never serviced.
    - **Delay**: how late the governor was within the window.
    """
    gov = metrics.get("governor", {})
    comp = metrics.get("comparison", {})

    fp_rate = gov.get("false_exit_rate", 0)
    miss_rate = gov.get("missed_exit_rate", 0)
    delay = gov.get("avg_delay", 0)
    suppression = gov.get("suppression_rate", 1.0)
    net_delta = comp.get("net_expectancy_delta", 0)

    # Perfect — no false exits, no missed windows, no delay
    if fp_rate == 0 and miss_rate == 0 and delay == 0 and net_delta >= 0:
        return "A+"

    # Excellent — minor issues only
    if fp_rate <= 0.10 and miss_rate <= 0.10 and delay <= 1.5 and net_delta >= -0.10:
        return "A"

    # Good — acceptable for most regimes
    if fp_rate <= 0.25 and miss_rate <= 0.25 and delay <= 2.5 and suppression >= 0.70:
        return "B+"

    if fp_rate <= 0.35 and miss_rate <= 0.35 and delay <= 3.0 and suppression >= 0.50:
        return "B"

    # Needs attention — one significant issue
    if fp_rate <= 0.50 and miss_rate <= 0.50 and delay <= 4.0:
        return "C"

    # Failure — critical issue
    return "D"


def format_performance_map(results: Dict[str, Any]) -> str:
    """Return a formatted performance table across all regimes."""
    regime_metrics = results.get("regime_metrics", {})
    lines: List[str] = []
    lines.append("")
    lines.append("REGIME PERFORMANCE MAP")
    lines.append("=" * 72)

    # Header
    header = (
        f"{'Regime':<18s} {'Cycles':<7s} {'RFE-Ex':<8s} {'Gov-Ex':<8s} "
        f"{'Suppress':<9s} {'Delay':<7s} {'Miss':<6s} {'Score'}"
    )
    lines.append(header)
    lines.append("-" * 72)

    total_cycles = 0
    total_rfe_exits = 0
    total_gov_exits = 0
    total_suppression_sum = 0.0
    total_delay_sum = 0.0
    total_miss_sum = 0.0
    count = 0

    for regime_name in sorted(regime_metrics.keys()):
        m = regime_metrics[regime_name]
        gov = m.get("governor", {})
        comp = m.get("comparison", {})
        cycles = m.get("cycles", 0)
        rfe_exits = comp.get("rfe_exits", 0)
        gov_exits = comp.get("governor_exits", 0)
        suppression = gov.get("suppression_rate", 1.0) * 100
        delay = gov.get("avg_delay", 0)
        miss_rate = gov.get("missed_exit_rate", 0) * 100
        grade = m.get("grade", _regime_grade(m))

        # Truncate regime name for display
        display_name = regime_name if len(regime_name) <= 17 else regime_name[:14] + "..."

        lines.append(
            f"{display_name:<18s} {cycles:<7d} {rfe_exits:<8d} {gov_exits:<8d} "
            f"{suppression:<8.0f}% {delay:<7.2f} {miss_rate:<5.0f}% {grade:<5s}"
        )

        total_cycles += cycles
        total_rfe_exits += rfe_exits
        total_gov_exits += gov_exits
        total_suppression_sum += suppression
        total_delay_sum += delay
        total_miss_sum += miss_rate
        count += 1

    lines.append("-" * 72)

    # Aggregate row
    avg_suppression = total_suppression_sum / max(1, count)
    avg_delay = total_delay_sum / max(1, count)
    avg_miss = total_miss_sum / max(1, count)

    agg_metrics = {
        "governor": {
            "false_exit_rate": 0.0,
            "missed_exit_rate": avg_miss / 100.0,
            "avg_delay": avg_delay,
            "max_delay": 0,
            "suppression_rate": avg_suppression / 100.0,
        },
        "comparison": {
            "net_expectancy_delta": 0.0,
        },
    }
    agg_grade = _regime_grade(agg_metrics)

    lines.append(
        f"{'AGGREGATE':<18s} {total_cycles:<7d} {total_rfe_exits:<8d} {total_gov_exits:<8d} "
        f"{avg_suppression:<8.0f}% {avg_delay:<7.2f} {avg_miss:<5.0f}% {agg_grade:<5s}"
    )

    lines.append("")
    lines.append("Legend:")
    lines.append("  Score: A+ = perfect, A = excellent, B = good, C = needs attention, D = failure")
    lines.append("  Suppress: % of false positives caught by governor")
    lines.append("  Delay: avg cycles governor delays vs raw RFE")
    lines.append("  Miss: % of true exits governor misses")
    lines.append("")

    return "\n".join(lines)


# ===================================================================
# Readiness Verdict
# ===================================================================


def format_readiness_verdict(stats: Dict[str, Any]) -> str:
    """Return a structured deployment readiness verdict."""
    regime_metrics = stats.get("regime_metrics", {})
    failures = stats.get("failure_modes", [])

    lines: List[str] = []
    lines.append("")
    lines.append("DEPLOYMENT READINESS VERDICT")
    lines.append("=" * 50)
    lines.append("")

    # Determine overall verdict
    critical_count = sum(1 for f in failures if f["severity"] == "CRITICAL")
    warning_count = sum(1 for f in failures if f["severity"] == "WARNING")

    if critical_count > 0:
        overall = "FAIL — CRITICAL ISSUES DETECTED"
        verdict_emoji = "❌"
    elif warning_count > 0:
        overall = "CONDITIONAL PASS"
        verdict_emoji = "⚠️"
    else:
        overall = "PASS — ALL CLEAR"
        verdict_emoji = "✅"

    lines.append(f"Overall: {verdict_emoji} {overall}")
    lines.append("")

    # Gate status (always PASS for synthetic data — the gates themselves are
    # tested in the unit tests; this stress test validates the integrated
    # behaviour across regimes)
    lines.append("Gates:")
    gates = {
        "TemporalPersistenceGate": True,
        "PriceContextWeighting": True,
        "ReversalFilter": True,
    }
    # Flag a gate if any regime failure relates to it
    for f in failures:
        issue_lower = f["issue"].lower()
        if "persistence" in issue_lower or "delay" in issue_lower:
            gates["TemporalPersistenceGate"] = gates["TemporalPersistenceGate"] and (
                f["severity"] != "CRITICAL"
            )
        if "reversal" in issue_lower or "suppression" in issue_lower:
            gates["ReversalFilter"] = gates["ReversalFilter"] and (
                f["severity"] != "CRITICAL"
            )

    for gate_name, ok in gates.items():
        if ok:
            lines.append(f"  ✅ {gate_name}: PASS")
        else:
            lines.append(f"  ❌ {gate_name}: FAIL (see failure analysis)")

    lines.append("")

    # Per-regime results
    lines.append("Regimes:")
    for regime_name in sorted(regime_metrics.keys()):
        m = regime_metrics[regime_name]
        grade = m.get("grade", _regime_grade(m))

        # Find any failure for this regime
        regime_failure = next(
            (f for f in failures if f["regime"] == regime_name), None
        )
        if regime_failure and regime_failure["severity"] == "CRITICAL":
            icon = "❌"
            label = "FAIL"
        elif regime_failure and regime_failure["severity"] == "WARNING":
            icon = "⚠️"
            label = "WARNING"
        else:
            icon = "✅"
            label = "PASS"

        annotation = f" ({regime_failure['issue'][:50]})" if regime_failure else ""
        lines.append(
            f"  {icon} {regime_name}: {label} ({grade}-grade){annotation}"
        )
    lines.append("")

    # Recommendations
    if failures:
        lines.append("Recommendations:")
        for i, f in enumerate(failures, 1):
            lines.append(f"  {i}. {f['recommendation']}")
        lines.append("")

    # Final verdict
    if critical_count > 0:
        lines.append(
            "Verdict: ❌ DO NOT DEPLOY — resolve critical issues before wiring to MT5"
        )
    elif warning_count > 0:
        lines.append(
            "Verdict: ⚠️ CONDITIONAL DEPLOY — connect when trend-reversal delay accepted"
        )
    else:
        lines.append(
            "Verdict: ✅ READY FOR DEPLOYMENT — governor safe to wire to MT5"
        )

    # Summary stats
    total_cycles = sum(
        m.get("cycles", 0) for m in regime_metrics.values()
    )
    total_fp = sum(
        m.get("comparison", {}).get("false_positives_prevented", 0)
        for m in regime_metrics.values()
    )
    total_correct = sum(
        m.get("comparison", {}).get("correct_exits_allowed", 0)
        for m in regime_metrics.values()
    )
    lines.append("")
    lines.append(f"  Total cycles tested: {total_cycles}")
    lines.append(f"  False positives prevented: {total_fp}")
    lines.append(f"  Correct exits allowed: {total_correct}")
    lines.append("")

    return "\n".join(lines)


# ===================================================================
# GovernorStressTest
# ===================================================================


class GovernorStressTest:
    """Stress-test the Execution Governor across diverse regime topologies.

    Tests the 3-gate system (Persistence, Price-Context, Reversal) against
    synthetic but realistic market regimes to find failure boundaries.
    """

    def __init__(self, governor: Optional[ExecutionGovernor] = None) -> None:
        self.governor = governor or ExecutionGovernor()

        # Storage for results
        self._results: Dict[str, Any] = {
            "regime_metrics": {},
            "raw_data": {},
            "failure_modes": [],
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_regime_battery(self) -> Dict[str, Any]:
        """Run all predefined regime types and collect results.

        Returns
        -------
        dict
            Nested results structure with per-regime metrics.
        """
        self._results = {
            "regime_metrics": {},
            "raw_data": {},
            "failure_modes": [],
        }

        for regime_name, generator in REGIME_GENERATORS.items():
            logger.info(f"Running regime: {regime_name}")
            result = self._run_single_regime(regime_name, generator)
            self._results["regime_metrics"][regime_name] = result["metrics"]
            self._results["raw_data"][regime_name] = result["raw"]

        # Compute failure modes
        self._results["failure_modes"] = analyze_failure_modes(self._results)

        return self._results

    def report_statistics(self) -> Dict[str, Any]:
        """Compute aggregate metrics across all regimes.

        Returns
        -------
        dict
            Aggregated statistics.
        """
        regime_metrics = self._results.get("regime_metrics", {})
        if not regime_metrics:
            return {}

        agg: Dict[str, Any] = {
            "total_cycles": 0,
            "total_rfe_exits": 0,
            "total_governor_exits": 0,
            "total_false_positives_prevented": 0,
            "total_correct_exits_allowed": 0,
            "total_false_exits": 0,
            "total_missed_exits": 0,
            "avg_delay": 0.0,
            "avg_suppression_rate": 0.0,
            "avg_net_expectancy_delta": 0.0,
            "regime_count": len(regime_metrics),
        }

        delay_sum = 0.0
        suppression_sum = 0.0
        delta_sum = 0.0
        count = 0

        for m in regime_metrics.values():
            gov = m.get("governor", {})
            comp = m.get("comparison", {})

            agg["total_cycles"] += m.get("cycles", 0)
            agg["total_rfe_exits"] += comp.get("rfe_exits", 0)
            agg["total_governor_exits"] += comp.get("governor_exits", 0)
            agg["total_false_positives_prevented"] += comp.get(
                "false_positives_prevented", 0
            )
            agg["total_correct_exits_allowed"] += comp.get(
                "correct_exits_allowed", 0
            )
            agg["total_false_exits"] += comp.get("false_exits", 0)
            agg["total_missed_exits"] += comp.get("missed_exits", 0)

            delay_sum += gov.get("avg_delay", 0)
            suppression_sum += gov.get("suppression_rate", 0)
            delta_sum += comp.get("net_expectancy_delta", 0)
            count += 1

        agg["avg_delay"] = round(delay_sum / max(1, count), 2)
        agg["avg_suppression_rate"] = round(suppression_sum / max(1, count), 4)
        agg["avg_net_expectancy_delta"] = round(delta_sum / max(1, count), 4)

        return agg

    def regime_breakdown(self) -> Dict[str, Any]:
        """Per-regime performance breakdown."""
        return self._results.get("regime_metrics", {})

    def failure_mode_analysis(self) -> List[Dict[str, Any]]:
        """Identify regime-dependent failure modes."""
        return self._results.get("failure_modes", [])

    def readiness_verdict(self) -> Dict[str, Any]:
        """Final deployment readiness assessment.

        Returns
        -------
        dict
            Structured verdict with overall status, per-gate results,
            per-regime results, and recommendations.
        """
        regime_metrics = self._results.get("regime_metrics", {})
        failures = self._results.get("failure_modes", [])

        critical_count = sum(1 for f in failures if f["severity"] == "CRITICAL")
        warning_count = sum(1 for f in failures if f["severity"] == "WARNING")

        if critical_count > 0:
            overall = "FAIL"
        elif warning_count > 0:
            overall = "CONDITIONAL_PASS"
        else:
            overall = "PASS"

        gates = {
            "TemporalPersistenceGate": True,
            "PriceContextWeighting": True,
            "ReversalFilter": True,
        }
        for f in failures:
            issue_lower = f["issue"].lower()
            if "persistence" in issue_lower or "delay" in issue_lower:
                if f["severity"] == "CRITICAL":
                    gates["TemporalPersistenceGate"] = False
            if "reversal" in issue_lower or "suppression" in issue_lower:
                if f["severity"] == "CRITICAL":
                    gates["ReversalFilter"] = False

        regimes_status = {}
        for regime_name in sorted(regime_metrics.keys()):
            m = regime_metrics[regime_name]
            grade = m.get("grade", _regime_grade(m))
            regime_failure = next(
                (f for f in failures if f["regime"] == regime_name), None
            )
            if regime_failure and regime_failure["severity"] == "CRITICAL":
                status = "FAIL"
            elif regime_failure and regime_failure["severity"] == "WARNING":
                status = "WARNING"
            else:
                status = "PASS"
            regimes_status[regime_name] = {
                "status": status,
                "grade": grade,
                "issue": regime_failure["issue"] if regime_failure else None,
            }

        return {
            "overall": overall,
            "gates": gates,
            "regimes": regimes_status,
            "failures": failures,
            "recommendations": [f["recommendation"] for f in failures],
        }

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def format_performance_map(self) -> str:
        """Return formatted performance table."""
        return format_performance_map(self._results)

    def format_readiness_verdict(self) -> str:
        """Return formatted deployment readiness verdict."""
        return format_readiness_verdict(self._results)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_single_regime(
        self,
        regime_name: str,
        generator,
    ) -> Dict[str, Any]:
        """Run a single regime through the governor and compute metrics."""
        # Generate cycles
        regime_cycles = generator()
        cycles = len(regime_cycles)
        symbol = next(iter(regime_cycles[0]["evaluations"].keys()))

        # Extract pressure series
        pressures = extract_pressure_series(regime_cycles)

        # Compute ground truth
        ground_truth = compute_ground_truth(regime_cycles)

        # Extract raw RFE decisions
        rfe_decisions = extract_rfe_decisions(regime_cycles)

        # Run through governor
        self.governor.reset()
        governor_exit_decisions: List[bool] = []

        for i, cycle in enumerate(regime_cycles):
            # No price history for synthetic regimes (price context = neutral)
            gov_result = self.governor.evaluate(cycle)
            # Get the decision for the symbol
            decision = gov_result["decisions"][symbol]
            action_type = decision["action"]["type"]
            governor_exit_decisions.append(
                action_type in ("CLOSE", "CLOSE_PARTIAL")
            )

        # Compute metrics
        metrics = StressMetrics.compute(
            rfe_decisions, governor_exit_decisions, ground_truth
        )
        metrics["cycles"] = cycles
        metrics["grade"] = _regime_grade(metrics)

        return {
            "metrics": metrics,
            "raw": {
                "pressures": pressures,
                "rfe_decisions": rfe_decisions,
                "governor_decisions": governor_exit_decisions,
                "ground_truth": ground_truth,
            },
        }
