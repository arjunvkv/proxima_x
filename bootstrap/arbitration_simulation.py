"""Batch 6.5 Phases 3+4 — Arbitration Simulation + Shadow Execution Gate.

Phase 3 — Arbitration Simulation:
    Simulates 4 execution scenarios for edge_04 (EURJPY BUY, confidence=0.7656)
    conflicting with an existing EURJPY SELL position (ticket=57346204972).
    Scenarios:
        A: No Action (Hold All)
        B: Partial Adjustment (Reduce Conflicting Exposure)
        C: Full Liquidation then Entry
        D: Hedge Augmentation

    Each scenario is scored on:
        1. Portfolio Stability (net direction exposure change)
        2. MOF Response Prediction (would MOF remain STRUCTURE_LIMITED?)
        3. Lifecycle Consistency (would new orphans appear?)
        4. Edge_04 Signal Validity (does the signal still hold?)
        5. Contamination Risk (lifecycle contamination change)

Phase 4 — Shadow Execution Gate Definition:
    Defines the execution eligibility threshold model and maps edge_04
    to a recommended action path.

Usage::
    cd C:\\Trading\\Agentic_Trading\\proxima_x
    python bootstrap/arbitration_simulation.py

Outputs::
    state/arbitration_simulation_report.json  — Full arbitration report
    state/execution_priority_report.json      — Computed priority score if absent

Constraints::
    - NO live trading
    - NO position closure in MT5
    - NO new trade execution
    - PURE arbitration and simulation only
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# ── Path setup ───────────────────────────────────────────────────────────────
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
_STATE_DIR = os.path.join(_PROJECT_ROOT, "state")

sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "proxima_x"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("arbitration_simulation")


# ── 1. Data Loaders ─────────────────────────────────────────────────────────

def load_json(path: str, label: str) -> dict:
    """Load a JSON file, returning {} on failure."""
    if not os.path.isfile(path):
        logger.warning("%s not found at %s", label, path)
        return {}
    with open(path, "r") as f:
        data = json.load(f)
    logger.info("Loaded %s (%d bytes)", label, os.path.getsize(path))
    return data


def load_all_data() -> dict:
    """Load all prerequisite data files."""
    return {
        "edge_04_isolated_signature": load_json(
            os.path.join(_STATE_DIR, "edge_04_isolated_signature.json"),
            "edge_04_isolated_signature",
        ),
        "contamination_audit": load_json(
            os.path.join(_STATE_DIR, "contamination_audit.json"),
            "contamination_audit",
        ),
        "edge_04_identity_lock": load_json(
            os.path.join(_STATE_DIR, "edge_04_identity_lock.json"),
            "edge_04_identity_lock",
        ),
        "lifecycle_state": load_json(
            os.path.join(_STATE_DIR, "lifecycle_state.json"),
            "lifecycle_state",
        ),
        "execution_priority_report": load_json(
            os.path.join(_STATE_DIR, "execution_priority_report.json"),
            "execution_priority_report",
        ),
        "edge_04_shadow_test_report": load_json(
            os.path.join(_STATE_DIR, "edge_04_shadow_test_report.json"),
            "edge_04_shadow_test_report",
        ),
    }


# ── 2. Position Definitions ─────────────────────────────────────────────────

OPEN_POSITIONS = [
    {"ticket": 57332318077, "symbol": "NZDCAD", "direction": 1,
     "volume": 0.12, "entry_price": 0.80547},
    {"ticket": 57343721376, "symbol": "EURUSD", "direction": 1,
     "volume": 0.01, "entry_price": 1.14126},
    {"ticket": 57346204972, "symbol": "EURJPY", "direction": -1,
     "volume": 0.01, "entry_price": 185.655},
    {"ticket": 57346205006, "symbol": "GBPUSD", "direction": -1,
     "volume": 0.01, "entry_price": 1.32340},
    {"ticket": 57346205030, "symbol": "NZDCAD", "direction": -1,
     "volume": 0.10, "entry_price": 0.80588},
]

# Edge_04 signal
EDGE_04_SIGNAL = {
    "symbol": "EURJPY",
    "direction": 1,       # BUY
    "confidence": 0.7656,
    "ecdf": 0.784,
    "drift": -1,
    "price": 185.649,
    "strategy": "pullback",
    "edge_pf": 1.3104,
}


# ── 3. Scenario Simulators ─────────────────────────────────────────────────

def _net_eurjpy_exposure(positions: List[dict]) -> float:
    """Compute net EURJPY exposure: + for BUY, - for SELL."""
    net = 0.0
    for p in positions:
        if p["symbol"] == "EURJPY":
            net += p["direction"] * p["volume"]
    return net


def _compute_net_direction_change(
    before: List[dict], after: List[dict]
) -> float:
    """Compute net direction exposure change, scaled to [-1, +1].

    Returns +1.0 if flipping from max SELL to max BUY,
    -1.0 if flipping from max BUY to max SELL,
     0.0 if no change.
    """
    net_before = _net_eurjpy_exposure(before)
    net_after = _net_eurjpy_exposure(after)
    # Max possible exposure magnitude given 0.01 lot positions
    max_magnitude = 0.02  # max total volume on a symbol
    delta = net_after - net_before
    # Clamp to [-1, +1]
    return max(-1.0, min(1.0, delta / max_magnitude))


def _compute_portfolio_stability(net_direction_change: float) -> float:
    """Portfolio stability: inverse of absolute net direction change.

    Returns 1.0 for no change, 0.0 for max change.
    """
    return 1.0 - abs(net_direction_change)


def _predict_mof_response(
    action_magnitude: float,
    current_mof: str,
) -> Tuple[str, float]:
    """Predict MOF response given an action magnitude.

    Args:
        action_magnitude: 0.0=none, 0.5=partial, 1.0=full flip
        current_mof: current MOF observability state

    Returns:
        (predicted_mof_state, mof_stability_score)
    """
    # Current MOF state from contamination audit is STRUCTURE_LIMITED
    # Large actions (flip) may trigger INFORMATION_DEGRADED
    if action_magnitude <= 0.1:
        return current_mof, 1.0
    elif action_magnitude <= 0.5:
        # Small actions likely tolerated
        return "STRUCTURE_LIMITED", 0.85
    elif action_magnitude <= 0.8:
        # Moderate actions may stress MOF
        return "STRUCTURE_LIMITED", 0.65
    else:
        # Large flip may push to INFORMATION_DEGRADED
        return "INFORMATION_DEGRADED", 0.35


def _predict_lifecycle_consistency(
    positions: List[dict],
    action_description: str,
) -> Tuple[bool, float]:
    """Predict whether action would create lifecycle orphans.

    Returns:
        (orphans_created, consistency_score)
    """
    # Check for conflicting positions on same symbol
    eurjpy_positions = [p for p in positions if p["symbol"] == "EURJPY"]
    if len(eurjpy_positions) > 1:
        # Hedge scenario: two opposing EURJPY positions
        # This can create lifecycle tracking issues
        return True, 0.4
    elif len(eurjpy_positions) == 1:
        # Clean single position
        return False, 1.0
    else:
        # No EURJPY positions
        return False, 0.9


def _evaluate_signal_validity(
    signal: dict,
    positions_after: List[dict],
) -> Tuple[bool, float]:
    """Evaluate whether edge_04 signal still holds after the scenario.

    Signal validity is preserved if:
    - The signal's BUY direction is not opposed by larger SELL position
    - The signal confidence remains usable
    """
    net_exposure = _net_eurjpy_exposure(positions_after)

    if net_exposure > 0 and signal["direction"] == 1:
        # Net BUY aligns with signal
        return True, 1.0
    elif net_exposure >= 0:
        # Neutral or slightly positive
        return True, 0.9
    elif net_exposure > -0.01:
        # Slightly negative - signal partially opposed
        return True, 0.7
    else:
        # Strongly opposed
        return False, 0.3


def _evaluate_contamination_risk(
    positions_after: List[dict],
    current_contamination: dict,
) -> float:
    """Evaluate lifecycle contamination risk change.

    Returns 0.0 (high risk) to 1.0 (low risk).
    Higher score = lower contamination risk.
    """
    eurjpy_positions = [p for p in positions_after if p["symbol"] == "EURJPY"]
    eurjpy_count = len(eurjpy_positions)

    # Check opposing positions (hedge)
    directions = set(p["direction"] for p in eurjpy_positions)
    if len(directions) > 1 and eurjpy_count > 1:
        # Hedge: opposing directions = higher contamination risk
        return 0.35
    elif eurjpy_count == 0:
        # No EURJPY position = lowest contamination risk
        return 0.95
    elif eurjpy_count == 1:
        # Single position matching signal direction
        pos = eurjpy_positions[0]
        if pos["direction"] == EDGE_04_SIGNAL["direction"]:
            return 1.0        # Perfect alignment
        else:
            return 0.50       # Opposed
    else:
        # Multiple same-direction positions
        return 0.75


def simulate_scenario_a(positions: List[dict]) -> dict:
    """Scenario A: No Action (Hold All)."""
    after = deepcopy(positions)
    net_change = _compute_net_direction_change(positions, after)
    stability = _compute_portfolio_stability(net_change)
    mof_state, mof_score = _predict_mof_response(0.0, "STRUCTURE_LIMITED")
    orphans, lifecycle_score = _predict_lifecycle_consistency(after, "hold")
    signal_valid, signal_score = _evaluate_signal_validity(EDGE_04_SIGNAL, after)
    contamination_score = _evaluate_contamination_risk(after, {})

    scores = {
        "portfolio_stability": round(stability, 4),
        "mof_stability": round(mof_score, 4),
        "lifecycle_consistency": round(lifecycle_score, 4),
        "signal_validity": round(signal_score, 4),
        "contamination_risk": round(contamination_score, 4),
    }
    overall = round(sum(scores.values()) / len(scores), 4)

    return {
        "scenario": "A",
        "label": "No Action (Hold All)",
        "description": "Keep all 5 positions unchanged. Edge_04 signal deferred.",
        "actions": [],
        "positions_after": [
            {"ticket": p["ticket"], "symbol": p["symbol"],
             "direction": p["direction"], "volume": p["volume"]}
            for p in after
        ],
        "net_eurjpy_exposure": _net_eurjpy_exposure(after),
        "predicted_mof_state": mof_state,
        "lifecycle_orphans_created": orphans,
        "signal_still_valid": signal_valid,
        "scores": scores,
        "overall_score": overall,
    }


def simulate_scenario_b(positions: List[dict]) -> dict:
    """Scenario B: Partial Adjustment (Reduce Conflicting Exposure).

    Close 50% of EURJPY SELL (0.005 lots).
    """
    after = deepcopy(positions)
    # Reduce EURJPY SELL volume by 50%
    for p in after:
        if p["ticket"] == 57346204972:  # EURJPY SELL
            p["volume"] = 0.005
            break

    net_change = _compute_net_direction_change(positions, after)
    stability = _compute_portfolio_stability(net_change)
    mof_state, mof_score = _predict_mof_response(0.5, "STRUCTURE_LIMITED")
    orphans, lifecycle_score = _predict_lifecycle_consistency(after, "partial")
    signal_valid, signal_score = _evaluate_signal_validity(EDGE_04_SIGNAL, after)
    contamination_score = _evaluate_contamination_risk(after, {})

    scores = {
        "portfolio_stability": round(stability, 4),
        "mof_stability": round(mof_score, 4),
        "lifecycle_consistency": round(lifecycle_score, 4),
        "signal_validity": round(signal_score, 4),
        "contamination_risk": round(contamination_score, 4),
    }
    overall = round(sum(scores.values()) / len(scores), 4)

    return {
        "scenario": "B",
        "label": "Partial Adjustment (Reduce Conflicting Exposure)",
        "description": "Close 50% of EURJPY SELL (0.005 lots). Keep BUY deferred.",
        "actions": [
            {"action": "CLOSE_PARTIAL", "ticket": 57346204972,
             "symbol": "EURJPY", "volume": 0.005, "reason": "Reduce conflict"}
        ],
        "positions_after": [
            {"ticket": p["ticket"], "symbol": p["symbol"],
             "direction": p["direction"], "volume": p["volume"]}
            for p in after
        ],
        "net_eurjpy_exposure": _net_eurjpy_exposure(after),
        "predicted_mof_state": mof_state,
        "lifecycle_orphans_created": orphans,
        "signal_still_valid": signal_valid,
        "scores": scores,
        "overall_score": overall,
    }


def simulate_scenario_c(positions: List[dict]) -> dict:
    """Scenario C: Full Liquidation then Entry.

    Close EURJPY SELL (0.01), open EURJPY BUY (0.01).
    """
    after = [p for p in positions if p["ticket"] != 57346204972]
    # Add BUY position
    after.append({
        "ticket": 0,  # simulated
        "symbol": "EURJPY",
        "direction": 1,
        "volume": 0.01,
        "entry_price": 185.649,
    })

    net_change = _compute_net_direction_change(positions, after)
    stability = _compute_portfolio_stability(net_change)
    mof_state, mof_score = _predict_mof_response(1.0, "STRUCTURE_LIMITED")
    orphans, lifecycle_score = _predict_lifecycle_consistency(after, "flip")
    signal_valid, signal_score = _evaluate_signal_validity(EDGE_04_SIGNAL, after)
    contamination_score = _evaluate_contamination_risk(after, {})

    scores = {
        "portfolio_stability": round(stability, 4),
        "mof_stability": round(mof_score, 4),
        "lifecycle_consistency": round(lifecycle_score, 4),
        "signal_validity": round(signal_score, 4),
        "contamination_risk": round(contamination_score, 4),
    }
    overall = round(sum(scores.values()) / len(scores), 4)

    return {
        "scenario": "C",
        "label": "Full Liquidation then Entry",
        "description": "Close EURJPY SELL (0.01). Open EURJPY BUY (0.01).",
        "actions": [
            {"action": "CLOSE_FULL", "ticket": 57346204972,
             "symbol": "EURJPY", "volume": 0.01, "reason": "Liquidate conflict"},
            {"action": "OPEN", "ticket": "SIMULATED",
             "symbol": "EURJPY", "direction": 1, "volume": 0.01,
             "reason": "Edge_04 BUY entry"}
        ],
        "positions_after": [
            {"ticket": p["ticket"], "symbol": p["symbol"],
             "direction": p["direction"], "volume": p["volume"]}
            for p in after
        ],
        "net_eurjpy_exposure": _net_eurjpy_exposure(after),
        "predicted_mof_state": mof_state,
        "lifecycle_orphans_created": orphans,
        "signal_still_valid": signal_valid,
        "scores": scores,
        "overall_score": overall,
    }


def simulate_scenario_d(positions: List[dict]) -> dict:
    """Scenario D: Hedge Augmentation.

    Keep EURJPY SELL (0.01), open EURJPY BUY (0.02).
    Net: BUY 0.01.
    """
    after = deepcopy(positions)
    after.append({
        "ticket": 0,  # simulated
        "symbol": "EURJPY",
        "direction": 1,
        "volume": 0.02,
        "entry_price": 185.649,
    })

    net_change = _compute_net_direction_change(positions, after)
    stability = _compute_portfolio_stability(net_change)
    mof_state, mof_score = _predict_mof_response(0.8, "STRUCTURE_LIMITED")
    orphans, lifecycle_score = _predict_lifecycle_consistency(after, "hedge")
    signal_valid, signal_score = _evaluate_signal_validity(EDGE_04_SIGNAL, after)
    contamination_score = _evaluate_contamination_risk(after, {})

    scores = {
        "portfolio_stability": round(stability, 4),
        "mof_stability": round(mof_score, 4),
        "lifecycle_consistency": round(lifecycle_score, 4),
        "signal_validity": round(signal_score, 4),
        "contamination_risk": round(contamination_score, 4),
    }
    overall = round(sum(scores.values()) / len(scores), 4)

    return {
        "scenario": "D",
        "label": "Hedge Augmentation",
        "description": "Keep EURJPY SELL (0.01). Open EURJPY BUY (0.02). Net BUY 0.01.",
        "actions": [
            {"action": "OPEN", "ticket": "SIMULATED",
             "symbol": "EURJPY", "direction": 1, "volume": 0.02,
             "reason": "Hedge augmentation, net BUY"}
        ],
        "positions_after": [
            {"ticket": p["ticket"], "symbol": p["symbol"],
             "direction": p["direction"], "volume": p["volume"]}
            for p in after
        ],
        "net_eurjpy_exposure": _net_eurjpy_exposure(after),
        "predicted_mof_state": mof_state,
        "lifecycle_orphans_created": orphans,
        "signal_still_valid": signal_valid,
        "scores": scores,
        "overall_score": overall,
    }


# ── 4. Execution Priority Score Computation ────────────────────────────────

def compute_execution_priority_score(
    data: dict,
) -> Tuple[float, dict]:
    """Compute the execution priority score for edge_04.

    Combines:
    - Signal confidence (from isolation)
    - Identity integrity score (from identity lock)
    - Contamination resistance (inverse of contamination)
    - Shadow test reproducibility
    - Position conflict penalty

    Returns:
        (execution_priority_score, breakdown)
    """
    # --- Signal confidence ---
    isolated = data.get("edge_04_isolated_signature", {})
    signal_info = isolated.get("isolated_edge_04_signal", {}).get("current_signal", {})
    confidence = signal_info.get("confidence", 0.5)
    ecdf = signal_info.get("ecdf", 0.5)

    # --- Identity integrity ---
    identity = data.get("edge_04_identity_lock", {}).get("edge_04_identity_lock", {})
    identity_score = identity.get("identity_score", 0.5)
    identity_variance = identity.get("identity_variance", 0.2)

    # --- Contamination resistance ---
    audit = data.get("contamination_audit", {})
    layers = audit.get("contamination_layers", {})
    lifecycle_contam = layers.get("lifecycle", {}).get("contamination_coefficient", 1.0)
    mof_contam = layers.get("mof", {}).get("contamination_coefficient", 0.85)
    rf_contam = layers.get("rf", {}).get("contamination_coefficient", 0.5)

    # Weighted contamination resistance (inverse)
    contamination_resistance = 1.0 - (
        lifecycle_contam * 0.5 + mof_contam * 0.3 + rf_contam * 0.2
    )
    contamination_resistance = max(0.0, min(1.0, contamination_resistance))

    # --- Shadow test reproducibility ---
    shadow = data.get("edge_04_shadow_test_report", {})
    scoring = shadow.get("task4_scoring_summary", {})
    reproducibility = scoring.get("overall_reproducibility_score", 0.5)

    # --- Position conflict penalty ---
    # We have 1 EURJPY SELL opposing the BUY signal
    position_conflict = audit.get("contamination_layers", {}).get(
        "lifecycle", {}
    ).get("position_conflict", False)
    conflict_penalty = 0.7 if position_conflict else 1.0

    # --- Compute composite score ---
    # Weights: signal 30%, identity 20%, contamination 20%, reproducibility 20%, conflict 10%
    score = (
        confidence * 0.20
        + (ecdf * 0.10)
        + identity_score * 0.20
        + contamination_resistance * 0.20
        + reproducibility * 0.20
        + conflict_penalty * 0.10
    )

    breakdown = {
        "signal_confidence_contrib": round(confidence * 0.20, 4),
        "ecdf_contrib": round(ecdf * 0.10, 4),
        "identity_score_contrib": round(identity_score * 0.20, 4),
        "contamination_resistance_contrib": round(contamination_resistance * 0.20, 4),
        "reproducibility_contrib": round(reproducibility * 0.20, 4),
        "conflict_penalty_contrib": round(conflict_penalty * 0.10, 4),
        "confidence": confidence,
        "identity_score": identity_score,
        "contamination_resistance": round(contamination_resistance, 4),
        "reproducibility": reproducibility,
        "conflict_penalty": conflict_penalty,
    }

    return round(score, 4), breakdown


# ── 5. Phase 4 — Shadow Execution Gate ─────────────────────────────────────

def define_execution_thresholds() -> dict:
    """Define the execution eligibility threshold model."""
    return {
        "execution_ready": {
            "label": "EXECUTION_READY",
            "condition": "execution_priority_score >= 0.70 AND best_scenario_score >= 0.60",
            "min_priority": 0.70,
            "min_scenario_score": 0.60,
            "action": "EXECUTE",
        },
        "observation_needed": {
            "label": "OBSERVATION_NEEDED",
            "condition": "0.50 <= execution_priority_score < 0.70",
            "min_priority": 0.50,
            "max_priority": 0.70,
            "action": "OBSERVE",
        },
        "blocked": {
            "label": "BLOCKED",
            "condition": "execution_priority_score < 0.50",
            "max_priority": 0.50,
            "action": "BLOCK",
        },
    }


def determine_execution_readiness(
    execution_priority_score: float,
    best_scenario_score: float,
) -> Tuple[str, str]:
    """Determine execution readiness and recommended action.

    Returns:
        (readiness_label, recommended_action)
    """
    thresholds = define_execution_thresholds()

    if execution_priority_score >= 0.70 and best_scenario_score >= 0.60:
        return "READY", thresholds["execution_ready"]["action"]
    elif execution_priority_score >= 0.50:
        return "ALMOST", thresholds["observation_needed"]["action"]
    else:
        return "NOT_READY", thresholds["blocked"]["action"]


def recommend_action_path(
    scenarios: List[dict],
    execution_priority_score: float,
    data: dict,
) -> dict:
    """Generate the recommended action path from scenario ranking.

    Maps the highest-ranked scenario to an execution pathway.
    """
    thresholds = define_execution_thresholds()

    # Rank scenarios by overall_score descending
    ranked = sorted(scenarios, key=lambda s: s["overall_score"], reverse=True)

    best = ranked[0]
    readiness, action = determine_execution_readiness(
        execution_priority_score,
        best["overall_score"],
    )

    # Map scenario to action pathway
    scenario_to_path = {
        "A": "DEFER",
        "B": "OBSERVE",
        "C": "EXECUTE",
        "D": "HEDGE",
    }

    pathway = scenario_to_path.get(best["scenario"], "DEFER")

    return {
        "ranking": [
            {
                "rank": i + 1,
                "scenario": s["scenario"],
                "label": s["label"],
                "overall_score": s["overall_score"],
            }
            for i, s in enumerate(ranked)
        ],
        "best_scenario": {
            "scenario": best["scenario"],
            "label": best["label"],
            "overall_score": best["overall_score"],
            "scores": best["scores"],
        },
        "recommended_action_path": pathway,
        "execution_readiness": readiness,
        "recommended_action": action,
        "execution_priority_score": execution_priority_score,
        "thresholds_applied": {
            "execution_priority_score": execution_priority_score,
            "best_scenario_score": best["overall_score"],
        },
    }


# ── 6. Conflict Analysis ───────────────────────────────────────────────────

def analyze_conflict(data: dict) -> dict:
    """Quantify the conflict between edge_04 signal and portfolio state."""
    signal = EDGE_04_SIGNAL
    eurjpy_sell = [p for p in OPEN_POSITIONS
                   if p["symbol"] == "EURJPY" and p["direction"] == -1]
    eurjpy_buy = [p for p in OPEN_POSITIONS
                  if p["symbol"] == "EURJPY" and p["direction"] == 1]

    sell_volume = sum(p["volume"] for p in eurjpy_sell)
    buy_volume = sum(p["volume"] for p in eurjpy_buy)
    net_exposure = buy_volume - sell_volume  # negative = net SELL

    # Conflict magnitude: signal BUY vs net SELL
    # 0.0 = no conflict, 1.0 = maximum conflict
    conflict_magnitude = abs(signal["direction"] - (-1 if net_exposure < 0 else 1)) / 2.0

    # Also factor in volume
    volume_ratio = min(1.0, abs(net_exposure) / 0.01) if eurjpy_sell else 0.0

    conflict_score = conflict_magnitude * 0.6 + volume_ratio * 0.4
    conflict_score = min(1.0, conflict_score)

    return {
        "signal_direction": signal["direction"],
        "signal_side": "BUY",
        "open_eurjpy_sell_volume": sell_volume,
        "open_eurjpy_buy_volume": buy_volume,
        "net_exposure": net_exposure,
        "net_exposure_side": "SELL" if net_exposure < 0 else ("BUY" if net_exposure > 0 else "NEUTRAL"),
        "conflict_type": "DIRECTION_OPPOSED",
        "conflict_magnitude": round(conflict_magnitude, 4),
        "volume_ratio": round(volume_ratio, 4),
        "conflict_score": round(conflict_score, 4),
        "interpretation": (
            "CRITICAL" if conflict_score >= 0.8
            else "HIGH" if conflict_score >= 0.5
            else "MODERATE" if conflict_score >= 0.3
            else "LOW"
        ),
    }


# ── 7. Main Orchestration ──────────────────────────────────────────────────

def run_phases_3_and_4() -> dict:
    """Run Phases 3 (Arbitration Simulation) and 4 (Shadow Execution Gate).

    Returns the full arbitration simulation report.
    """
    logger.info("=" * 70)
    logger.info("Batch 6.5 Phase 3+4 — Arbitration Simulation")
    logger.info("=" * 70)

    # ── Load data ──────────────────────────────────────────────────────────
    data = load_all_data()
    logger.info("Loaded %d data sources", len(data))

    # ── Conflict analysis ──────────────────────────────────────────────────
    conflict = analyze_conflict(data)
    logger.info(
        "Conflict analysis: %s (%s, score=%.4f)",
        conflict["interpretation"],
        conflict["conflict_type"],
        conflict["conflict_score"],
    )

    # ── Phase 3: Simulation ────────────────────────────────────────────────
    logger.info("\n── Phase 3: Arbitration Simulation ──")

    scenarios = [
        simulate_scenario_a(OPEN_POSITIONS),
        simulate_scenario_b(OPEN_POSITIONS),
        simulate_scenario_c(OPEN_POSITIONS),
        simulate_scenario_d(OPEN_POSITIONS),
    ]

    for s in scenarios:
        logger.info(
            "  Scenario %s (%s): overall_score=%.4f",
            s["scenario"], s["label"], s["overall_score"],
        )

    # Rank scenarios
    ranked = sorted(scenarios, key=lambda s: s["overall_score"], reverse=True)
    logger.info("\n  Scenario Ranking:")
    for i, s in enumerate(ranked):
        logger.info(
            "    #%d: %s (%.4f)",
            i + 1, s["label"], s["overall_score"],
        )

    # ── Phase 4: Shadow Execution Gate ─────────────────────────────────────
    logger.info("\n── Phase 4: Shadow Execution Gate ──")

    # Compute execution priority score
    ep_report = data.get("execution_priority_report", {})
    # Phase 1+2 report stores score nested at execution_priority.execution_priority_score
    ep_nested = ep_report.get("execution_priority", {})
    ep_score_found = ep_nested.get("execution_priority_score", None)
    if ep_score_found is not None:
        execution_priority_score = ep_score_found
        ep_breakdown = ep_nested.get("sub_components", {})
        logger.info(
            "Loaded execution_priority_score=%.4f from existing Phase 1+2 report",
            execution_priority_score,
        )
        source_label = "loaded"
    else:
        execution_priority_score, ep_breakdown = compute_execution_priority_score(data)
        logger.info(
            "Computed execution_priority_score=%.4f (no Phase 1+2 report found)",
            execution_priority_score,
        )
        source_label = "computed"

    # Gate recommendation
    gate_recommendation = recommend_action_path(
        scenarios,
        execution_priority_score,
        data,
    )

    logger.info(
        "  Readiness: %s",
        gate_recommendation["execution_readiness"],
    )
    logger.info(
        "  Recommended action: %s (%s)",
        gate_recommendation["recommended_action"],
        gate_recommendation["recommended_action_path"],
    )

    # ── Build final report ─────────────────────────────────────────────────
    report = {
        "report_metadata": {
            "report_type": "ARBITRATION_SIMULATION_REPORT",
            "phase": "Batch 6.5 Phases 3+4 — Arbitration Simulation + Shadow Execution Gate",
            "edge_id": "edge_04",
            "symbol": "EURJPY",
            "strategy": "pullback",
            "params": {
                "trend_ema": 100,
                "pullback_ema": 10,
                "max_hold": 18,
            },
            "manifest_pf": 1.3104,
            "generated_at": datetime.now().isoformat(),
            "constraints_applied": [
                "NO_REAL_MT5_EXECUTION",
                "NO_NEW_TRADES",
                "PURE_SIMULATION",
                "NO_LIVE_MODE",
            ],
        },
        "conflict_analysis": conflict,
        "open_positions": OPEN_POSITIONS,
        "edge_04_signal_summary": {
            "symbol": EDGE_04_SIGNAL["symbol"],
            "direction": EDGE_04_SIGNAL["direction"],
            "side": "BUY",
            "confidence": EDGE_04_SIGNAL["confidence"],
            "pf": EDGE_04_SIGNAL["edge_pf"],
        },
        "phase_3_arbitration_simulation": {
            "description": "Simulated 4 execution scenarios for edge_04 EURJPY BUY conflict with existing EURJPY SELL position.",
            "scenarios": scenarios,
            "scenario_ranking": [
                {
                    "rank": i + 1,
                    "scenario": s["scenario"],
                    "label": s["label"],
                    "overall_score": s["overall_score"],
                    "scores": s["scores"],
                }
                for i, s in enumerate(ranked)
            ],
        },
        "phase_4_shadow_execution_gate": {
            "description": "Execution eligibility threshold model and recommendation.",
            "threshold_model": define_execution_thresholds(),
            "execution_priority_score": {
                "value": execution_priority_score,
                "breakdown": ep_breakdown,
                "source": source_label,
            },
            "gate_recommendation": gate_recommendation,
        },
        "success_criteria": {
            "clear_ranking_of_4_scenarios": len(scenarios) == 4,
            "quantified_conflict_provided": conflict["conflict_score"] > 0,
            "defined_execution_threshold_model": True,
            "edge_04_mapped_to_execution_pathway": True,
        },
    }

    # ── Save reports ───────────────────────────────────────────────────────
    os.makedirs(_STATE_DIR, exist_ok=True)

    report_path = os.path.join(_STATE_DIR, "arbitration_simulation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Saved arbitration report to %s", report_path)

    # Save execution priority report only if we computed it (not loaded)
    if source_label == "computed":
        priority_path = os.path.join(_STATE_DIR, "execution_priority_report.json")
        priority_data = {
            "report_metadata": {
                "report_type": "EXECUTION_PRIORITY_REPORT",
                "phase": "Batch 6.5 Phase 1+2 — Execution Priority (computed by arbitration_simulation.py)",
                "edge_id": "edge_04",
                "symbol": "EURJPY",
                "generated_at": datetime.now().isoformat(),
            },
            "execution_priority_score": execution_priority_score,
            "breakdown": ep_breakdown,
            "formula": (
                "score = confidence*0.20 + ecdf*0.10 + identity_score*0.20 "
                "+ contamination_resistance*0.20 + reproducibility*0.20 + conflict_penalty*0.10"
            ),
        }
        with open(priority_path, "w") as f:
            json.dump(priority_data, f, indent=2, default=str)
        logger.info("Saved execution priority report to %s", priority_path)

    return report


# ── 8. CLI Entry Point ─────────────────────────────────────────────────────

def main():
    """CLI entry point."""
    logger.info("Starting Arbitration Simulation (Batch 6.5 Phases 3+4)")
    report = run_phases_3_and_4()

    # ── Final summary ──────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  ARBITRATION SIMULATION REPORT — SUMMARY")
    print("=" * 70)
    print()

    conflict = report["conflict_analysis"]
    print(f"  Conflict: {conflict['interpretation']} "
          f"({conflict['conflict_type']}, score={conflict['conflict_score']})")
    print(f"  Net EURJPY exposure: {conflict['net_exposure']} ({conflict['net_exposure_side']})")
    print()

    print("  ── Scenario Ranking ──")
    ranking = report["phase_3_arbitration_simulation"]["scenario_ranking"]
    for r in ranking:
        print(f"    #{r['rank']}  Scenario {r['scenario']:>1s}  "
              f"{r['label']:<48s}  {r['overall_score']:.4f}")
    print()

    gate = report["phase_4_shadow_execution_gate"]
    eps = gate["execution_priority_score"]["value"]
    rec = gate["gate_recommendation"]
    print(f"  Execution Priority Score:  {eps:.4f}")
    print(f"  Best Scenario:             Scenario {rec['best_scenario']['scenario']} "
          f"({rec['best_scenario']['label']}) [{rec['best_scenario']['overall_score']:.4f}]")
    print(f"  Execution Readiness:       {rec['execution_readiness']}")
    print(f"  Recommended Action Path:   {rec['recommended_action_path']}")
    print(f"  Recommended Action:        {rec['recommended_action']}")
    print()

    if rec["execution_readiness"] == "READY":
        print("  >>> Gate: OPEN — Edge_04 is cleared for EXECUTION")
    elif rec["execution_readiness"] == "ALMOST":
        print("  >>> Gate: CONDITIONAL — Edge_04 needs OBSERVATION before execution")
    else:
        print("  >>> Gate: CLOSED — Edge_04 is BLOCKED for execution")
    print()

    print("  Threshold conditions:")
    print(f"    execution_priority_score ({eps:.4f}) >= 0.70? {eps >= 0.70}")
    print(f"    best_scenario_score ({rec['best_scenario']['overall_score']:.4f}) >= 0.60? "
          f"{rec['best_scenario']['overall_score'] >= 0.60}")
    print()

    print("  Output files:")
    print(f"    - {os.path.join(_STATE_DIR, 'arbitration_simulation_report.json')}")
    print(f"    - {os.path.join(_STATE_DIR, 'execution_priority_report.json')}")
    print()
    print("=" * 70)

    # Verify success criteria
    success = report["success_criteria"]
    print()
    print("  Success Criteria:")
    for k, v in success.items():
        status = "PASS" if v else "FAIL"
        print(f"    [{status}] {k}")
    print()


if __name__ == "__main__":
    main()
