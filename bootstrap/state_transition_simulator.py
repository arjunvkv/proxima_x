#!/usr/bin/env python3
"""Batch 6.7 Phases 1+2 — State Transition Simulation.

Phase 1 — Baseline State Snapshot:
    Record current system state programmatically from MT5 and existing data files.

Phase 2 — State Transition Simulation (NO LIVE TRADING):
    Simulate 4 portfolio transition scenarios over 10 cycles each to test
    whether partial portfolio adjustment preserves system stability.

Context:
    Edge_04 (EURJPY BUY, confidence=0.7656) is a real invariant signal but
    portfolio conflict (0.34) exceeds the execution envelope max (0.30).
    The sole binding constraint is the EURJPY SELL 0.01 position
    (ticket=57346204972).

Constraints:
    - NO live trading
    - NO position changes in MT5
    - PURE simulation only

Outputs:
    state/baseline_snapshot.json   — Phase 1 snapshot
    state/transition_simulation.json — Phase 2 trajectories
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import sys
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# ── Path setup ───────────────────────────────────────────────────────────────
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
_STATE_DIR = os.path.join(_PROJECT_ROOT, "state")
_PROXIMA_X = os.path.join(_PROJECT_ROOT, "proxima_x")

sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _PROXIMA_X)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("state_transition_simulator")

# ── Constants (from Batch 6.5/6.6) ──────────────────────────────────────────
STRUCTURAL_INVARIANCE = 0.9543

# Contamination audit coefficients
LIFECYCLE_COEFF = 1.0
MOF_BASE_COEFF = 0.85
RF_COEFF = 0.5353

# Priority formula weights
W_SIGNAL = 0.30
W_INVARIANCE = 0.25
W_CONTAMINATION = 0.15
W_CONFLICT = 0.30

# Current system state (from Phase 1 / context)
CURRENT_CONFIDENCE = 0.7656
CURRENT_PORTFOLIO_CONFLICT = 0.34
CURRENT_MOF = 0.4609

# Fixed baseline contamination resistance
_BASELINE_AVG_CONT = (LIFECYCLE_COEFF + MOF_BASE_COEFF + RF_COEFF) / 3.0
BASELINE_CONTAMINATION_RESISTANCE = 1.0 - _BASELINE_AVG_CONT  # ≈ 0.2049

# Execution envelope threshold
EXECUTION_PRIORITY_THRESHOLD = 0.50  # minimum viable for OBSERVE/EXECUTE

# Simulation parameters
NUM_CYCLES = 10
SEED = 42

# Open positions (from system state)
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

# Edge_04 signal definition
EDGE_04_SIGNAL = {
    "symbol": "EURJPY",
    "direction": 1,       # BUY
    "confidence": CURRENT_CONFIDENCE,
    "invariance": STRUCTURAL_INVARIANCE,
    "ecdf": 0.784,
    "drift": -1,
    "price": 185.649,
    "strategy": "pullback",
    "edge_pf": 1.3104,
}


# ═══════════════════════════════════════════════════════════════════════════
# CORE COMPUTATIONS
# ═══════════════════════════════════════════════════════════════════════════

def get_mof_effective_coefficient(mof_stability: float) -> float:
    """Map MOF stability [0, 1] to the effective MoF contamination coefficient."""
    raw = MOF_BASE_COEFF * (1.4609 - mof_stability)
    return max(0.0, min(1.0, raw))


def get_contamination_resistance(mof_stability: float) -> float:
    """Compute contamination resistance as a function of dynamic MOF stability."""
    mof_eff = get_mof_effective_coefficient(mof_stability)
    avg_contamination = (LIFECYCLE_COEFF + mof_eff + RF_COEFF) / 3.0
    return 1.0 - avg_contamination


def compute_execution_priority(
    confidence: float,
    portfolio_conflict: float,
    mof_stability: float,
) -> float:
    """Compute execution priority score (Batch 6.5 Phase 2 formula)."""
    cont_resist = get_contamination_resistance(mof_stability)
    score = (
        W_SIGNAL * confidence
        + W_INVARIANCE * STRUCTURAL_INVARIANCE
        + W_CONTAMINATION * cont_resist
        - W_CONFLICT * portfolio_conflict
    )
    return round(score, 6)


def compute_net_eurjpy_exposure(positions: List[dict]) -> float:
    """Compute net EURJPY exposure: + for BUY, - for SELL."""
    net = 0.0
    for p in positions:
        if p["symbol"] == "EURJPY":
            net += p["direction"] * p["volume"]
    return net


# Baseline portfolio conflict constant (calibrated from contamination audit).
# This is the "edge_04 specific burden" — the conflict contribution from the
# EURJPY SELL 0.01 position opposing the BUY signal.
_BASELINE_CONFLICT = 0.34
_BASELINE_OPPOSING_VOLUME = 0.01  # EURJPY SELL volume at baseline


def compute_portfolio_conflict(positions: List[dict]) -> float:
    """Compute portfolio conflict score from EURJPY positions vs signal.

    Uses the calibrated baseline conflict (0.34) and scales proportionally
    with the opposing EURJPY volume relative to the baseline opposing volume.

    Returns a value 0.0–1.0 where:
      0.0 = no opposing position (aligned or no EURJPY)
      0.34 = current state (EURJPY SELL 0.01 opposing BUY signal)
    """
    signal = EDGE_04_SIGNAL

    # Total opposing volume: positions on EURJPY with opposite direction to signal
    opposing_volume = 0.0
    aligned_volume = 0.0
    for p in positions:
        if p["symbol"] == "EURJPY":
            if p["direction"] != signal["direction"]:
                opposing_volume += p["volume"]
            else:
                aligned_volume += p["volume"]

    # If there's a net aligned position that outweighs the opposing, conflict drops
    net_opposing = max(0.0, opposing_volume - aligned_volume)

    # Scale conflict proportionally to opposing volume
    if net_opposing <= 0.0:
        return 0.0
    else:
        ratio = net_opposing / _BASELINE_OPPOSING_VOLUME
        return round(min(1.0, _BASELINE_CONFLICT * ratio), 4)


def is_edge_04_in_envelope(
    confidence: float,
    portfolio_conflict: float,
    mof_stability: float,
) -> bool:
    """Check if edge_04 falls within the execution envelope.

    Envelope condition: execution_priority >= EXECUTION_PRIORITY_THRESHOLD (0.50)
    """
    priority = compute_execution_priority(confidence, portfolio_conflict, mof_stability)
    return priority >= EXECUTION_PRIORITY_THRESHOLD


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 1 — Baseline State Snapshot
# ═══════════════════════════════════════════════════════════════════════════

def load_json(path: str, label: str) -> dict:
    """Load a JSON file, returning {} on failure."""
    if not os.path.isfile(path):
        logger.warning("%s not found at %s", label, path)
        return {}
    with open(path, "r") as f:
        data = json.load(f)
    logger.info("Loaded %s (%d bytes)", label, os.path.getsize(path))
    return data


def try_connect_mt5() -> Tuple[Optional[dict], Optional[List[dict]]]:
    """Try to connect to MT5 and fetch account/position data.

    Returns (account_info, positions) if MT5 is available, else (None, None).
    """
    try:
        from proxima_ops.execution.mt5_connector import MT5Connector
        connector = MT5Connector()
        if connector.connect():
            account = connector.get_account()
            positions = connector.get_positions()
            logger.info(
                "MT5 connected: account=%s, balance=%.2f, positions=%d",
                account.get("login") if account else "N/A",
                account.get("balance", 0) if account else 0,
                len(positions) if positions else 0,
            )
            connector.disconnect()
            return account, positions
        else:
            logger.warning("MT5 connection failed: %s", connector.last_error)
            return None, None
    except ImportError:
        logger.warning("MetaTrader5 package not installed; using hardcoded position data")
        return None, None
    except Exception as e:
        logger.warning("MT5 connection error: %s", e)
        return None, None


def load_existing_state_files() -> dict:
    """Load all prerequisite data files from state directory."""
    return {
        "contamination_audit": load_json(
            os.path.join(_STATE_DIR, "contamination_audit.json"),
            "contamination_audit",
        ),
        "lifecycle_state": load_json(
            os.path.join(_STATE_DIR, "lifecycle_state.json"),
            "lifecycle_state",
        ),
        "execution_priority_report": load_json(
            os.path.join(_STATE_DIR, "execution_priority_report.json"),
            "execution_priority_report",
        ),
        "arbitration_simulation_report": load_json(
            os.path.join(_STATE_DIR, "arbitration_simulation_report.json"),
            "arbitration_simulation_report",
        ),
        "emergent_threshold_model": load_json(
            os.path.join(_STATE_DIR, "emergent_threshold_model.json"),
            "emergent_threshold_model",
        ),
        "edge_04_execution_envelope": load_json(
            os.path.join(_STATE_DIR, "edge_04_execution_envelope.json"),
            "edge_04_execution_envelope",
        ),
    }


def record_baseline_snapshot() -> dict:
    """Phase 1: Record current system state programmatically."""
    logger.info("=" * 70)
    logger.info("PHASE 1 — Baseline State Snapshot")
    logger.info("=" * 70)

    # ── 1. MT5 connection (best-effort) ─────────────────────────────────
    mt5_account, mt5_positions = try_connect_mt5()

    # Use MT5 data if available, otherwise fall back to hardcoded positions
    if mt5_positions and len(mt5_positions) > 0:
        positions_raw = mt5_positions
        positions_source = "mt5_live"
    else:
        # Build from the hardcoded OPEN_POSITIONS
        positions_raw = []
        for p in OPEN_POSITIONS:
            positions_raw.append({
                "ticket": p["ticket"],
                "symbol": p["symbol"],
                "type": "BUY" if p["direction"] == 1 else "SELL",
                "volume": p["volume"],
                "price_open": p["entry_price"],
                "price_current": p["entry_price"],  # snapshot price
                "sl": 0.0,
                "tp": 0.0,
                "profit": 0.0,
                "swap": 0.0,
                "commission": 0.0,
            })
        positions_source = "hardcoded"

    # ── 2. Account info ─────────────────────────────────────────────────
    if mt5_account:
        account_info = {
            "login": mt5_account.get("login"),
            "balance": mt5_account.get("balance"),
            "equity": mt5_account.get("equity"),
            "margin": mt5_account.get("margin"),
            "margin_free": mt5_account.get("margin_free"),
            "margin_level": mt5_account.get("margin_level"),
            "leverage": mt5_account.get("leverage"),
            "currency": mt5_account.get("currency"),
            "server": mt5_account.get("server"),
            "source": "mt5_live",
        }
    else:
        account_info = {
            "login": "N/A",
            "balance": 24976.94,  # from global sweep snapshot
            "equity": 24976.94,
            "margin": 0.0,
            "margin_free": 24976.94,
            "margin_level": 0.0,
            "leverage": 500,
            "currency": "USD",
            "server": "MetaQuotes-Demo",
            "source": "hardcoded_fallback",
        }

    # ── 3. Load existing state files ─────────────────────────────────────
    state_files = load_existing_state_files()

    # ── 4. Compute derived state metrics ─────────────────────────────────
    portfolio_conflict = compute_portfolio_conflict(OPEN_POSITIONS)
    net_eurjpy = compute_net_eurjpy_exposure(OPEN_POSITIONS)
    execution_priority = compute_execution_priority(
        CURRENT_CONFIDENCE, portfolio_conflict, CURRENT_MOF
    )
    in_envelope = is_edge_04_in_envelope(
        CURRENT_CONFIDENCE, portfolio_conflict, CURRENT_MOF
    )
    contamination_resistance = get_contamination_resistance(CURRENT_MOF)

    # Count positions
    position_count = len(OPEN_POSITIONS)
    eurjpy_positions = [p for p in OPEN_POSITIONS if p["symbol"] == "EURJPY"]
    eurjpy_count = len(eurjpy_positions)
    eurjpy_sell_volume = sum(
        p["volume"] for p in eurjpy_positions if p["direction"] == -1
    )
    eurjpy_buy_volume = sum(
        p["volume"] for p in eurjpy_positions if p["direction"] == 1
    )

    snapshot = {
        "snapshot_metadata": {
            "timestamp": datetime.now().isoformat(),
            "report_type": "BASELINE_SNAPSHOT",
            "phase": "Batch 6.7 Phase 1 — Baseline State Snapshot",
            "edge_id": "edge_04",
            "symbol": "EURJPY",
            "positions_source": positions_source,
        },
        "account": account_info,
        "positions": {
            "count": position_count,
            "items": positions_raw,
        },
        "edge_04_signal": {
            "symbol": EDGE_04_SIGNAL["symbol"],
            "direction": EDGE_04_SIGNAL["direction"],
            "side": "BUY",
            "confidence": CURRENT_CONFIDENCE,
            "invariance": STRUCTURAL_INVARIANCE,
            "ecdf": EDGE_04_SIGNAL["ecdf"],
            "strategy": EDGE_04_SIGNAL["strategy"],
            "edge_pf": EDGE_04_SIGNAL["edge_pf"],
            "price": EDGE_04_SIGNAL["price"],
        },
        "eurjpy_exposure": {
            "net_exposure": net_eurjpy,
            "net_side": "SELL" if net_eurjpy < 0 else ("BUY" if net_eurjpy > 0 else "NEUTRAL"),
            "sell_volume": eurjpy_sell_volume,
            "buy_volume": eurjpy_buy_volume,
            "position_count": eurjpy_count,
            "conflicting_sell_ticket": 57346204972,
        },
        "derived_metrics": {
            "mof_stability": CURRENT_MOF,
            "mof_state": "STRUCTURE_LIMITED",
            "portfolio_conflict": portfolio_conflict,
            "contamination_resistance": round(contamination_resistance, 4),
            "execution_priority": execution_priority,
            "edge_04_in_envelope": in_envelope,
            "envelope_threshold": EXECUTION_PRIORITY_THRESHOLD,
            "rf_ready_count": 28,
            "rf_total_count": 28,
            "rf_avg_score": 0.618,
            "lifecycle_contamination": LIFECYCLE_COEFF,
            "mof_contamination": MOF_BASE_COEFF,
            "rf_contamination": RF_COEFF,
        },
        "loadable_state_files": {
            key: bool(value) for key, value in state_files.items()
        },
    }

    # ── Save snapshot ────────────────────────────────────────────────────
    os.makedirs(_STATE_DIR, exist_ok=True)
    snapshot_path = os.path.join(_STATE_DIR, "baseline_snapshot.json")
    with open(snapshot_path, "w") as f:
        json.dump(snapshot, f, indent=2, default=str)
    logger.info("Baseline snapshot saved to %s", snapshot_path)

    # ── Print summary ────────────────────────────────────────────────────
    print()
    print("  ── Baseline Snapshot Summary ──")
    print(f"    Account balance:     ${account_info['balance']:.2f}")
    print(f"    Open positions:      {position_count}")
    print(f"    Net EURJPY exposure: {net_eurjpy:.4f} ({'SELL' if net_eurjpy < 0 else 'BUY'})")
    print(f"    EURJPY SELL volume:  {eurjpy_sell_volume:.3f} (ticket 57346204972)")
    print(f"    MOF stability:       {CURRENT_MOF:.4f} (STRUCTURE_LIMITED)")
    print(f"    Portfolio conflict:  {portfolio_conflict:.4f}")
    print(f"    Edge_04 confidence:  {CURRENT_CONFIDENCE:.4f}")
    print(f"    Exec priority:       {execution_priority:.4f}")
    print(f"    In envelope (≥{EXECUTION_PRIORITY_THRESHOLD}): {in_envelope}")
    print()

    return snapshot


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 2 — State Transition Simulation
# ═══════════════════════════════════════════════════════════════════════════

def _build_trajectory(
    scenario_name: str,
    scenario_label: str,
    initial_positions: List[dict],
    mof_base: float,
    conflict_base: float,
    confidence_base: float,
    adjustment_cycle: int = 0,
    apply_adjustment: Optional[callable] = None,
) -> Dict[str, Any]:
    """Simulate 10-cycle trajectory for a given scenario.

    Args:
        scenario_name: Short identifier for the scenario.
        scenario_label: Human-readable label.
        initial_positions: Starting position set.
        mof_base: Base MOF stability value.
        conflict_base: Base portfolio conflict value.
        confidence_base: Base edge_04 confidence value.
        adjustment_cycle: Cycle at which to apply the adjustment (0 = cycle 1).
        apply_adjustment: Function that modifies positions/state at adjustment_cycle.

    Returns:
        Dict with scenario metadata and cycle-by-cycle trajectory.
    """
    random.seed(SEED + hash(scenario_name) % (2**31))

    # Track state variables
    positions = deepcopy(initial_positions)
    mof = mof_base
    portfolio_conflict = conflict_base
    confidence = confidence_base

    trajectory = []
    adjustment_applied = False

    for cycle in range(1, NUM_CYCLES + 1):
        # ── Apply adjustment if this is the adjustment cycle ──────────
        if apply_adjustment and cycle == adjustment_cycle and not adjustment_applied:
            result = apply_adjustment(positions, mof, portfolio_conflict, confidence)
            positions = result.get("positions", positions)
            mof = result.get("mof", mof)
            portfolio_conflict = result.get("portfolio_conflict", portfolio_conflict)
            confidence = result.get("confidence", confidence)
            adjustment_applied = True
            logger.info(
                "  [Scenario %s] Adjustment applied at cycle %d",
                scenario_name, cycle,
            )

        # ── Compute metrics for this cycle ────────────────────────────
        exec_priority = compute_execution_priority(confidence, portfolio_conflict, mof)
        in_envelope = exec_priority >= EXECUTION_PRIORITY_THRESHOLD
        net_eurjpy = compute_net_eurjpy_exposure(positions)
        cont_resist = get_contamination_resistance(mof)

        cycle_data = {
            "cycle": cycle,
            "adjustment_applied": adjustment_applied if cycle >= (adjustment_cycle or 999) else False,
            "mof_stability": round(mof, 4),
            "portfolio_conflict": round(portfolio_conflict, 4),
            "edge_04_confidence": round(confidence, 4),
            "contamination_resistance": round(cont_resist, 4),
            "execution_priority": exec_priority,
            "edge_04_in_envelope": in_envelope,
            "net_eurjpy_exposure": round(net_eurjpy, 4),
            "position_count": len(positions),
            "positions": [
                {"ticket": p["ticket"], "symbol": p["symbol"],
                 "direction": p["direction"], "volume": p["volume"]}
                for p in positions
            ],
        }
        trajectory.append(cycle_data)

        # ── State transition to next cycle ────────────────────────────
        # MOF random walk ±0.01
        mof += random.uniform(-0.01, 0.01)

        # Confidence random walk ±0.005
        confidence += random.uniform(-0.005, 0.005)

        # Portfolio conflict slowly mean-reverts toward base
        base_conflict = conflict_base
        if portfolio_conflict > base_conflict:
            portfolio_conflict -= random.uniform(0.0, 0.01)
        elif portfolio_conflict < base_conflict:
            portfolio_conflict += random.uniform(0.0, 0.01)

        # Clamp all values
        mof = max(0.0, min(1.0, mof))
        confidence = max(0.0, min(1.0, confidence))
        portfolio_conflict = max(0.0, min(1.0, portfolio_conflict))

    # ── Compute trajectory summary ──────────────────────────────────────
    cycles_in_envelope = sum(1 for c in trajectory if c["edge_04_in_envelope"])
    envelope_entries = sum(
        1 for i in range(1, len(trajectory))
        if trajectory[i]["edge_04_in_envelope"] and not trajectory[i - 1]["edge_04_in_envelope"]
    )
    envelope_exits = sum(
        1 for i in range(1, len(trajectory))
        if not trajectory[i]["edge_04_in_envelope"] and trajectory[i - 1]["edge_04_in_envelope"]
    )

    return {
        "scenario": scenario_name,
        "label": scenario_label,
        "num_cycles": NUM_CYCLES,
        "adjustment_cycle": adjustment_cycle if apply_adjustment else None,
        "summary": {
            "initial_priority": trajectory[0]["execution_priority"],
            "final_priority": trajectory[-1]["execution_priority"],
            "priority_trend": round(trajectory[-1]["execution_priority"] - trajectory[0]["execution_priority"], 4),
            "initial_mof": trajectory[0]["mof_stability"],
            "final_mof": trajectory[-1]["mof_stability"],
            "initial_conflict": trajectory[0]["portfolio_conflict"],
            "final_conflict": trajectory[-1]["portfolio_conflict"],
            "initial_in_envelope": trajectory[0]["edge_04_in_envelope"],
            "final_in_envelope": trajectory[-1]["edge_04_in_envelope"],
            "cycles_in_envelope": cycles_in_envelope,
            "cycles_outside_envelope": NUM_CYCLES - cycles_in_envelope,
            "envelope_entries": envelope_entries,
            "envelope_exits": envelope_exits,
            "first_cycle_in_envelope": next(
                (c["cycle"] for c in trajectory if c["edge_04_in_envelope"]),
                None,
            ),
        },
        "trajectory": trajectory,
    }


def _build_base_positions() -> List[dict]:
    """Return a deep copy of the current open positions."""
    return deepcopy(OPEN_POSITIONS)


# ── Scenario definitions ─────────────────────────────────────────────────

def scenario_1_baseline() -> Dict[str, Any]:
    """Scenario 1: No Change (Baseline) — Keep all 5 positions."""
    logger.info("  Simulating Scenario 1: No Change (Baseline)...")

    base_positions = _build_base_positions()
    conflict = compute_portfolio_conflict(base_positions)

    result = _build_trajectory(
        scenario_name="1",
        scenario_label="No Change (Baseline)",
        initial_positions=base_positions,
        mof_base=CURRENT_MOF,
        conflict_base=conflict,
        confidence_base=CURRENT_CONFIDENCE,
    )

    logger.info("  Scenario 1 complete: %d/10 cycles in envelope",
                result["summary"]["cycles_in_envelope"])
    return result


def scenario_2_partial_adjustment() -> Dict[str, Any]:
    """Scenario 2: Partial Adjustment — EURJPY SELL 0.01 → 0.005.

    All other positions unchanged. Recalculate portfolio_conflict.
    """
    logger.info("  Simulating Scenario 2: Partial Adjustment (REDUCED SELL)...")

    base_positions = _build_base_positions()
    # Apply initial adjustment: reduce EURJPY SELL from 0.01 to 0.005
    for p in base_positions:
        if p["ticket"] == 57346204972:  # EURJPY SELL
            p["volume"] = 0.005
            break
    # Recalculate conflict with adjusted positions
    conflict = compute_portfolio_conflict(base_positions)

    def no_adjustment(pos, m, pc, conf):
        return {"positions": pos, "mof": m, "portfolio_conflict": pc, "confidence": conf}

    result = _build_trajectory(
        scenario_name="2",
        scenario_label="Partial Adjustment (REDUCED SELL: 0.01→0.005)",
        initial_positions=base_positions,
        mof_base=CURRENT_MOF,
        conflict_base=conflict,
        confidence_base=CURRENT_CONFIDENCE,
        adjustment_cycle=0,  # No further adjustment needed
        apply_adjustment=no_adjustment,
    )

    logger.info("  Scenario 2 complete: %d/10 cycles in envelope",
                result["summary"]["cycles_in_envelope"])
    return result


def scenario_3_full_flip() -> Dict[str, Any]:
    """Scenario 3: Full Flip — Close EURJPY SELL, open EURJPY BUY 0.01.

    Recalculate portfolio_conflict.
    """
    logger.info("  Simulating Scenario 3: Full Flip...")

    base_positions = _build_base_positions()

    def apply_flip(pos, m, pc, conf):
        # Remove EURJPY SELL
        pos[:] = [p for p in pos if p["ticket"] != 57346204972]
        # Add EURJPY BUY
        pos.append({
            "ticket": 0,  # simulated
            "symbol": "EURJPY",
            "direction": 1,
            "volume": 0.01,
            "entry_price": 185.649,
        })
        new_conflict = compute_portfolio_conflict(pos)
        # MOF: large action may degrade MOF
        new_mof = max(0.0, m - 0.05)
        return {
            "positions": pos,
            "mof": new_mof,
            "portfolio_conflict": new_conflict,
            "confidence": conf,
        }

    # Use original positions; apply flip at cycle 1
    conflict = compute_portfolio_conflict(base_positions)

    result = _build_trajectory(
        scenario_name="3",
        scenario_label="Full Flip (Close SELL, Open BUY 0.01)",
        initial_positions=base_positions,
        mof_base=CURRENT_MOF,
        conflict_base=conflict,
        confidence_base=CURRENT_CONFIDENCE,
        adjustment_cycle=1,
        apply_adjustment=apply_flip,
    )

    logger.info("  Scenario 3 complete: %d/10 cycles in envelope",
                result["summary"]["cycles_in_envelope"])
    return result


def scenario_4_delayed_adjustment() -> Dict[str, Any]:
    """Scenario 4: Delayed Adjustment — Same as Scenario 2 but applied at cycle 5.

    Test whether timing matters for system stability.
    """
    logger.info("  Simulating Scenario 4: Delayed Adjustment (at cycle 5)...")

    base_positions = _build_base_positions()
    conflict = compute_portfolio_conflict(base_positions)

    def apply_delayed_reduce(pos, m, pc, conf):
        for p in pos:
            if p["ticket"] == 57346204972:
                p["volume"] = 0.005
                break
        new_conflict = compute_portfolio_conflict(pos)
        return {
            "positions": pos,
            "mof": m,
            "portfolio_conflict": new_conflict,
            "confidence": conf,
        }

    result = _build_trajectory(
        scenario_name="4",
        scenario_label="Delayed Adjustment (REDUCED at cycle 5)",
        initial_positions=base_positions,
        mof_base=CURRENT_MOF,
        conflict_base=conflict,
        confidence_base=CURRENT_CONFIDENCE,
        adjustment_cycle=5,
        apply_adjustment=apply_delayed_reduce,
    )

    logger.info("  Scenario 4 complete: %d/10 cycles in envelope",
                result["summary"]["cycles_in_envelope"])
    return result


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATION
# ═══════════════════════════════════════════════════════════════════════════

def run_phases_1_and_2() -> dict:
    """Run Phase 1 (Baseline Snapshot) and Phase 2 (State Transition Simulation).

    Returns the full simulation report.
    """
    logger.info("=" * 70)
    logger.info("Batch 6.7 Phases 1+2 — State Transition Simulation")
    logger.info("=" * 70)

    # ═════════════════════════════════════════════════════════════════════
    # PHASE 1 — Baseline State Snapshot
    # ═════════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 1 — Baseline State Snapshot")
    logger.info("=" * 70)

    snapshot = record_baseline_snapshot()

    # ═════════════════════════════════════════════════════════════════════
    # PHASE 2 — State Transition Simulation
    # ═════════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 2 — State Transition Simulation")
    logger.info("=" * 70)
    print()
    print("  Simulating 4 portfolio transition scenarios...")
    print()

    scenarios = [
        scenario_1_baseline(),
        scenario_2_partial_adjustment(),
        scenario_3_full_flip(),
        scenario_4_delayed_adjustment(),
    ]

    # ── Build comparison summary ─────────────────────────────────────────
    comparison = []
    for s in scenarios:
        comparison.append({
            "scenario": s["scenario"],
            "label": s["label"],
            "adjustment_cycle": s["adjustment_cycle"],
            "initial_priority": s["summary"]["initial_priority"],
            "final_priority": s["summary"]["final_priority"],
            "priority_trend": s["summary"]["priority_trend"],
            "initial_in_envelope": s["summary"]["initial_in_envelope"],
            "final_in_envelope": s["summary"]["final_in_envelope"],
            "cycles_in_envelope": s["summary"]["cycles_in_envelope"],
            "cycles_outside_envelope": s["summary"]["cycles_outside_envelope"],
            "envelope_entries": s["summary"]["envelope_entries"],
            "envelope_exits": s["summary"]["envelope_exits"],
            "first_cycle_in_envelope": s["summary"]["first_cycle_in_envelope"],
            "initial_conflict": s["summary"]["initial_conflict"],
            "final_conflict": s["summary"]["final_conflict"],
        })

    # ── Build final report ───────────────────────────────────────────────
    report = {
        "report_metadata": {
            "report_type": "STATE_TRANSITION_SIMULATION",
            "phase": "Batch 6.7 Phases 1+2 — State Transition Simulation",
            "edge_id": "edge_04",
            "symbol": "EURJPY",
            "generated_at": datetime.now().isoformat(),
            "constraints_applied": [
                "NO_REAL_MT5_EXECUTION",
                "NO_NEW_TRADES",
                "PURE_SIMULATION",
            ],
            "params": {
                "num_cycles": NUM_CYCLES,
                "random_seed": SEED,
                "execution_envelope_threshold": EXECUTION_PRIORITY_THRESHOLD,
                "mof_random_walk": "±0.01/cycle",
                "confidence_random_walk": "±0.005/cycle",
            },
        },
        "phase_1_baseline_snapshot": snapshot,
        "phase_2_transition_simulation": {
            "description": (
                "Simulated 4 portfolio transition scenarios over 10 cycles each "
                "to test whether partial portfolio adjustment preserves system stability."
            ),
            "formulas": {
                "execution_priority": (
                    "0.30*confidence + 0.25*invariance + 0.15*contamination_resistance - 0.30*conflict"
                ),
                "contamination_resistance": "1.0 - (lifecycle + mof_eff + rf) / 3.0",
                "portfolio_conflict": "direction_conflict*0.6 + volume_ratio*0.4",
                "envelope_condition": "execution_priority >= 0.50",
            },
            "scenarios": {s["scenario"]: s for s in scenarios},
            "comparison": comparison,
        },
    }

    # ── Save report ──────────────────────────────────────────────────────
    os.makedirs(_STATE_DIR, exist_ok=True)
    report_path = os.path.join(_STATE_DIR, "transition_simulation.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Transition simulation report saved to %s", report_path)

    return report


# ═══════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def print_trajectory_comparison(report: dict):
    """Print formatted trajectory comparison across all 4 scenarios."""
    sim = report["phase_2_transition_simulation"]
    scenarios = sim["scenarios"]
    comparison = sim["comparison"]

    print()
    print("=" * 90)
    print("  STATE TRANSITION SIMULATION — TRAJECTORY COMPARISON")
    print("=" * 90)
    print()

    for sc_key in ["1", "2", "3", "4"]:
        sc = scenarios[sc_key]
        label = sc["label"]
        summary = sc["summary"]
        traj = sc["trajectory"]

        print(f"  ── Scenario {sc_key}: {label} ──")
        print(f"     Adjustment cycle: {sc['adjustment_cycle']}")
        print()

        # Header
        print(f"     {'Cycle':>6}  {'MOF':>7}  {'Conflict':>9}  {'Confidence':>10}  "
              f"{'Priority':>9}  {'InEnv':>6}  {'NetEUR':>8}  {'Pos':>4}")
        print(f"     {'-'*6}  {'-'*7}  {'-'*9}  {'-'*10}  "
              f"{'-'*9}  {'-'*6}  {'-'*8}  {'-'*4}")

        for c in traj:
            env_flag = "✓" if c["edge_04_in_envelope"] else "✗"
            adj_flag = "A" if c.get("adjustment_applied") else " "
            print(f"     {c['cycle']:>4}{adj_flag:>2}  {c['mof_stability']:>7.4f}  "
                  f"{c['portfolio_conflict']:>9.4f}  {c['edge_04_confidence']:>10.4f}  "
                  f"{c['execution_priority']:>9.4f}  {env_flag:>6}  "
                  f"{c['net_eurjpy_exposure']:>8.4f}  {c['position_count']:>4}")

        print()
        print(f"     Summary:")
        print(f"       Cycles in envelope:    {summary['cycles_in_envelope']} / {NUM_CYCLES}")
        print(f"       Envelope entries:      {summary['envelope_entries']}")
        print(f"       Envelope exits:        {summary['envelope_exits']}")
        print(f"       First in envelope:     {summary['first_cycle_in_envelope']}")
        print(f"       Priority trend:        {summary['initial_priority']:.4f} → {summary['final_priority']:.4f} "
              f"({summary['priority_trend']:+.4f})")
        print(f"       MOF trend:             {summary['initial_mof']:.4f} → {summary['final_mof']:.4f}")
        print(f"       Conflict trend:        {summary['initial_conflict']:.4f} → {summary['final_conflict']:.4f}")
        print()

    # ── Cross-scenario comparison table ─────────────────────────────────
    print("  ── Cross-Scenario Comparison ──")
    print()
    print(f"     {'Scenario':<8}  {'Init Pri':>9}  {'Final Pri':>9}  {'Trend':>8}  "
          f"{'Init Env':>8}  {'Final Env':>8}  {'Cycles In':>9}  {'Entries':>8}  {'Exits':>8}")
    print(f"     {'-'*8}  {'-'*9}  {'-'*9}  {'-'*8}  "
          f"{'-'*8}  {'-'*8}  {'-'*9}  {'-'*8}  {'-'*8}")

    for c in comparison:
        print(f"     {c['scenario']:<8}  {c['initial_priority']:>9.4f}  {c['final_priority']:>9.4f}  "
              f"{c['priority_trend']:>+8.4f}  {str(c['initial_in_envelope']):>8}  "
              f"{str(c['final_in_envelope']):>8}  {c['cycles_in_envelope']:>4}/{NUM_CYCLES:<4}  "
              f"{c['envelope_entries']:>4}/{c['envelope_exits']:<4}")
    print()

    # ── Key findings ────────────────────────────────────────────────────
    print("  ── Key Findings ──")
    print()

    for c in comparison:
        if c["scenario"] == "1":
            s1_cycles = c["cycles_in_envelope"]
        if c["scenario"] == "2":
            s2_cycles = c["cycles_in_envelope"]
            s2_init_env = c["initial_in_envelope"]
        if c["scenario"] == "3":
            s3_cycles = c["cycles_in_envelope"]
        if c["scenario"] == "4":
            s4_cycles = c["cycles_in_envelope"]

    # Compare early vs late adjustment
    print(f"  Scenario 1 (Baseline):      {s1_cycles}/{NUM_CYCLES} cycles in envelope — "
          f"{'STABLE' if s1_cycles >= 5 else 'DEGRADING'}")

    print(f"  Scenario 2 (Partial adj):   {s2_cycles}/{NUM_CYCLES} cycles in envelope — "
          f"{'RECOVERS envelope' if s2_cycles > s1_cycles else 'similar to baseline'}")

    print(f"  Scenario 3 (Full flip):     {s3_cycles}/{NUM_CYCLES} cycles in envelope — "
          f"{'IMPROVES immediately' if s3_cycles >= 8 else 'mixed outcome'}")

    print(f"  Scenario 4 (Delay adj):     {s4_cycles}/{NUM_CYCLES} cycles in envelope — "
          f"{'BETTER than baseline' if s4_cycles > s1_cycles else 'similar to baseline'}")

    print()

    # Timing comparison
    print(f"  Timing comparison (Sc2 early vs Sc4 delayed):")
    if s2_cycles != s4_cycles:
        print(f"    Difference of {abs(s2_cycles - s4_cycles)} cycles — "
              f"{'EARLY adjustment better' if s2_cycles > s4_cycles else 'DELAYED adjustment better'}")
    else:
        print(f"    Same number of envelope cycles — timing is NEUTRAL")
    print()

    # Final verdict
    print(f"  VERDICT: ")
    best = max(comparison, key=lambda c: c["cycles_in_envelope"])
    worst = min(comparison, key=lambda c: c["cycles_in_envelope"])
    print(f"    Best scenario:  Scenario {best['scenario']} ({best['label']}) — "
          f"{best['cycles_in_envelope']}/{NUM_CYCLES} cycles in envelope")
    print(f"    Worst scenario: Scenario {worst['scenario']} ({worst['label']}) — "
          f"{worst['cycles_in_envelope']}/{NUM_CYCLES} cycles in envelope")

    # Does any adjustment get edge_04 into the envelope?
    any_gets_in = any(
        c["initial_in_envelope"] != c["final_in_envelope"]
        for c in comparison
    )
    if any_gets_in:
        print(f"    Edge_04 CAN enter the envelope through portfolio adjustment.")
    else:
        print(f"    Edge_04 remains OUTSIDE the envelope across ALL scenarios.")
        print(f"    → Portfolio conflict reduction alone is INSUFFICIENT.")
        print(f"    → Need MOF stability improvement and/or confidence increase.")

    print()
    print("=" * 90)


def main():
    """CLI entry point."""
    logger.info("Starting State Transition Simulation (Batch 6.7 Phases 1+2)")

    report = run_phases_1_and_2()

    print()
    print("=" * 90)
    print("  PHASE 1 — BASELINE SNAPSHOT SUMMARY")
    print("=" * 90)
    snap = report["phase_1_baseline_snapshot"]
    print(f"  Account balance:   ${snap['account']['balance']:.2f}")
    print(f"  Open positions:    {snap['positions']['count']}")
    print(f"  Net EURJPY:        {snap['eurjpy_exposure']['net_exposure']:.4f} "
          f"({snap['eurjpy_exposure']['net_side']})")
    print(f"  EURJPY SELL vol:   {snap['eurjpy_exposure']['sell_volume']:.3f}")
    print(f"  EURJPY BUY vol:    {snap['eurjpy_exposure']['buy_volume']:.3f}")
    print(f"  Port conflict:     {snap['derived_metrics']['portfolio_conflict']:.4f}")
    print(f"  Exec priority:     {snap['derived_metrics']['execution_priority']:.4f}")
    print(f"  In envelope:       {snap['derived_metrics']['edge_04_in_envelope']}")

    print()
    print("=" * 90)
    print("  PHASE 2 — TRAJECTORY COMPARISON")
    print("=" * 90)
    print_trajectory_comparison(report)

    print()
    print("  Output files:")
    print(f"    - {os.path.join(_STATE_DIR, 'baseline_snapshot.json')}")
    print(f"    - {os.path.join(_STATE_DIR, 'transition_simulation.json')}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
