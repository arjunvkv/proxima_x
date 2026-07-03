#!/usr/bin/env python3
"""
Batch 6.6 Phases 3+4 — Hysteresis Stability Test + Edge_04 Execution Envelope

Phase 3: Simulate repeated decision cycles (20 cycles × 3 scenarios) to test
         whether the system is decision-stable or decision-unstable.
Phase 4: Compute a 3D execution envelope bounding box and compare current
         edge_04 state against it.

Constraints: NO live trading, NO position changes, PURE simulation only.
"""

import json
import math
import os
import random
import sys
from datetime import datetime

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE_DIR = os.path.join(os.path.dirname(BASE_DIR), "state")
os.makedirs(STATE_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(STATE_DIR, "edge_04_execution_envelope.json")

# ── Fixed system parameters (from Batch 6.5/6.6) ───────────────────────────
STRUCTURAL_INVARIANCE = 0.9543

# Contamination audit coefficients (fixed layers)
LIFECYCLE_COEFF = 1.0
MOF_BASE_COEFF = 0.85
RF_COEFF = 0.5353

# Priority formula weights (from Batch 6.5 Phase 2)
W_SIGNAL = 0.30
W_INVARIANCE = 0.25
W_CONTAMINATION = 0.15
W_CONFLICT = 0.30

# ── Current system state ───────────────────────────────────────────────────
CURRENT_CONFIDENCE = 0.7656
CURRENT_CONFLICT = 0.34
CURRENT_MOF = 0.4609

# ── Thresholds ─────────────────────────────────────────────────────────────
EXECUTION_PRIORITY_THRESHOLD = 0.50  # minimum viable for OBSERVE/EXECUTE
NUM_CYCLES = 20
SEED = 42

# Baseline contamination resistance at current state (fixed reference)
_BASELINE_MOF_EFF = MOF_BASE_COEFF  # 0.85
_BASELINE_AVG_CONT = (LIFECYCLE_COEFF + _BASELINE_MOF_EFF + RF_COEFF) / 3.0
BASELINE_CONTAMINATION_RESISTANCE = 1.0 - _BASELINE_AVG_CONT  # 0.2049


# ═══════════════════════════════════════════════════════════════════════════
# CORE COMPUTATIONS
# ═══════════════════════════════════════════════════════════════════════════

def get_mof_effective_coefficient(mof_stability):
    """
    Map MOF stability [0, 1] to the effective MoF contamination coefficient.

    Higher MOF stability = better data quality = less contamination.
    The current state (mof=0.4609) maps to the audit baseline of 0.85.
    As mof_stability → 1.0, the coefficient drops toward ~0.39 (less contamination).
    As mof_stability → 0.0, the coefficient rises toward 1.0  (max contamination).
    """
    # Linear model calibrated to current state:
    # mof_eff = MOF_BASE_COEFF * (1.4609 - mof_stability)
    # At mof=0.4609 → 0.85; at mof=1.0 → 0.85*0.4609≈0.39; at mof=0.0 → 1.24→clamped to 1.0
    raw = MOF_BASE_COEFF * (1.4609 - mof_stability)
    return max(0.0, min(1.0, raw))


def get_contamination_resistance(mof_stability):
    """Compute contamination resistance as a function of dynamic MOF stability."""
    mof_eff = get_mof_effective_coefficient(mof_stability)
    avg_contamination = (LIFECYCLE_COEFF + mof_eff + RF_COEFF) / 3.0
    return 1.0 - avg_contamination


def compute_execution_priority(confidence, portfolio_conflict, mof_stability):
    """
    Compute execution priority score using the dynamic version of the
    Batch 6.5 Phase 2 formula where contamination resistance varies with MOF.
    """
    cont_resist = get_contamination_resistance(mof_stability)
    score = (
        W_SIGNAL * confidence
        + W_INVARIANCE * STRUCTURAL_INVARIANCE
        + W_CONTAMINATION * cont_resist
        - W_CONFLICT * portfolio_conflict
    )
    return round(score, 6)


def decide(priority, threshold=EXECUTION_PRIORITY_THRESHOLD):
    """Return EXECUTE if priority >= threshold, else HOLD."""
    return "EXECUTE" if priority >= threshold else "HOLD"


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 3 — HYSTERESIS STABILITY TEST
# ═══════════════════════════════════════════════════════════════════════════

def simulate_scenario(scenario_type, cycles=NUM_CYCLES):
    """
    Simulate hysteresis over `cycles` decision cycles.

    scenario_type:
        'hold_all'     — always HOLD (no execution ever)
        'single_at_5'  — HOLD except a single EXECUTE at cycle 5
        'repeated'     — threshold-based decision every cycle (natural)
    """
    random.seed(SEED)
    conflict = CURRENT_CONFLICT
    mof = CURRENT_MOF
    history = []
    last_exec_cycle = -999  # far in the past

    for c in range(1, cycles + 1):
        priority = compute_execution_priority(CURRENT_CONFIDENCE, conflict, mof)
        decision = decide(priority)

        # ── Scenario override ──────────────────────────────────────────
        if scenario_type == 'hold_all':
            decision = 'HOLD'
        elif scenario_type == 'single_at_5':
            decision = 'EXECUTE' if c == 5 else 'HOLD'
        # 'repeated' uses the threshold-based decision as-is

        # Record cycle snapshot
        history.append({
            "cycle": c,
            "conflict": round(conflict, 4),
            "mof_stability": round(mof, 4),
            "contamination_resistance": round(get_contamination_resistance(mof), 4),
            "execution_priority": priority,
            "decision": decision,
        })

        # ── State transition ───────────────────────────────────────────
        if decision == "EXECUTE":
            # Overload penalty if repeated execution within 3 cycles
            if c - last_exec_cycle <= 3:
                mof -= 0.10
            mof += 0.05          # more data → better stability
            conflict += 0.10     # new position adds complexity
            last_exec_cycle = c
        else:  # HOLD
            mof += random.uniform(-0.01, 0.01)  # small random walk drift
            # conflict unchanged on HOLD

        # Clamp to valid ranges
        mof = max(0.0, min(1.0, mof))
        conflict = max(0.0, min(1.0, conflict))

    return history


def analyze_hysteresis(history):
    """Quantify oscillation, convergence, and stability metrics."""
    decisions = [h["decision"] for h in history]
    mofs = [h["mof_stability"] for h in history]
    priorities = [h["execution_priority"] for h in history]

    # Oscillation: count flips between HOLD and EXECUTE
    flips = sum(1 for i in range(1, len(decisions)) if decisions[i] != decisions[i - 1])
    oscillation_freq = round(flips / len(history), 4)

    # Convergence: last 5 cycles all same decision?
    last5 = decisions[-5:]
    converged = len(set(last5)) == 1

    # MOF volatility: average absolute change per cycle
    mof_vol = (
        sum(abs(mofs[i] - mofs[i - 1]) for i in range(1, len(mofs)))
        / len(mofs)
    )

    # Priority volatility
    pri_vol = (
        sum(abs(priorities[i] - priorities[i - 1]) for i in range(1, len(priorities)))
        / len(priorities)
    )

    # Count EXECUTE decisions
    execute_count = sum(1 for d in decisions if d == "EXECUTE")

    return {
        "total_cycles": len(history),
        "execute_count": execute_count,
        "hold_count": len(history) - execute_count,
        "flip_count": flips,
        "oscillation_frequency": oscillation_freq,
        "converged": converged,
        "steady_state_decision": last5[-1] if converged else "OSCILLATING",
        "last_5_decisions": last5,
        "mof_start": round(mofs[0], 4),
        "mof_end": round(mofs[-1], 4),
        "mof_trend": round(mofs[-1] - mofs[0], 4),
        "mof_volatility": round(mof_vol, 6),
        "priority_start": round(priorities[0], 4),
        "priority_end": round(priorities[-1], 4),
        "priority_trend": round(priorities[-1] - priorities[0], 4),
        "priority_volatility": round(pri_vol, 6),
    }


def classify_hysteresis(analyses):
    """
    Based on the three scenarios, produce an overall verdict:
    CONVERGENT, OSCILLATORY, or BORDERLINE.
    """
    rep = analyses["repeated"]

    if rep["converged"]:
        return (
            "CONVERGENT — System settles into a steady state "
            "under repeated activation; no decision-loop instability detected."
        )
    elif rep["flip_count"] >= 4:
        return (
            f"OSCILLATORY — System flips {rep['flip_count']} times in "
            f"{rep['total_cycles']} cycles; decision-loop is unstable "
            "and may cause thrashing between HOLD and EXECUTE."
        )
    else:
        return (
            f"BORDERLINE — System shows mild oscillation "
            f"({rep['flip_count']} flips); may need damping or "
            "a dead-band to prevent repeated flip-flopping."
        )


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 4 — EXECUTION ENVELOPE
# ═══════════════════════════════════════════════════════════════════════════

def find_execution_envelope():
    """
    Compute the 3D execution envelope bounding box.

    The envelope is: [conf_min, conf_max] × [conflict_min, conflict_max] × [mof_min, mof_max]
    where execution_priority >= EXECUTION_PRIORITY_THRESHOLD (0.50).

    Returns the bounding box plus the count of viable grid points.
    """
    # Sweep over reasonable ranges (81×41×41 ≈ 136k points — fast)
    conf_range = [round(0.60 + i * (1.00 - 0.60) / 40, 4) for i in range(41)]  # 0.60-1.00
    conflict_range = [round(0.00 + i * (1.00 - 0.00) / 40, 4) for i in range(41)]  # 0.00-1.00
    mof_range = [round(0.00 + i * (1.00 - 0.00) / 40, 4) for i in range(41)]  # 0.00-1.00

    viable = []

    for conf in conf_range:
        for conflict in conflict_range:
            # Inner loop: scan MOF — break early when priority drops below threshold
            for mof in mof_range:
                priority = compute_execution_priority(conf, conflict, mof)
                if priority >= EXECUTION_PRIORITY_THRESHOLD:
                    viable.append({
                        "confidence": conf,
                        "portfolio_conflict": conflict,
                        "mof_stability": mof,
                        "execution_priority": round(priority, 6),
                    })

    if not viable:
        return {
            "conf_min": None,
            "conf_max": None,
            "conflict_min": None,
            "conflict_max": None,
            "mof_min": None,
            "mof_max": None,
            "num_viable_points": 0,
            "note": "No viable execution points found at threshold >= 0.50",
        }

    confs = [p["confidence"] for p in viable]
    conflicts = [p["portfolio_conflict"] for p in viable]
    mofs = [p["mof_stability"] for p in viable]

    # Bounding box from the viable set (projection)
    envelope = {
        "conf_min": round(min(confs), 4),
        "conf_max": round(max(confs), 4),
        "conflict_min": round(min(conflicts), 4),
        "conflict_max": round(max(conflicts), 4),
        "mof_min": round(min(mofs), 4),
        "mof_max": round(max(mofs), 4),
        "num_viable_points": len(viable),
        "total_grid_points": len(conf_range) * len(conflict_range) * len(mof_range),
        "threshold_applied": EXECUTION_PRIORITY_THRESHOLD,
    }

    return envelope, viable


def check_state_in_envelope(envelope):
    """Return detailed check of current state against the envelope."""
    if envelope.get("conf_min") is None:
        return False, {
            "inside_envelope": False,
            "reason": "No viable envelope defined (zero viable points)",
        }

    in_conf = envelope["conf_min"] <= CURRENT_CONFIDENCE <= envelope["conf_max"]
    in_conflict = (
        envelope["conflict_min"] <= CURRENT_CONFLICT <= envelope["conflict_max"]
    )
    in_mof = envelope["mof_min"] <= CURRENT_MOF <= envelope["mof_max"]

    inside = in_conf and in_conflict and in_mof

    details = {
        "current_confidence": CURRENT_CONFIDENCE,
        "confidence_range": [envelope["conf_min"], envelope["conf_max"]],
        "confidence_in_range": in_conf,
        "current_portfolio_conflict": CURRENT_CONFLICT,
        "conflict_range": [envelope["conflict_min"], envelope["conflict_max"]],
        "conflict_in_range": in_conflict,
        "current_mof_stability": CURRENT_MOF,
        "mof_range": [envelope["mof_min"], envelope["mof_max"]],
        "mof_in_range": in_mof,
        "inside_envelope": inside,
    }

    return inside, details


# ═══════════════════════════════════════════════════════════════════════════
# SAVE RESULTS
# ═══════════════════════════════════════════════════════════════════════════

def save_report(hysteresis_results, envelope, envelope_check_details, verdict):
    """Assemble and write the final JSON report."""
    payload = {
        "report_metadata": {
            "report_type": "HYSTERESIS_AND_ENVELOPE_REPORT",
            "phase": (
                "Batch 6.6 Phases 3+4 — Hysteresis Stability Test "
                "+ Edge_04 Execution Envelope"
            ),
            "edge_id": "edge_04",
            "symbol": "EURJPY",
            "generated_at": datetime.now().isoformat(),
            "constraints": [
                "NO_REAL_MT5_EXECUTION",
                "NO_NEW_TRADES",
                "PURE_SIMULATION",
            ],
            "current_system_state": {
                "confidence": CURRENT_CONFIDENCE,
                "portfolio_conflict": CURRENT_CONFLICT,
                "mof_stability": CURRENT_MOF,
                "execution_priority": compute_execution_priority(
                    CURRENT_CONFIDENCE, CURRENT_CONFLICT, CURRENT_MOF
                ),
                "contamination_resistance_baseline": BASELINE_CONTAMINATION_RESISTANCE,
            },
            "fixed_parameters": {
                "structural_invariance": STRUCTURAL_INVARIANCE,
                "contamination_audit_coefficients": {
                    "lifecycle": LIFECYCLE_COEFF,
                    "mof_base": MOF_BASE_COEFF,
                    "rf": RF_COEFF,
                },
                "weights": {
                    "signal": W_SIGNAL,
                    "invariance": W_INVARIANCE,
                    "contamination": W_CONTAMINATION,
                    "conflict": W_CONFLICT,
                },
                "execution_priority_threshold": EXECUTION_PRIORITY_THRESHOLD,
            },
        },
        "phase_3_hysteresis_test": hysteresis_results,
        "phase_4_execution_envelope": {
            "description": (
                "3D bounding box [conf_min,conf_max] × [conflict_min,conflict_max] "
                "× [mof_min,mof_max] where execution_priority >= 0.50"
            ),
            "envelope": envelope,
            "current_state_check": envelope_check_details,
        },
        "verdict": verdict,
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(payload, f, indent=2)

    return OUTPUT_FILE


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  Batch 6.6 Phases 3+4 — Hysteresis Stability Test")
    print("  + Edge_04 Execution Envelope")
    print("=" * 70)

    # ── Sanity check: baseline priority matches known value ──────────────
    baseline_priority = compute_execution_priority(
        CURRENT_CONFIDENCE, CURRENT_CONFLICT, CURRENT_MOF
    )
    print(f"\n  Baseline priority (current state): {baseline_priority}")
    print(f"  Expected: 0.397 — delta: {abs(baseline_priority - 0.397):.6f}")
    assert abs(baseline_priority - 0.397) < 0.001, (
        f"Baseline priority mismatch: {baseline_priority} vs 0.397"
    )
    print("  ✓ Baseline verified\n")

    # ═════════════════════════════════════════════════════════════════════
    # PHASE 3 — Hysteresis Stability Test
    # ═════════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("  PHASE 3 — HYSTERESIS STABILITY TEST")
    print("=" * 70)

    scenarios = ["hold_all", "single_at_5", "repeated"]
    scenario_labels = {
        "hold_all": "1. No execution ever (HOLD all 20 cycles)",
        "single_at_5": "2. Single activation at cycle 5 (then back to HOLD)",
        "repeated": "3. Threshold-based activation every cycle (natural)",
    }

    analyses = {}
    histories = {}

    for sc in scenarios:
        print(f"\n  {scenario_labels[sc]}")
        print("  " + "-" * 55)

        history = simulate_scenario(sc)
        histories[sc] = history
        analysis = analyze_hysteresis(history)
        analyses[sc] = analysis

        seq = "".join("E" if h["decision"] == "EXECUTE" else "H" for h in history)

        print(f"    Decision sequence ({NUM_CYCLES} cycles): {seq}")
        print(f"    Oscillations (flips):       {analysis['flip_count']}")
        print(f"    Oscillation frequency:      {analysis['oscillation_frequency']}")
        print(f"    Converged (last 5 steady):  {analysis['converged']}")
        print(f"    Steady-state decision:      {analysis['steady_state_decision']}")
        print(f"    MOF:  {analysis['mof_start']} → {analysis['mof_end']}  "
              f"(trend={analysis['mof_trend']:+.4f}, "
              f"volatility={analysis['mof_volatility']:.6f})")
        print(f"    Priority: {analysis['priority_start']} → "
              f"{analysis['priority_end']}  "
              f"(trend={analysis['priority_trend']:+.4f})")
        print(f"    EXECUTE count: {analysis['execute_count']} / "
              f"{analysis['total_cycles']}")

    # Overall verdict
    hysteresis_verdict = classify_hysteresis(analyses)
    print(f"\n  ▸ Hysteresis verdict: {hysteresis_verdict}")

    # ═════════════════════════════════════════════════════════════════════
    # PHASE 4 — Execution Envelope
    # ═════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  PHASE 4 — EDGE_04 EXECUTION ENVELOPE")
    print("=" * 70)

    print("\n  Scanning 3D parameter space for viable execution points...")
    result = find_execution_envelope()

    if isinstance(result, tuple):
        envelope, viable_points = result
    else:
        envelope = result
        viable_points = []

    print(f"  Grid points scanned: {envelope.get('total_grid_points', 'N/A')}")
    print(f"  Viable points found: {envelope['num_viable_points']}")
    print(f"\n  Execution envelope (priority >= {EXECUTION_PRIORITY_THRESHOLD}):")

    if envelope["conf_min"] is None:
        print("  ⚠  No viable execution region found!")
    else:
        print(f"    Confidence:      [{envelope['conf_min']:.4f}, "
              f"{envelope['conf_max']:.4f}]")
        print(f"    Portfolio conflict: [{envelope['conflict_min']:.4f}, "
              f"{envelope['conflict_max']:.4f}]")
        print(f"    MOF stability:   [{envelope['mof_min']:.4f}, "
              f"{envelope['mof_max']:.4f}]")

    # Compare current state
    print(f"\n  Current state check:")
    print(f"    Confidence = {CURRENT_CONFIDENCE:.4f}")
    print(f"    Conflict   = {CURRENT_CONFLICT:.4f}")
    print(f"    MOF        = {CURRENT_MOF:.4f}")

    inside, check_details = check_state_in_envelope(envelope)
    print(f"\n  Envelope membership:")
    print(f"    Confidence in range:  {check_details.get('confidence_in_range', 'N/A')}")
    print(f"    Conflict in range:    {check_details.get('conflict_in_range', 'N/A')}")
    print(f"    MOF in range:         {check_details.get('mof_in_range', 'N/A')}")

    if inside:
        print(f"\n  ▸ RESULT: INSIDE ENVELOPE ✅ — Execution is safe")
    else:
        print(f"\n  ▸ RESULT: OUTSIDE ENVELOPE ❌ — Execution is blocked")

    # ── Assemble overall verdict ────────────────────────────────────────
    verdict = {
        "hysteresis_characteristic": hysteresis_verdict,
        "inside_execution_envelope": inside,
        "execution_readiness": "DEFER" if not inside else "CONDITIONAL",
        "note": (
            "Current edge_04 signal does not satisfy minimum viable activation "
            "conditions. Execution remains BLOCKED until MOF stability improves "
            "or portfolio conflict decreases."
            if not inside else
            "Edge_04 signal satisfies envelope conditions. Monitor for "
            "decision-loop stability before execution."
        ),
    }

    # ── Build output object ─────────────────────────────────────────────
    hysteresis_output = {
        "description": (
            f"Simulated {NUM_CYCLES} decision cycles under 3 scenarios "
            "to test decision stability and hysteresis."
        ),
        "simulation_parameters": {
            "num_cycles": NUM_CYCLES,
            "threshold": EXECUTION_PRIORITY_THRESHOLD,
            "state_transition_rules": {
                "on_execute": {
                    "mof_stability_increase": 0.05,
                    "portfolio_conflict_increase": 0.10,
                    "repeated_within_3_cycles_penalty": -0.10,
                },
                "on_hold": {
                    "mof_random_walk_drift": "±0.01",
                    "conflict_unchanged": True,
                },
            },
        },
        "scenarios": {},
    }

    for sc in scenarios:
        hysteresis_output["scenarios"][sc] = {
            "label": scenario_labels[sc],
            "analysis": analyses[sc],
            "history": histories[sc],
        }

    # ── Save ────────────────────────────────────────────────────────────
    out_path = save_report(hysteresis_output, envelope, check_details, verdict)
    print(f"\n  Report saved to: {out_path}")

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Hysteresis: {hysteresis_verdict}")
    print(f"  Envelope: [{envelope.get('conf_min', 'N/A'):.4f}–{envelope.get('conf_max', 'N/A'):.4f}] × "
          f"[{envelope.get('conflict_min', 'N/A'):.4f}–{envelope.get('conflict_max', 'N/A'):.4f}] × "
          f"[{envelope.get('mof_min', 'N/A'):.4f}–{envelope.get('mof_max', 'N/A'):.4f}]")
    print(f"  Current state inside envelope: {inside}")
    print(f"  Recommendation: {verdict['execution_readiness'].upper()}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
