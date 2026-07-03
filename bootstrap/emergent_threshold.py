#!/usr/bin/env python3
"""
Batch 6.6 Phases 1+2 — Stability vs Signal Tradeoff Surface + Dynamic Threshold Extraction

Builds a 3D simulation surface across confidence, portfolio conflict, and MOF stability
to discover emergent execution thresholds that replace the static 0.70 rule.

Phase 1: Generate 1000-scenario surface (10×10×10)
Phase 2: Extract dynamic threshold boundary, compare with static rule

The ratio threshold (χ = 1.0) has an analytic solution:
    χ = exec_priority / max(0.01, stability_delta + 0.5)
    Setting χ = 1.0 → exec_priority = stability_delta + 0.5
    → 0.3·c + 0.25·v + 0.15·r - 0.3·π = μ - 0.5·π + 0.5
    → c* = (μ + 0.5 - 0.2·π - 0.25·v - 0.15·r) / 0.3

    where: c = confidence, μ = MOF stability, π = portfolio conflict
           v = structural invariance (0.9543), r = contamination resistance (0.2049)

    Plugging v and r: c* = (μ + 0.23069 - 0.2·π) / 0.3
"""

import json
import os
import sys
from itertools import product

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE_DIR = os.path.join(os.path.dirname(BASE_DIR), "state")
os.makedirs(STATE_DIR, exist_ok=True)

# ── Fixed system parameters (from Batch 6.5 Phase 2) ───────────────────────
STRUCTURAL_INVARIANCE = 0.9543      # identity lock, weight 0.25
CONTAMINATION_RESISTANCE = 0.2049   # 1 - avg_contamination, weight 0.15
W_SIGNAL = 0.30
W_INVARIANCE = 0.25
W_CONTAMINATION = 0.15
W_CONFLICT = 0.30

# Derived constant: 0.25*0.9543 + 0.15*0.2049 = 0.238575 + 0.030735 = 0.26931
FIXED_CONTRIBUTION = W_INVARIANCE * STRUCTURAL_INVARIANCE + W_CONTAMINATION * CONTAMINATION_RESISTANCE

# ── Current system state ───────────────────────────────────────────────────
CURRENT_CONFIDENCE = 0.7656
CURRENT_PORTFOLIO_CONFLICT = 0.34       # edge_04 conflict burden normalised
CURRENT_MOF_STABILITY = 0.4609          # MOF at STRUCTURE_LIMITED


# ═══════════════════════════════════════════════════════════════════════════
# CORE FORMULAS
# ═══════════════════════════════════════════════════════════════════════════

def compute_execution_priority(confidence, portfolio_conflict):
    """
    Execution Priority Score (Batch 6.5 Phase 2 formula):
      score = 0.30*confidence + 0.25*invariance + 0.15*contamination_resistance - 0.30*conflict
    """
    score = (
        W_SIGNAL * confidence
        + FIXED_CONTRIBUTION
        - W_CONFLICT * portfolio_conflict
    )
    return round(score, 6)


def compute_stability_delta(mof_stability, portfolio_conflict):
    """System Stability Delta = MOF_stability - (portfolio_conflict × 0.5)."""
    return round(mof_stability - portfolio_conflict * 0.5, 6)


def compute_ratio(exec_priority, stability_delta):
    """
    Signal Gain vs Stability Loss Ratio.

    χ = exec_priority / max(0.01, stability_delta + 0.5)

    χ > 1.0  →  signal gain > stability loss (favourable)
    χ < 1.0  →  stability loss dominates (unfavourable)
    """
    denominator = max(0.01, stability_delta + 0.5)
    return round(exec_priority / denominator, 6)


def theoretical_threshold(mof_stability, portfolio_conflict):
    """
    Analytical solution for the confidence threshold where χ = 1.0.

    Derived from: 0.3*c + FIXED_CONTRIBUTION - 0.3*π = μ - 0.5*π + 0.5
    where FIXED_CONTRIBUTION = 0.25*v + 0.15*r = 0.26931

    → c* = (μ + 0.5 - 0.2*π - FIXED_CONTRIBUTION) / 0.3
    → c* = (μ + 0.23069 - 0.2*π) / 0.3
    """
    numerator = mof_stability + 0.5 - 0.2 * portfolio_conflict - FIXED_CONTRIBUTION
    # = mof_stability + 0.23069 - 0.2 * portfolio_conflict
    return round(numerator / W_SIGNAL, 6)


def the_threshold_formula():
    """Return the analytic threshold formula with current constants."""
    a = (0.5 - FIXED_CONTRIBUTION) / W_SIGNAL  # (0.5 - 0.26931) / 0.3 ≈ 0.76897
    b = -0.2 / W_SIGNAL                         # -0.2 / 0.3 ≈ -0.66667
    c_coeff = 1.0 / W_SIGNAL                    # 1.0 / 0.3 ≈ 3.33333
    return {
        "coefficients": {
            "intercept_a": round(a, 6),
            "conflict_coefficient_b": round(b, 6),
            "stability_coefficient_c": round(c_coeff, 6),
        },
        "formula": "threshold = a + b*portfolio_conflict + c*MOF_stability",
        "exact_formula": "threshold = (MOF_stability + 0.23069 - 0.2*portfolio_conflict) / 0.3",
        "model": "analytic_linear",
    }


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 1 — Build the 3D Tradeoff Surface
# ═══════════════════════════════════════════════════════════════════════════

def build_surface():
    """
    Vary 3 parameters across 10 steps each → 1000 scenarios.

    Parameters:
        confidence:         0.60 → 0.90  (10 steps)
        portfolio_conflict: 0.00 → 1.00  (10 steps)
        mof_stability:      0.30 → 0.70  (10 steps)
    """
    confidence_steps = [round(0.60 + i * (0.90 - 0.60) / 9, 4) for i in range(10)]
    conflict_steps = [round(0.0 + i * (1.0 - 0.0) / 9, 4) for i in range(10)]
    stability_steps = [round(0.30 + i * (0.70 - 0.30) / 9, 4) for i in range(10)]

    surface = []
    for conf, confl, stab in product(confidence_steps, conflict_steps, stability_steps):
        exec_priority = compute_execution_priority(conf, confl)
        stab_delta = compute_stability_delta(stab, confl)
        ratio = compute_ratio(exec_priority, stab_delta)
        analytic_thresh = theoretical_threshold(stab, confl)

        static_rule = "EXECUTE" if conf >= 0.70 else "DEFER"
        emergent_rule = "EXECUTE" if ratio > 1.0 else "DEFER"

        surface.append({
            "confidence": conf,
            "portfolio_conflict": confl,
            "mof_stability": stab,
            "execution_priority_score": exec_priority,
            "stability_delta": stab_delta,
            "signal_gain_ratio": ratio,
            "theoretical_threshold": analytic_thresh,
            "static_rule": static_rule,
            "emergent_rule": emergent_rule,
        })

    return surface


def save_surface(surface):
    path = os.path.join(STATE_DIR, "execution_viability_surface.json")
    payload = {
        "report_metadata": {
            "report_type": "EXECUTION_VIABILITY_SURFACE",
            "phase": "Batch 6.6 Phase 1 — Stability vs Signal Tradeoff Surface",
            "description": "1000-point 3D simulation surface across confidence, portfolio_conflict, MOF_stability",
            "parameters": {
                "confidence_range": [0.60, 0.90],
                "portfolio_conflict_range": [0.0, 1.0],
                "mof_stability_range": [0.30, 0.70],
                "steps_per_dimension": 10,
                "total_scenarios": 1000
            },
            "fixed_parameters": {
                "structural_invariance": STRUCTURAL_INVARIANCE,
                "contamination_resistance": CONTAMINATION_RESISTANCE,
                "fixed_contribution": FIXED_CONTRIBUTION,
                "weights": {
                    "signal": W_SIGNAL,
                    "invariance": W_INVARIANCE,
                    "contamination": W_CONTAMINATION,
                    "conflict": W_CONFLICT
                }
            }
        },
        "surface": surface
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[PHASE 1] Surface saved to {path} ({len(surface)} points)")
    return path


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 2 — Dynamic Threshold Discovery
# ═══════════════════════════════════════════════════════════════════════════

def analyze_surface_statistics(surface):
    """Compute summary statistics on the surface."""
    ratios = [p["signal_gain_ratio"] for p in surface]
    thresholds = [p["theoretical_threshold"] for p in surface]

    stats = {
        "ratio_min": min(ratios),
        "ratio_max": max(ratios),
        "ratio_mean": round(sum(ratios) / len(ratios), 6),
        "scenarios_with_ratio_above_1": sum(1 for r in ratios if r > 1.0),
        "scenarios_with_ratio_above_1_pct": round(sum(1 for r in ratios if r > 1.0) / len(ratios) * 100, 2),
        "theoretical_threshold_min": min(thresholds),
        "theoretical_threshold_max": max(thresholds),
        "theoretical_threshold_mean": round(sum(thresholds) / len(thresholds), 6),
        "theoretical_threshold_below_1_count": sum(1 for t in thresholds if t <= 1.0),
        "theoretical_threshold_below_0_90_count": sum(1 for t in thresholds if t <= 0.90),
        "theoretical_threshold_below_0_70_count": sum(1 for t in thresholds if t <= 0.70),
    }
    return stats


def compare_with_static_rule(surface):
    """Compare static 0.70 rule vs emergent rule across all 1000 scenarios."""
    total = len(surface)
    same = sum(1 for p in surface if p["static_rule"] == p["emergent_rule"])
    diff = total - same

    static_execute_emergent_defer = sum(
        1 for p in surface
        if p["static_rule"] == "EXECUTE" and p["emergent_rule"] == "DEFER"
    )
    static_defer_emergent_execute = sum(
        1 for p in surface
        if p["static_rule"] == "DEFER" and p["emergent_rule"] == "EXECUTE"
    )

    # Breakdown by conflict level
    by_conflict = {}
    for confl_val in sorted(set(p["portfolio_conflict"] for p in surface)):
        subset = [p for p in surface if p["portfolio_conflict"] == confl_val]
        same_s = sum(1 for p in subset if p["static_rule"] == p["emergent_rule"])
        over = sum(1 for p in subset if p["static_rule"] == "EXECUTE" and p["emergent_rule"] == "DEFER")
        under = sum(1 for p in subset if p["static_rule"] == "DEFER" and p["emergent_rule"] == "EXECUTE")
        by_conflict[str(confl_val)] = {
            "total": len(subset),
            "same": same_s,
            "static_execute_emergent_defer": over,
            "static_defer_emergent_execute": under,
        }

    return {
        "total_scenarios": total,
        "same_decision_count": same,
        "same_decision_pct": round(same / total * 100, 2),
        "different_decision_count": diff,
        "different_decision_pct": round(diff / total * 100, 2),
        "static_execute_emergent_defer": {
            "count": static_execute_emergent_defer,
            "label": "Static says EXECUTE, Emergent says DEFER (over-execution / false positives)",
            "pct": round(static_execute_emergent_defer / total * 100, 2),
            "interpretation": (
                "The static 0.70 threshold approves trades where signal gain "
                "does NOT compensate for stability loss. These are false-positive "
                "executions that degrade system stability."
            )
        },
        "static_defer_emergent_execute": {
            "count": static_defer_emergent_execute,
            "label": "Static says DEFER, Emergent says EXECUTE (missed opportunity / false negatives)",
            "pct": round(static_defer_emergent_execute / total * 100, 2),
            "interpretation": (
                "The static 0.70 threshold blocks trades where signal gain "
                "WOULD compensate for stability loss. These are missed opportunities."
            )
        },
        "by_portfolio_conflict": by_conflict,
    }


def compute_current_dynamic_threshold(theoretical_formula, portfolio_conflict, mof_stability):
    """Compute the dynamic threshold for the current system state."""
    c = theoretical_formula["coefficients"]
    threshold = c["intercept_a"] + c["conflict_coefficient_b"] * portfolio_conflict + c["stability_coefficient_c"] * mof_stability
    return round(threshold, 6)


def run_phase_2(surface):
    """Execute Phase 2 analysis on the surface data."""
    print("\n" + "=" * 70)
    print("PHASE 2 — Dynamic Threshold Discovery")
    print("=" * 70)

    # ── 1. Surface statistics ────────────────────────────────────────────
    stats = analyze_surface_statistics(surface)
    print(f"[PHASE 2] Surface statistics:")
    print(f"         Signal gain ratio range:  {stats['ratio_min']} — {stats['ratio_max']}")
    print(f"         Scenarios with ratio > 1: {stats['scenarios_with_ratio_above_1']} / 1000")
    print(f"         Theoretical threshold range:  {stats['theoretical_threshold_min']} — {stats['theoretical_threshold_max']}")
    print(f"         Thresholds ≤ 1.0:  {stats['theoretical_threshold_below_1_count']} / 1000")
    print(f"         Thresholds ≤ 0.90: {stats['theoretical_threshold_below_0_90_count']} / 1000")
    print(f"         Thresholds ≤ 0.70: {stats['theoretical_threshold_below_0_70_count']} / 1000")

    # ── 2. Derive emergent threshold function (analytic) ─────────────────
    threshold_func = the_threshold_formula()
    print(f"\n[PHASE 2] Emergent threshold function (analytic):")
    print(f"         {threshold_func['exact_formula']}")
    coeffs = threshold_func["coefficients"]
    print(f"         threshold = {coeffs['intercept_a']} + {coeffs['conflict_coefficient_b']}*conflict + {coeffs['stability_coefficient_c']}*stability")
    print(f"         Inverting: as MOF_stability↑, threshold↑ (need more confidence)")
    print(f"                    as portfolio_conflict↑, threshold↓ (less confidence needed because")
    print(f"                    stability delta is already depressed, so the marginal cost is lower)")

    # ── 3. Compare with static rule ──────────────────────────────────────
    comparison = compare_with_static_rule(surface)
    diverge = comparison["different_decision_count"]
    same = comparison["same_decision_count"]
    print(f"\n[PHASE 2] Static (≥0.70) vs Emergent (ratio > 1.0) rule comparison:")
    print(f"         Same decisions:       {same}/1000 ({comparison['same_decision_pct']}%)")
    print(f"         Different decisions:  {diverge}/1000 ({comparison['different_decision_pct']}%)")
    print(f"         Static EXECUTE → Emergent DEFER: {comparison['static_execute_emergent_defer']['count']} ({comparison['static_execute_emergent_defer']['pct']}%)")
    print(f"           → {comparison['static_execute_emergent_defer']['interpretation']}")
    print(f"         Static DEFER → Emergent EXECUTE: {comparison['static_defer_emergent_execute']['count']} ({comparison['static_defer_emergent_execute']['pct']}%)")
    print(f"           → {comparison['static_defer_emergent_execute']['interpretation']}")

    # ── 4. Current state analysis ────────────────────────────────────────
    dynamic_threshold = compute_current_dynamic_threshold(
        threshold_func, CURRENT_PORTFOLIO_CONFLICT, CURRENT_MOF_STABILITY
    )
    current_priority = compute_execution_priority(
        CURRENT_CONFIDENCE, CURRENT_PORTFOLIO_CONFLICT
    )
    current_stab_delta = compute_stability_delta(
        CURRENT_MOF_STABILITY, CURRENT_PORTFOLIO_CONFLICT
    )
    current_ratio = compute_ratio(current_priority, current_stab_delta)

    current_emergent_decision = "EXECUTE" if current_ratio > 1.0 else "DEFER"
    current_static_decision = "EXECUTE" if CURRENT_CONFIDENCE >= 0.70 else "DEFER"

    current_state_result = {
        "edge_04_confidence": CURRENT_CONFIDENCE,
        "portfolio_conflict": CURRENT_PORTFOLIO_CONFLICT,
        "mof_stability": CURRENT_MOF_STABILITY,
        "dynamic_threshold_confidence_required": dynamic_threshold,
        "execution_priority_score": current_priority,
        "stability_delta": current_stab_delta,
        "signal_gain_ratio": current_ratio,
        "emergent_decision": current_emergent_decision,
        "static_decision": current_static_decision,
        "static_rule": f"confidence >= 0.70 → {current_static_decision}",
        "emergent_rule": f"ratio > 1.0 → {current_emergent_decision}",
        "decisions_match": current_emergent_decision == current_static_decision,
        "interpretation": (
            f"The dynamic threshold of {dynamic_threshold} means edge_04 needs "
            f"confidence ≥ {dynamic_threshold} for signal gain to exceed stability loss. "
            f"Current confidence is {CURRENT_CONFIDENCE}, which is "
            f"{'ABOVE' if CURRENT_CONFIDENCE >= dynamic_threshold else 'BELOW'} this threshold. "
            f"Therefore the emergent rule says {current_emergent_decision}. "
            f"The static rule (≥0.70) says {current_static_decision}, which "
            f"{'agrees with' if current_emergent_decision == current_static_decision else 'disagrees with — and is more permissive than'} "
            f"the emergent stability-aware rule."
        ),
        "required_improvement": (
            f"To reach execute threshold: need confidence to increase by "
            f"{round(dynamic_threshold - CURRENT_CONFIDENCE, 4)} "
            f"OR reduce portfolio conflict by "
            f"{round((dynamic_threshold - CURRENT_CONFIDENCE) * 0.3 / 0.2, 4)} "
            f"OR increase MOF stability by "
            f"{round((dynamic_threshold - CURRENT_CONFIDENCE) * 0.3, 4)}"
        ) if dynamic_threshold > CURRENT_CONFIDENCE else "Already past threshold."
    }

    print(f"\n[PHASE 2] Current System State Analysis:")
    print(f"         edge_04 confidence:              {CURRENT_CONFIDENCE}")
    print(f"         Portfolio conflict:              {CURRENT_PORTFOLIO_CONFLICT}")
    print(f"         MOF stability:                   {CURRENT_MOF_STABILITY}")
    print(f"         Execution priority score:        {current_priority}")
    print(f"         Stability delta:                 {current_stab_delta}")
    print(f"         Signal gain ratio:               {current_ratio}")
    print(f"         Dynamic threshold required:      {dynamic_threshold}")
    print(f"         Static rule (≥0.70):             {current_static_decision}")
    print(f"         Emergent rule (ratio>1.0):       {current_emergent_decision}")
    print(f"         Decisions match:                 {current_emergent_decision == current_static_decision}")
    print(f"         {current_state_result['interpretation']}")

    return {
        "surface_statistics": stats,
        "threshold_model": threshold_func,
        "comparison_with_static": comparison,
        "current_state": current_state_result,
    }


def save_threshold_model(phase2_results):
    """Save the complete threshold model to disk."""
    path = os.path.join(STATE_DIR, "emergent_threshold_model.json")
    payload = {
        "report_metadata": {
            "report_type": "EMERGENT_THRESHOLD_MODEL",
            "phase": "Batch 6.6 Phase 2 — Dynamic Threshold Discovery",
            "generated_at": "2026-07-01",
            "description": (
                "Dynamic execution threshold extracted from 3D stability-vs-signal "
                "tradeoff surface. Uses analytic derivation because the signal gain "
                "ratio never numerically crosses 1.0 within the 0.60-0.90 confidence "
                "range for any scenario — discovery is performed analytically."
            )
        },
        "surface_statistics": phase2_results["surface_statistics"],
        "threshold_model": phase2_results["threshold_model"],
        "comparison_with_static_rule": phase2_results["comparison_with_static"],
        "current_system_state": phase2_results["current_state"],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[PHASE 2] Threshold model saved to {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("Batch 6.6 Phases 1+2 — Emergent Execution Threshold Model")
    print("=" * 70)

    # ── Phase 1 — Build surface ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 1 — Stability vs Signal Tradeoff Surface")
    print("=" * 70)
    print("Parameters:")
    print("  Confidence:         0.60 → 0.90 (10 steps)")
    print("  Portfolio conflict: 0.00 → 1.00 (10 steps)")
    print("  MOF stability:      0.30 → 0.70 (10 steps)")
    print(f"  Total scenarios:    10 × 10 × 10 = 1000")
    print(f"\nFixed contributions:")
    print(f"  Structural invariance:      {STRUCTURAL_INVARIANCE} × {W_INVARIANCE} = {W_INVARIANCE * STRUCTURAL_INVARIANCE}")
    print(f"  Contamination resistance:   {CONTAMINATION_RESISTANCE} × {W_CONTAMINATION} = {W_CONTAMINATION * CONTAMINATION_RESISTANCE}")
    print(f"  Fixed contribution total:   {FIXED_CONTRIBUTION}")

    surface = build_surface()
    save_surface(surface)

    # ── Phase 2 — Dynamic threshold ──────────────────────────────────────
    phase2_results = run_phase_2(surface)
    save_threshold_model(phase2_results)

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    cs = phase2_results["current_state"]
    print(f"  Dynamic threshold (current state):        {cs['dynamic_threshold_confidence_required']}")
    print(f"  Current edge_04 confidence:               {cs['edge_04_confidence']}")
    print(f"  Signal gain vs stability loss ratio (χ):  {cs['signal_gain_ratio']}")
    print(f"  Emergent decision (χ > 1.0?):             {cs['emergent_decision']}")
    print(f"  Static rule decision (≥0.70?):            {cs['static_decision']}")
    print(f"  Static ≈ Emergent?                        {cs['decisions_match']}")
    print(f"")
    comp = phase2_results["comparison_with_static"]
    print(f"  VERDICT: The static 0.70 threshold is too permissive.")
    print(f"  For {comp['different_decision_count']}/1000 scenarios")
    print(f"  ({comp['different_decision_pct']}%), the static rule approves trades")
    print(f"  that the emergent stability-aware rule would correctly defer.")
    print(f"  edge_04 EURJPY BUY at confidence={CURRENT_CONFIDENCE}")
    print(f"  should be DEFERRED (emergent) vs EXECUTED (static).")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
