"""Execution Arbitration Layer — Batch 6.5 Phases 1+2.

Resolves competing valid truths by building a Position-State Conflict Graph and
computing Execution Priority Scores for candidate signals against open positions.

Phase 1 — Position-State Conflict Graph
----------------------------------------
- Build a graph of all open positions + candidate signal(s)
- Compute pairwise conflict weights: direct, cross-hedge, exposure overlap, none
- Derive net portfolio contradiction score and edge-specific conflict burden

Phase 2 — Execution Priority Scoring
-------------------------------------
- Composite score from signal confidence, structural invariance,
  contamination resistance, minus portfolio conflict penalty
- Threshold-based eligibility classification (High / Medium / Low)

Usage::
    cd proxima_x
    python bootstrap/execution_arbitration.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
_STATE_DIR = os.path.join(_PROJECT_ROOT, "state")

sys.path.insert(0, _PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("execution_arbitration")

# ---------------------------------------------------------------------------
# Constants — open positions from specification
# ---------------------------------------------------------------------------

OPEN_POSITIONS = [
    {
        "label": "P1",
        "symbol": "NZDCAD",
        "direction": 1,       # BUY
        "volume": 0.12,
        "entry_price": 0.80547,
        "ticket": 57332318077,
        "pnl": 3.46,
    },
    {
        "label": "P2",
        "symbol": "EURUSD",
        "direction": 1,       # BUY
        "volume": 0.01,
        "entry_price": 1.14126,
        "ticket": 57343721376,
        "pnl": -0.99,
    },
    {
        "label": "P3",
        "symbol": "EURJPY",
        "direction": -1,      # SELL
        "volume": 0.01,
        "entry_price": 185.655,
        "ticket": 57346204972,
        "pnl": 0.01,
    },
    {
        "label": "P4",
        "symbol": "GBPUSD",
        "direction": -1,      # SELL
        "volume": 0.01,
        "entry_price": 1.32340,
        "ticket": 57346205006,
        "pnl": 0.09,
    },
    {
        "label": "P5",
        "symbol": "NZDCAD",
        "direction": -1,      # SELL
        "volume": 0.10,
        "entry_price": 0.80588,
        "ticket": 57346205030,
        "pnl": 0.21,
    },
]

CANDIDATE_SIGNAL = {
    "label": "edge_04",
    "symbol": "EURJPY",
    "direction": 1,          # BUY
    "confidence": 0.7656,
    "strategy": "pullback",
}

# Cross-hedge correlation weights between symbol pairs.
# Keyed by (symbol_a, symbol_b) in sorted order.
# Only explicitly specified correlations are included.
CROSS_HEDGE_WEIGHTS: Dict[Tuple[str, str], float] = {
    ("EURJPY", "EURUSD"): 0.7,
}

# Base currency lookup (simplified forex mapping)
BASE_CURRENCY: Dict[str, str] = {
    "EURJPY": "EUR",
    "EURUSD": "EUR",
    "GBPUSD": "GBP",
    "NZDCAD": "NZD",
}

# Weights for execution priority formula
W1 = 0.30   # signal_confidence_score
W2 = 0.25   # structural_invariance_score
W3 = 0.15   # contamination_resistance_score
W4 = 0.30   # portfolio_conflict_penalty

# Eligibility thresholds
HIGH_PRIORITY_THRESHOLD = 0.70
MEDIUM_PRIORITY_THRESHOLD = 0.50


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _symkey(a: str, b: str) -> Tuple[str, str]:
    """Return sorted tuple for consistent cross-hedge lookup."""
    return tuple(sorted((a, b)))


def compute_pair_conflict(
    node_a: Dict[str, Any],
    node_b: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute conflict weight and type between two nodes.

    Resolution order (first match wins):
      1. Direct conflict — same symbol, opposite direction  →  weight 1.0
      2. Cross-hedge conflict — correlated symbol pairs     →  weight 0.7 (or per-pair)
      3. Exposure overlap — same base currency              →  weight 0.5
      4. No conflict                                        →  weight 0.0
    """
    sym_a = node_a["symbol"]
    sym_b = node_b["symbol"]
    dir_a = node_a["direction"]
    dir_b = node_b["direction"]

    # --- 1. Direct conflict ---
    if sym_a == sym_b and dir_a != dir_b:
        return {
            "node_a": node_a["label"],
            "node_b": node_b["label"],
            "conflict_type": "direct",
            "conflict_weight": 1.0,
            "detail": f"Same symbol {sym_a}, opposite direction",
        }

    # --- 2. Cross-hedge conflict ---
    ch_key = _symkey(sym_a, sym_b)
    if ch_key in CROSS_HEDGE_WEIGHTS:
        ch_weight = CROSS_HEDGE_WEIGHTS[ch_key]
        return {
            "node_a": node_a["label"],
            "node_b": node_b["label"],
            "conflict_type": "cross_hedge",
            "conflict_weight": ch_weight,
            "detail": f"Correlated pairs {sym_a} ↔ {sym_b}",
        }

    # --- 3. Exposure overlap ---
    base_a = BASE_CURRENCY.get(sym_a)
    base_b = BASE_CURRENCY.get(sym_b)
    if base_a and base_b and base_a == base_b:
        return {
            "node_a": node_a["label"],
            "node_b": node_b["label"],
            "conflict_type": "exposure_overlap",
            "conflict_weight": 0.5,
            "detail": f"Same base currency {base_a}",
        }

    # --- 4. No conflict ---
    return {
        "node_a": node_a["label"],
        "node_b": node_b["label"],
        "conflict_type": "none",
        "conflict_weight": 0.0,
        "detail": "No material conflict",
    }


# ---------------------------------------------------------------------------
# Phase 1 — Conflict Graph
# ---------------------------------------------------------------------------

def build_conflict_graph(
    positions: List[Dict[str, Any]],
    candidate: Dict[str, Any],
) -> Dict[str, Any]:
    """Build complete conflict graph from all nodes."""
    all_nodes = positions + [candidate]
    n = len(all_nodes)
    edges: List[Dict[str, Any]] = []
    total_weight = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            edge = compute_pair_conflict(all_nodes[i], all_nodes[j])
            edges.append(edge)
            total_weight += edge["conflict_weight"]

    max_possible = n * (n - 1) / 2  # complete graph, all weights=1.0
    net_contradiction = total_weight / max_possible if max_possible > 0 else 0.0

    # Edge_04 specific conflict burden (sum of conflicts with candidate)
    edge_04_burden = 0.0
    edge_04_conflicts: List[Dict[str, Any]] = []
    for edge in edges:
        if candidate["label"] in (edge["node_a"], edge["node_b"]):
            edge_04_burden += edge["conflict_weight"]
            edge_04_conflicts.append(edge)

    # Normalise edge_04 burden to [0, 1] (divide by number of positions)
    num_positions = len(positions)
    edge_04_burden_normalised = (
        edge_04_burden / num_positions if num_positions > 0 else 0.0
    )

    return {
        "total_nodes": n,
        "num_open_positions": num_positions,
        "num_candidates": 1,
        "edges": edges,
        "total_conflict_weight": total_weight,
        "max_possible_conflict_weight": max_possible,
        "net_portfolio_contradiction_score": round(net_contradiction, 4),
        "edge_04_conflict_burden_raw": round(edge_04_burden, 4),
        "edge_04_conflict_burden_normalised": round(edge_04_burden_normalised, 4),
        "edge_04_conflicts": edge_04_conflicts,
    }


# ---------------------------------------------------------------------------
# Phase 2 — Execution Priority Scoring
# ---------------------------------------------------------------------------

def compute_execution_priority(
    candidate: Dict[str, Any],
    identity_score: float,
    contamination_audit: Dict[str, Any],
    conflict_graph: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute execution priority score for the candidate signal.

    Parameters
    ----------
    candidate : dict
        Candidate signal node (must have 'confidence' key).
    identity_score : float
        Structural invariance / identity score from Phase 3 lock (0-1).
    contamination_audit : dict
        Contamination audit data with contamination_coefficients per layer.
    conflict_graph : dict
        Output from build_conflict_graph().

    Returns
    -------
    dict with all sub-component scores and final aggregate.
    """
    # 1. Signal confidence score (from isolated signature)
    signal_confidence_score = candidate["confidence"]

    # 2. Structural invariance score (from identity lock)
    structural_invariance_score = identity_score

    # 3. Contamination resistance score
    #    avg_contamination = mean of (lifecycle, MOF, RF) coefficients
    #    resistance = 1 - avg_contamination
    layers = contamination_audit.get("contamination_layers", {})
    coeffs = []
    for layer_key in ("lifecycle", "mof", "rf"):
        layer_data = layers.get(layer_key, {})
        coeff = layer_data.get("contamination_coefficient", 0.0)
        coeffs.append(coeff)

    avg_contamination = sum(coeffs) / len(coeffs) if coeffs else 0.0
    contamination_resistance_score = 1.0 - avg_contamination

    # 4. Portfolio conflict penalty (normalised edge_04 burden)
    portfolio_conflict_penalty = conflict_graph["edge_04_conflict_burden_normalised"]

    # Weighted composite
    execution_priority_score = (
        W1 * signal_confidence_score
        + W2 * structural_invariance_score
        + W3 * contamination_resistance_score
        - W4 * portfolio_conflict_penalty
    )
    execution_priority_score = max(0.0, min(1.0, execution_priority_score))

    # Eligibility
    if execution_priority_score >= HIGH_PRIORITY_THRESHOLD:
        eligibility = "HIGH"
    elif execution_priority_score >= MEDIUM_PRIORITY_THRESHOLD:
        eligibility = "MEDIUM"
    else:
        eligibility = "LOW"

    return {
        "candidate": candidate["label"],
        "candidate_symbol": candidate["symbol"],
        "candidate_direction": candidate["direction"],
        "execution_priority_score": round(execution_priority_score, 4),
        "execution_eligibility_level": eligibility,
        "eligibility_thresholds": {
            "high_priority_min": HIGH_PRIORITY_THRESHOLD,
            "medium_priority_min": MEDIUM_PRIORITY_THRESHOLD,
            "low_priority_max": MEDIUM_PRIORITY_THRESHOLD,
        },
        "sub_components": {
            "signal_confidence_score": {
                "value": round(signal_confidence_score, 4),
                "weight": W1,
                "weighted_contribution": round(
                    W1 * signal_confidence_score, 4
                ),
                "source": "edge_04_isolated_signature.json",
            },
            "structural_invariance_score": {
                "value": round(structural_invariance_score, 4),
                "weight": W2,
                "weighted_contribution": round(
                    W2 * structural_invariance_score, 4
                ),
                "source": "edge_04_identity_lock.json",
            },
            "contamination_resistance_score": {
                "value": round(contamination_resistance_score, 4),
                "weight": W3,
                "weighted_contribution": round(
                    W3 * contamination_resistance_score, 4
                ),
                "source": "contamination_audit.json",
                "contamination_coefficients_used": {
                    "lifecycle": coeffs[0] if len(coeffs) > 0 else None,
                    "mof": coeffs[1] if len(coeffs) > 1 else None,
                    "rf": coeffs[2] if len(coeffs) > 2 else None,
                },
                "avg_contamination": round(avg_contamination, 4),
            },
            "portfolio_conflict_penalty": {
                "value": round(portfolio_conflict_penalty, 4),
                "weight": W4,
                "weighted_contribution": round(
                    W4 * portfolio_conflict_penalty, 4
                ),
                "source": "conflict_graph (Phase 1)",
            },
        },
        "formula_used": (
            "execution_priority_score = "
            f"{W1}*signal_confidence + {W2}*structural_invariance + "
            f"{W3}*contamination_resistance - {W4}*portfolio_conflict_penalty"
        ),
    }


# ---------------------------------------------------------------------------
# Load state data
# ---------------------------------------------------------------------------

def load_json(filename: str) -> Dict[str, Any]:
    path = os.path.join(_STATE_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("=" * 60)
    log.info("Execution Arbitration Layer — Batch 6.5 Phases 1+2")
    log.info("=" * 60)

    # ------------------------------------------------------------------
    # Load state data files
    # ------------------------------------------------------------------
    log.info("Loading state data...")
    isolated_sig = load_json("edge_04_isolated_signature.json")
    identity_lock = load_json("edge_04_identity_lock.json")
    contamination = load_json("contamination_audit.json")

    identity_score = identity_lock["edge_04_identity_lock"]["identity_score"]
    log.info("  Identity score (structural invariance): %.4f", identity_score)

    # ------------------------------------------------------------------
    # Phase 1 — Conflict Graph
    # ------------------------------------------------------------------
    log.info("")
    log.info("Phase 1 — Position-State Conflict Graph")
    log.info("-" * 40)

    conflict_graph = build_conflict_graph(OPEN_POSITIONS, CANDIDATE_SIGNAL)

    log.info("  Total nodes: %d (5 positions + 1 candidate)", conflict_graph["total_nodes"])
    log.info("  Total pairwise edges: %d", len(conflict_graph["edges"]))
    log.info("  Net portfolio contradiction score: %.4f", conflict_graph["net_portfolio_contradiction_score"])
    log.info("  Edge_04 conflict burden (raw): %.4f", conflict_graph["edge_04_conflict_burden_raw"])
    log.info("  Edge_04 conflict burden (normalised): %.4f", conflict_graph["edge_04_conflict_burden_normalised"])

    log.info("")
    log.info("  Edge list:")
    for e in conflict_graph["edges"]:
        marker = " <-- CANDIDATE" if CANDIDATE_SIGNAL["label"] in (e["node_a"], e["node_b"]) else ""
        log.info(
            "    %-8s ↔ %-8s  weight=%.1f  type=%-18s%s",
            e["node_a"], e["node_b"],
            e["conflict_weight"],
            e["conflict_type"],
            marker,
        )

    # ------------------------------------------------------------------
    # Phase 2 — Execution Priority Scoring
    # ------------------------------------------------------------------
    log.info("")
    log.info("Phase 2 — Execution Priority Scoring")
    log.info("-" * 40)

    priority = compute_execution_priority(
        candidate=CANDIDATE_SIGNAL,
        identity_score=identity_score,
        contamination_audit=contamination,
        conflict_graph=conflict_graph,
    )

    log.info("  Candidate: %s (%s %s)", priority["candidate"], priority["candidate_symbol"],
              "BUY" if priority["candidate_direction"] == 1 else "SELL")
    log.info("  Sub-components:")
    for comp_name, comp_data in priority["sub_components"].items():
        log.info("    %-35s = %.4f  (w=%.2f, contrib=%.4f)",
                  comp_name,
                  comp_data["value"],
                  comp_data["weight"],
                  comp_data["weighted_contribution"])
    log.info("")
    log.info("  Execution Priority Score: %.4f", priority["execution_priority_score"])
    log.info("  Eligibility Level: %s", priority["execution_eligibility_level"])

    # ------------------------------------------------------------------
    # Build full report
    # ------------------------------------------------------------------
    report = {
        "report_metadata": {
            "report_type": "EXECUTION_PRIORITY_REPORT",
            "phase": "Batch 6.5 Phases 1+2 — Execution Arbitration Layer",
            "candidate_edge": "edge_04",
            "candidate_symbol": "EURJPY",
            "candidate_direction": "BUY",
            "generated_at": datetime.utcnow().isoformat(),
            "data_sources": [
                "edge_04_isolated_signature.json",
                "edge_04_identity_lock.json",
                "contamination_audit.json",
                "lifecycle_state.json (OPENED positions)",
            ],
        },
        "conflict_graph": conflict_graph,
        "execution_priority": priority,
    }

    # Save
    out_path = os.path.join(_STATE_DIR, "execution_priority_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    log.info("")
    log.info("Report saved to: %s", out_path)

    # ------------------------------------------------------------------
    # Summary output for verification
    # ------------------------------------------------------------------
    print("")
    print("=" * 60)
    print("EXECUTION ARBITRATION — RESULT SUMMARY")
    print("=" * 60)
    print(f"Conflict Graph: {len(conflict_graph['edges'])} edges, "
          f"net contradiction = {conflict_graph['net_portfolio_contradiction_score']:.4f}")
    for e in conflict_graph["edges"]:
        print(f"  {e['node_a']:>8s} ↔ {e['node_b']:<8s}  "
              f"weight={e['conflict_weight']:.1f}  [{e['conflict_type']}]")
    print()
    print(f"Edge_04 conflict burden (normalised): {conflict_graph['edge_04_conflict_burden_normalised']:.4f}")
    print()
    print(f"Execution Priority Score:      {priority['execution_priority_score']:.4f}")
    print(f"  Signal confidence:           {priority['sub_components']['signal_confidence_score']['value']:.4f} × {W1}")
    print(f"  Structural invariance:       {priority['sub_components']['structural_invariance_score']['value']:.4f} × {W2}")
    print(f"  Contamination resistance:    {priority['sub_components']['contamination_resistance_score']['value']:.4f} × {W3}")
    print(f"  Portfolio conflict penalty: -{priority['sub_components']['portfolio_conflict_penalty']['value']:.4f} × {W4}")
    print(f"Eligibility:                   {priority['execution_eligibility_level']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
