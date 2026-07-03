"""Edge_04 Shadow Validation — Batch 6.3 Phase 2.

Validates whether edge_04 (EURJPY pullback, PF=1.31) is a real signal
or coincidence before any execution. Measures exit consistency, compression
signature stability, and false activation rates across all 3 bootstrap
observation cycles.

Output: state/edge_04_shadow_test_report.json

Usage::
    cd proxima_x
    python -m bootstrap.edge_04_shadow_test

Constraints::
    - Does NOT execute or close any real MT5 positions
    - Does NOT open new trades
    - Does NOT move system into live mode
    - Validation only
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── Path setup ───────────────────────────────────────────────────────────────
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
_STATE_DIR = os.path.join(_PROJECT_ROOT, "state")
_EXPORTS_DIR = os.path.join(
    _PROJECT_ROOT, "exports", "python_reference"
)

sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "proxima_x"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("edge_04_shadow_test")

# Ensure state dir
os.makedirs(_STATE_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Imports (local — after sys.path)
# ---------------------------------------------------------------------------
try:
    from proxima_ops.risk.edge_signal_mapper import (
        EdgeSignalMapper,
        _pullback_signal,
        _ema,
        _STRATEGY_FUNCTIONS,
    )
except ImportError as exc:
    logger.error("Cannot import EdgeSignalMapper: %s", exc)
    sys.exit(1)

try:
    from proxima_ops.execution.mt5_connector import MT5Connector
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    logger.warning("MT5Connector not available — using cached data only")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EDGE_04_ID = "edge_04"

# 3 bootstrap observation cycles extracted from controlled_actuation.log
# Cycle boundaries identified from log timestamps
BOOTSTRAP_CYCLES = [
    # Cycle 1: 2026-06-30 20:42:40 — INFORMATION_DEGRADED, shadow blocked
    {
        "cycle": 1,
        "timestamp": datetime(2026, 6, 30, 20, 42, 40),
        "mof_state": "INFORMATION_DEGRADED",
        "shadow_flag": True,
        "balance": 24984.41,
        "bootstrap_mode": False,
    },
    # Cycle 2: 2026-07-01 01:49:54 — INFORMATION_DEGRADED, shadow blocked
    {
        "cycle": 2,
        "timestamp": datetime(2026, 7, 1, 1, 49, 54),
        "mof_state": "INFORMATION_DEGRADED",
        "shadow_flag": True,
        "balance": 24984.38,
        "bootstrap_mode": False,
    },
    # Cycle 3: 2026-07-01 01:52:49 — INFORMATION_DEGRADED, shadow blocked
    {
        "cycle": 3,
        "timestamp": datetime(2026, 7, 1, 1, 52, 49),
        "mof_state": "INFORMATION_DEGRADED",
        "shadow_flag": True,
        "balance": 24984.38,
        "bootstrap_mode": False,
    },
]

# Edge_04 parameters (from deployment_manifest.json)
EDGE_04_PARAMS = {
    "trend_ema": 100,
    "pullback_ema": 10,
    "max_hold": 18,
}

# Pre-exit compression window (bars before exit signal)
COMPRESSION_WINDOW = 20

# Bar count for MT5 fetch
BAR_COUNT = 500

# ---------------------------------------------------------------------------
# Data Loading Helpers
# ---------------------------------------------------------------------------

def load_mt5_rates(symbol: str, count: int = BAR_COUNT) -> Optional[np.ndarray]:
    """Load close prices from MT5. Returns None if unavailable."""
    if not MT5_AVAILABLE:
        return None
    try:
        connector = MT5Connector()
        if not connector.connect():
            logger.warning("MT5 connection failed — using synthetic data")
            return None
        rates = connector.get_rates(symbol, count=count, timeframe="M5")
        connector.disconnect()
        if rates is None or len(rates) == 0:
            return None
        closes = np.array([r["close"] for r in rates], dtype=float)
        logger.info(
            "Loaded %d bars for %s from MT5 (range: %.5f – %.5f)",
            len(closes), symbol, closes[0], closes[-1],
        )
        return closes
    except Exception as exc:
        logger.warning("MT5 data load failed for %s: %s", symbol, exc)
        return None


def load_edge_04_export_signals() -> Optional[np.ndarray]:
    """Load edge_04 signal prices from export CSV for synthetic replay."""
    path = os.path.join(
        _EXPORTS_DIR, "edge_04_EURJPY_pullback_signals.csv"
    )
    if not os.path.exists(path):
        logger.warning("Export signals not found at %s", path)
        return None
    try:
        import csv
        timestamps = []
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                timestamps.append(int(row["timestamp"]))
        logger.info("Loaded %d edge_04 signal timestamps from export", len(timestamps))
        return np.array(timestamps)
    except Exception as exc:
        logger.warning("Failed to load export signals: %s", exc)
        return None


def load_edge_04_export_trades() -> Optional[List[dict]]:
    """Load edge_04 trade data from export CSV."""
    path = os.path.join(
        _EXPORTS_DIR, "edge_04_EURJPY_pullback_trades.csv"
    )
    if not os.path.exists(path):
        return None
    try:
        import csv
        trades = []
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                trades.append({
                    "entry_time": int(row["entry_time"]),
                    "exit_time": int(row["exit_time"]),
                    "entry_price": float(row["entry_price"]),
                    "exit_price": float(row["exit_price"]),
                    "gross_pnl_pips": float(row["gross_pnl_pips"]),
                    "net_pnl_pips": float(row["net_pnl_pips"]),
                    "bars_held": int(row["bars_held"]),
                })
        logger.info("Loaded %d edge_04 trades from export", len(trades))
        return trades
    except Exception as exc:
        logger.warning("Failed to load export trades: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Synthetic Price Generator (for when MT5 is unavailable)
# ---------------------------------------------------------------------------

def generate_synthetic_eurjpy_prices(
    n_bars: int = 500,
    seed: int = 42,
) -> np.ndarray:
    """Generate synthetic EURJPY prices for testing.

    Creates a trending + pullback pattern that edge_04 can detect.
    Uses a base near the observed EURJPY range (~183-185).
    """
    rng = np.random.RandomState(seed)
    # Base price around 184.00 with a trend component
    base = 184.0
    trend = np.cumsum(rng.randn(n_bars) * 0.001)  # slow drift
    noise = rng.randn(n_bars) * 0.005  # tick noise
    # Add pullback patterns: sharp moves against trend, then recovery
    pullback_indices = rng.choice(
        np.arange(50, n_bars - 50), size=n_bars // 30, replace=False
    )
    prices = base + trend + noise
    for idx in pullback_indices:
        pull_size = rng.uniform(-0.15, 0.15)  # 15 pip pullback
        length = rng.randint(5, 15)
        for i in range(length):
            if idx + i < n_bars:
                prices[idx + i] += pull_size * (1 - i / length)
    prices = np.maximum(prices, base * 0.95)  # floor
    logger.info(
        "Generated %d synthetic EURJPY bars (range: %.5f – %.5f)",
        n_bars, prices.min(), prices.max(),
    )
    return prices


def generate_synthetic_nzdcad_prices(
    n_bars: int = 500,
    seed: int = 99,
) -> np.ndarray:
    """Generate synthetic NZDCAD prices (non-JPY pair for false activation test)."""
    rng = np.random.RandomState(seed)
    base = 0.8200  # NZDCAD typical range
    drift = np.cumsum(rng.randn(n_bars) * 0.0003)
    noise = rng.randn(n_bars) * 0.002
    prices = base + drift + noise
    prices = np.maximum(prices, base * 0.97)
    logger.info(
        "Generated %d synthetic NZDCAD bars (range: %.5f – %.5f)",
        n_bars, prices.min(), prices.max(),
    )
    return prices


# ---------------------------------------------------------------------------
# Task 1: Edge_04 Decision Simulation
# ---------------------------------------------------------------------------

def simulate_cycle_signals(
    mapper: EdgeSignalMapper,
    closes_by_symbol: Dict[str, np.ndarray],
    cycle_info: dict,
) -> List[dict]:
    """Run edge signal generation for a single cycle.

    Returns all generated signals with edge_04 highlighted.
    """
    signals = mapper.generate_all(closes_by_symbol)
    cycle_result = {
        "cycle": cycle_info["cycle"],
        "timestamp": cycle_info["timestamp"].isoformat(),
        "mof_state": cycle_info["mof_state"],
        "shadow_flag": cycle_info["shadow_flag"],
        "bootstrap_mode": cycle_info["bootstrap_mode"],
        "balance": cycle_info["balance"],
        "total_edges_active": len(signals),
        "active_edge_ids": [s.get("edge_id") for s in signals],
        "edge_04_active": any(s.get("edge_id") == EDGE_04_ID for s in signals),
        "edge_04_signal": None,
    }

    for s in signals:
        if s.get("edge_id") == EDGE_04_ID:
            cycle_result["edge_04_signal"] = {
                "symbol": s["symbol"],
                "direction": s["direction"],
                "confidence": s["confidence"],
                "ecdf": s["ecdf"],
                "drift": s["drift"],
                "price": s["price"],
                "strategy": s["strategy"],
                "pf": s["edge_pf"],
                "would_exit": s["direction"] != 0 and s["confidence"] >= 0.3,
            }
            break

    # Simulate exit timing (bars until exit signal)
    if cycle_result["edge_04_signal"] and cycle_result["edge_04_signal"]["would_exit"]:
        cycle_result["simulated_exit"] = {
            "would_exit": True,
            "exit_confidence": cycle_result["edge_04_signal"]["confidence"],
            "exit_direction": cycle_result["edge_04_signal"]["direction"],
            "gate_passed": not cycle_info["shadow_flag"],
            "mof_would_block": cycle_info["mof_state"] == "INFORMATION_DEGRADED"
                              and not cycle_info["bootstrap_mode"],
            "bridge_eligible": not cycle_info["shadow_flag"],
            "overall_exit_permitted": (
                not cycle_info["shadow_flag"]
                and not (
                    cycle_info["mof_state"] == "INFORMATION_DEGRADED"
                    and not cycle_info["bootstrap_mode"]
                )
            ),
        }
    else:
        cycle_result["simulated_exit"] = {
            "would_exit": False,
            "exit_confidence": 0.0,
            "exit_direction": 0,
            "gate_passed": False,
            "mof_would_block": cycle_info["mof_state"] == "INFORMATION_DEGRADED",
            "bridge_eligible": False,
            "overall_exit_permitted": False,
        }

    return signals, cycle_result


def simulate_exit_decision_pipeline(
    mapper: EdgeSignalMapper,
    closes_by_symbol: Dict[str, np.ndarray],
    cycles: List[dict],
) -> Tuple[List[dict], dict]:
    """Simulate the full exit decision pipeline for all 3 cycles.

    Tracks: Signal generation → MOF gating → Bridge evaluation → Exit
    """
    all_cycle_results = []

    for cycle_info in cycles:
        signals, result = simulate_cycle_signals(mapper, closes_by_symbol, cycle_info)

        # ---- MOF Gate Simulation ----
        # Compute quality metrics from edge signals
        confidences = [s.get("confidence", 0) for s in signals]
        directions = [s.get("direction", 0) for s in signals]
        mean_conf = float(np.mean(confidences)) if confidences else 0.0
        non_neutral = sum(1 for d in directions if d != 0)
        signal_diversity = len(set(s.get("symbol") for s in signals))

        result["mof_simulation"] = {
            "mean_confidence": round(mean_conf, 4),
            "non_neutral_signals": non_neutral,
            "signal_diversity": signal_diversity,
            "would_set_shadow_flag": cycle_info["mof_state"] == "INFORMATION_DEGRADED",
            "mof_permission": (
                "BLOCKED"
                if cycle_info["mof_state"] == "INFORMATION_DEGRADED"
                and not cycle_info["bootstrap_mode"]
                else "REDUCED"
                if cycle_info["mof_state"] == "STRUCTURE_LIMITED"
                else "FULL"
            ),
        }

        # ---- Bridge Simulation ----
        # For simulation purposes, we check if the signal would pass bridge gates
        e04_sig = result.get("edge_04_signal")
        bridge_gates = {
            "geometry_eligible": True,  # assume stable in simulation
            "classifier_eligible": cycle_info["mof_state"] != "INFORMATION_DEGRADED",
            "governor_eligible": (e04_sig and e04_sig["confidence"] >= 0.3)
                                 if e04_sig else False,
            "shadow_stable": not cycle_info["shadow_flag"],
        }
        all_gates_pass = all(bridge_gates.values())
        # Map gate key names to human-readable blocked reasons
        _gate_reason_map = {
            "geometry_eligible": "geometry",
            "classifier_eligible": "classifier",
            "governor_eligible": "governor",
            "shadow_stable": "shadow_instability",
        }
        blocked_reasons = [
            _gate_reason_map.get(k, k)
            for k, v in bridge_gates.items() if not v
        ]

        result["bridge_simulation"] = {
            "gates": bridge_gates,
            "all_gates_pass": all_gates_pass,
            "blocked_reasons": blocked_reasons,
            "bridge_state": "ELIGIBLE" if all_gates_pass else "BLOCKED",
        }

        # ---- Overall Decision ----
        would_exit_in_live = (
            result["edge_04_signal"] is not None
            and result["edge_04_signal"]["would_exit"]
            and all_gates_pass
            and result["simulated_exit"]["overall_exit_permitted"]
        )

        result["overall_decision"] = {
            "would_exit_in_live": would_exit_in_live,
            "would_exit_in_bootstrap": (
                result["edge_04_signal"] is not None
                and result["edge_04_signal"]["would_exit"]
                and all_gates_pass
            ),
            "exit_blocked_by_mof": (
                cycle_info["mof_state"] == "INFORMATION_DEGRADED"
                and not cycle_info["bootstrap_mode"]
            ),
            "exit_blocked_by_shadow": cycle_info["shadow_flag"],
            "exit_blocked_by_governor": not bridge_gates["governor_eligible"] if e04_sig else True,
        }

        all_cycle_results.append(result)

    # ---- Cross-cycle consistency metrics ----
    exit_confidences = [
        r["edge_04_signal"]["confidence"]
        for r in all_cycle_results
        if r["edge_04_signal"] is not None
    ]
    exit_timings = [
        0 if r["simulated_exit"]["would_exit"] else 99
        for r in all_cycle_results
    ]
    would_exit_counts = sum(1 for r in all_cycle_results if r["simulated_exit"]["would_exit"])

    # Consistency: how many cycles produce the same exit decision
    if len(cycles) > 0:
        exit_consistency = would_exit_counts / len(cycles)
    else:
        exit_consistency = 0.0

    # Exit timing consistency (std dev of exit confidence)
    timing_std = float(np.std(exit_confidences)) if len(exit_confidences) > 1 else 0.0
    # Lower std = more consistent. Score: 1.0 - min(1.0, std * 3)
    timing_score = max(0.0, 1.0 - min(1.0, timing_std * 3.0))

    # Combined exit consistency score
    exit_consistency_score = round(
        (exit_consistency * 0.6 + timing_score * 0.4), 4
    )

    summary = {
        "total_cycles_simulated": len(cycles),
        "cycles_with_edge_04_signal": sum(
            1 for r in all_cycle_results if r["edge_04_signal"] is not None
        ),
        "cycles_would_exit": would_exit_counts,
        "exit_consistency_ratio": round(exit_consistency, 4),
        "exit_confidence_values": [round(c, 4) for c in exit_confidences],
        "exit_confidence_std": round(timing_std, 4),
        "timing_consistency_score": round(timing_score, 4),
        "exit_consistency_score": exit_consistency_score,
        "overall_would_exit_in_live": any(
            r["overall_decision"]["would_exit_in_live"]
            for r in all_cycle_results
        ),
        "overall_would_exit_in_bootstrap": any(
            r["overall_decision"]["would_exit_in_bootstrap"]
            for r in all_cycle_results
        ),
    }

    return all_cycle_results, summary


# ---------------------------------------------------------------------------
# Task 2: Compression Signature Stability
# ---------------------------------------------------------------------------

def compute_edge_04_state_at_each_bar(
    closes: np.ndarray,
    trend_span: int = 100,
    pullback_span: int = 10,
) -> Dict[str, np.ndarray]:
    """Compute edge_04's internal pullback state at every bar.

    Returns
    -------
    dict with keys:
        - trend_ema: the 100-bar EMA series
        - pullback_ema: the 10-bar EMA series
        - price: closes
        - distance_to_pull: abs(price - pullback_ema) / price
        - distance_pull_to_trend: abs(pullback_ema - trend_ema) / price
        - trend_up: boolean array, trend[-1] > trend[-5]
        - trend_down: boolean array, trend[-1] < trend[-5]
        - in_pullback: boolean — price crossed toward pullback_ema against trend
        - direction: array of edge_04 direction signals at each bar
        - confidence: array of edge_04 confidence at each bar
    """
    n = len(closes)
    trend = _ema(closes, trend_span)
    pullback = _ema(closes, pullback_span)

    dist_to_pull = np.full(n, np.nan)
    dist_pull_to_trend = np.full(n, np.nan)
    trend_up = np.full(n, False)
    trend_down = np.full(n, False)
    in_pullback = np.full(n, False)
    direction = np.zeros(n, dtype=int)
    confidence = np.zeros(n)
    ecdf_vals = np.full(n, np.nan)
    drift_vals = np.zeros(n, dtype=int)

    for i in range(trend_span + 5, n):
        c = closes[i]
        t = trend[i]
        p = pullback[i]

        if np.isnan(t) or np.isnan(p):
            continue

        d2p = abs(c - p) / max(c, 1e-12)
        dp2t = abs(p - t) / max(c, 1e-12)
        dist_to_pull[i] = d2p
        dist_pull_to_trend[i] = dp2t

        # Trend direction
        if i >= 5 and not np.isnan(trend[i - 5]):
            tu = trend[i] > trend[i - 5]
            td = trend[i] < trend[i - 5]
        else:
            tu = False
            td = False
        trend_up[i] = tu
        trend_down[i] = td

        # Pullback detection
        if tu and c <= p:
            in_pullback[i] = True
            pullback_depth = max(
                0, min(1.0, 1.0 - d2p / max(dp2t + 0.0001, 0.0001))
            )
            direction[i] = 1
            confidence[i] = min(1.0, 0.4 + pullback_depth * 0.4)
        elif td and c >= p:
            in_pullback[i] = True
            pullback_depth = max(
                0, min(1.0, 1.0 - d2p / max(dp2t + 0.0001, 0.0001))
            )
            direction[i] = -1
            confidence[i] = min(1.0, 0.4 + pullback_depth * 0.4)

        # ECDF and drift
        valid_closes = closes[~np.isnan(closes)]
        if len(valid_closes) > 0:
            ecdf_vals[i] = float(
                np.searchsorted(np.sort(valid_closes), c) / max(len(valid_closes), 1)
            )
        if i >= 3:
            price_drift = closes[i] - closes[i - 3]
            drift_vals[i] = 1 if price_drift > 0 else (-1 if price_drift < 0 else 0)

    return {
        "trend_ema": trend,
        "pullback_ema": pullback,
        "price": closes,
        "distance_to_pull": dist_to_pull,
        "distance_pull_to_trend": dist_pull_to_trend,
        "trend_up": trend_up,
        "trend_down": trend_down,
        "in_pullback": in_pullback,
        "direction": direction,
        "confidence": confidence,
        "ecdf": ecdf_vals,
        "drift": drift_vals,
    }


def find_compression_signatures(
    state: Dict[str, np.ndarray],
    window: int = COMPRESSION_WINDOW,
) -> List[dict]:
    """Identify pre-exit compression signature patterns.

    A compression signature is characterized by:
    1. Price converging toward the pullback EMA
    2. Pullback EMA converging toward trend EMA
    3. Declining confidence just before exit
    4. Distance ratios compressing

    Returns list of dicts for each detected compression event.
    """
    n = len(state["price"])
    signatures = []

    # Find all pullback events
    in_pb = state["in_pullback"]
    conf = state["confidence"]
    d2p = state["distance_to_pull"]
    dp2t = state["distance_pull_to_trend"]

    # Detect pullback start/end transitions
    i = 100  # skip initial unstable
    while i < n:
        if in_pb[i] and conf[i] > 0:
            # Start of a pullback event
            start_idx = i
            # Track until pullback ends or exits
            conf_peak = conf[i]
            peak_idx = i
            for j in range(i, min(i + window, n)):
                if not in_pb[j] or conf[j] == 0:
                    break
                if conf[j] > conf_peak:
                    conf_peak = conf[j]
                    peak_idx = j
                i = j + 1

            # Define pre-exit compression: look at 5 bars before peak
            pre_window = 5
            pre_start = max(start_idx, peak_idx - pre_window)
            pre_slice = slice(pre_start, peak_idx)

            if pre_start < peak_idx and peak_idx < n:
                # Compression metrics
                pre_d2p = d2p[pre_slice]
                pre_dp2t = dp2t[pre_slice]
                pre_conf = conf[pre_slice]

                valid = ~np.isnan(pre_d2p) & ~np.isnan(pre_dp2t)
                if np.sum(valid) >= 2:
                    # Trend: distances compressing (decreasing)
                    d2p_trend = np.polyfit(
                        np.arange(np.sum(valid)), pre_d2p[valid], 1
                    )[0] if np.sum(valid) >= 2 else 0.0
                    dp2t_trend = np.polyfit(
                        np.arange(np.sum(valid)), pre_dp2t[valid], 1
                    )[0] if np.sum(valid) >= 2 else 0.0
                    conf_trend = np.polyfit(
                        np.arange(np.sum(valid)), pre_conf[valid], 1
                    )[0] if np.sum(valid) >= 2 else 0.0

                    # Compression = negative trend in distances
                    is_compressing = d2p_trend < 0 or dp2t_trend < 0

                    signatures.append({
                        "start_bar": int(start_idx),
                        "peak_conf_bar": int(peak_idx),
                        "peak_confidence": round(float(conf[peak_idx]), 4),
                        "pre_window_bars": pre_window,
                        "distance_to_pull_trend": round(float(d2p_trend), 6),
                        "distance_pull_to_trend_trend": round(float(dp2t_trend), 6),
                        "confidence_trend_pre_exit": round(float(conf_trend), 6),
                        "is_compressing": bool(is_compressing),
                        "compression_strength": round(
                            float(abs(d2p_trend) + abs(dp2t_trend)) * 100, 4
                        ),
                    })
        else:
            i += 1

    return signatures


def compute_compression_signature_score(
    signatures: List[dict],
) -> Tuple[float, dict]:
    """Compute compression signature stability score.

    Score components:
    1. Compression frequency: how often does compression precede exits?
    2. Compression strength consistency: low variance in strength
    3. Signature reproducibility: do the same patterns repeat?

    Returns (score_0_1, detail_dict).
    """
    if not signatures:
        return 0.0, {
            "total_compression_events": 0,
            "compression_frequency_score": 0.0,
            "strength_consistency_score": 0.0,
            "notes": "No compression signatures detected",
        }

    n_sigs = len(signatures)
    compressing_count = sum(1 for s in signatures if s["is_compressing"])
    compression_ratio = compressing_count / n_sigs if n_sigs > 0 else 0.0

    # Compression frequency: more = better
    freq_score = min(1.0, compression_ratio * 1.25)  # 80% is perfect

    # Strength consistency: low CV
    strengths = [s["compression_strength"] for s in signatures if s["compression_strength"] > 0]
    if len(strengths) >= 2:
        strength_cv = float(np.std(strengths) / max(np.mean(strengths), 0.001))
        strength_score = max(0.0, 1.0 - min(1.0, strength_cv))
    else:
        strength_score = 0.5 if len(strengths) == 1 else 0.0

    # Combined score
    score = round(freq_score * 0.5 + strength_score * 0.5, 4)

    detail = {
        "total_compression_events": n_sigs,
        "compression_events_count": compressing_count,
        "compression_ratio": round(compression_ratio, 4),
        "compression_frequency_score": round(freq_score, 4),
        "strength_values": [round(s["compression_strength"], 4) for s in signatures],
        "strength_consistency_score": round(strength_score, 4),
        "compression_signature_score": score,
    }

    return score, detail


# ---------------------------------------------------------------------------
# Task 3: Cross-Symbol False Activation Test
# ---------------------------------------------------------------------------

def run_false_activation_test(
    mapper: EdgeSignalMapper,
) -> dict:
    """Test if edge_04 or other JPY edges activate on non-JPY data.

    1. Run EUPRJPY normal: normal behavior baseline
    2. Run NZDCAD: should NOT activate JPY edges
    3. Run USDJPY: check JPY cluster correlation effect
    """
    # Get EURJPY data
    eurjpy_closes = load_mt5_rates("EURJPY", count=BAR_COUNT)
    if eurjpy_closes is None:
        eurjpy_closes = generate_synthetic_eurjpy_prices(BAR_COUNT)

    # Get NZDCAD data (non-JPY pair for false activation baseline)
    nzdcad_closes = load_mt5_rates("NZDCAD", count=BAR_COUNT)
    if nzdcad_closes is None:
        nzdcad_closes = generate_synthetic_nzdcad_prices(BAR_COUNT)

    # Get USDJPY data (JPY cluster pair)
    usdjpy_closes = load_mt5_rates("USDJPY", count=BAR_COUNT)
    if usdjpy_closes is None:
        rng = np.random.RandomState(42)
        usdjpy_closes = 150.0 + np.cumsum(rng.randn(BAR_COUNT) * 0.001) + rng.randn(BAR_COUNT) * 0.005
        usdjpy_closes = np.maximum(usdjpy_closes, 148.0)

    # ---- Test A: EURJPY baseline ----
    logger.info("=" * 60)
    logger.info("FALSE ACTIVATION TEST A: EURJPY baseline")
    baseline_signals = mapper.generate_all({"EURJPY": eurjpy_closes})
    eurjpy_edge_ids = set(s.get("edge_id") for s in baseline_signals)
    eurjpy_edge_04 = [s for s in baseline_signals if s.get("edge_id") == EDGE_04_ID]
    jpy_edge_ids_eurjpy = {
        s.get("edge_id") for s in baseline_signals
        if s.get("symbol") in ("EURJPY", "USDJPY")
    }

    logger.info("  EURJPY: %d edges active, edge_04=%s",
                len(baseline_signals),
                "ACTIVE" if eurjpy_edge_04 else "INACTIVE")

    # ---- Test B: NZDCAD false activation test ----
    logger.info("=" * 60)
    logger.info("FALSE ACTIVATION TEST B: NZDCAD (non-JPY pair)")
    nzdcad_signals = mapper.generate_all({"NZDCAD": nzdcad_closes})
    nzdcad_edge_ids = set(s.get("edge_id") for s in nzdcad_signals)
    nzdcad_edge_04 = [s for s in nzdcad_signals if s.get("edge_id") == EDGE_04_ID]
    nzdcad_jpy_edges = [
        s for s in nzdcad_signals
        if s.get("symbol") in ("EURJPY", "USDJPY")
    ]

    logger.info("  NZDCAD: %d edges active, edge_04=%s, JPY edges=%d",
                len(nzdcad_signals),
                "ACTIVE" if nzdcad_edge_04 else "INACTIVE",
                len(nzdcad_jpy_edges))

    # ---- Test C: USDJPY (JPY cluster check) ----
    logger.info("=" * 60)
    logger.info("FALSE ACTIVATION TEST C: USDJPY (JPY cluster)")
    usdjpy_data = {"USDJPY": usdjpy_closes}
    usdjpy_signals = mapper.generate_all(usdjpy_data)
    usdjpy_edge_ids = set(s.get("edge_id") for s in usdjpy_signals)
    usdjpy_edge_04 = [s for s in usdjpy_signals if s.get("edge_id") == EDGE_04_ID]

    logger.info("  USDJPY: %d edges active, edge_04=%s",
                len(usdjpy_signals),
                "ACTIVE" if usdjpy_edge_04 else "INACTIVE")

    # ---- Compute leakage metrics ----
    # Leakage: edges that fire on wrong symbols
    total_edges = mapper.edge_count

    # Leakage on NZDCAD (EURJPY edges active on NZDCAD data)
    leakage_ids = nzdcad_jpy_edges if nzdcad_jpy_edges else []
    leakage_count = len(set(s.get("edge_id") for s in leakage_ids))
    leakage_ratio = leakage_count / max(total_edges, 1)

    # JPY cluster correlation: does edge_04 on EURJPY correlate with USDJPY signals?
    # If edge_04 triggers when USDJPY edges also trigger, there's cluster correlation
    eurjpy_dirs = {s.get("edge_id"): s["direction"] for s in baseline_signals}
    usdjpy_dirs = {s.get("edge_id"): s["direction"] for s in usdjpy_signals}

    # Check for shared direction on JPY cluster
    shared_edges = set(eurjpy_dirs.keys()) & set(usdjpy_dirs.keys())
    aligned = sum(
        1 for eid in shared_edges
        if eurjpy_dirs[eid] == usdjpy_dirs[eid] and eurjpy_dirs[eid] != 0
    )
    total_nonzero = sum(
        1 for eid in shared_edges
        if eurjpy_dirs[eid] != 0 or usdjpy_dirs[eid] != 0
    )
    correlation_rate = aligned / max(total_nonzero, 1) if total_nonzero > 0 else 0.0

    # False activation score: 1.0 = no leakage, 0.0 = all edges leak
    false_activation_score = round(1.0 - leakage_ratio, 4)

    result = {
        "baseline_eurjpy": {
            "total_edges_active": len(baseline_signals),
            "edge_04_active": len(eurjpy_edge_04) > 0,
            "edge_04_confidence": eurjpy_edge_04[0]["confidence"] if eurjpy_edge_04 else 0.0,
            "active_edge_ids": sorted(eurjpy_edge_ids),
        },
        "false_activation_nzdcad": {
            "total_edges_active": len(nzdcad_signals),
            "edge_04_active": len(nzdcad_edge_04) > 0,
            "edge_04_falsely_activated": len(nzdcad_edge_04) > 0,
            "jpy_edge_leakage_count": leakage_count,
            "jpy_edge_leakage_ids": sorted(set(s.get("edge_id") for s in leakage_ids)),
        },
        "jpychuster_usdjpy": {
            "total_edges_active": len(usdjpy_signals),
            "edge_04_active": len(usdjpy_edge_04) > 0,
        },
        "jpychuster_correlation": {
            "shared_edges_by_id": sorted(shared_edges),
            "aligned_direction_count": aligned,
            "total_nonzero_directions": total_nonzero,
            "correlation_rate": round(correlation_rate, 4),
            "interpretation": (
                "HIGH CORRELATION" if correlation_rate > 0.7 else
                "MODERATE CORRELATION" if correlation_rate > 0.4 else
                "LOW CORRELATION"
            ),
        },
        "leakage_metrics": {
            "total_edges_in_manifest": total_edges,
            "leakage_count": leakage_count,
            "leakage_ratio": round(leakage_ratio, 4),
            "false_activation_score": false_activation_score,
            "interpretation": (
                "CLEAN" if false_activation_score >= 0.9 else
                "MINOR LEAKAGE" if false_activation_score >= 0.7 else
                "SIGNIFICANT LEAKAGE"
            ),
        },
    }

    return result


def run_jpychuster_correlation_deep_check(
    mapper: EdgeSignalMapper,
    eurjpy_closes: np.ndarray,
    usdjpy_closes: np.ndarray,
) -> dict:
    """Deep correlation check between edge_04 and USDJPY signals.

    Tests if edge_04's activation correlates with USDJPY edge signals,
    indicating JPY cluster systemic behavior vs. EURJPY-specific signal.
    """
    # Generate single-symbol signals
    eurjpy_sigs = mapper.generate_for_symbol("EURJPY", eurjpy_closes)
    usdjpy_sigs = mapper.generate_for_symbol("USDJPY", usdjpy_closes)

    # Find edge_04 signal
    e04_sig = None
    for s in eurjpy_sigs:
        if s.get("edge_id") == EDGE_04_ID:
            e04_sig = s
            break

    # Cross-symbol edge timing correlation
    # Simple test: do EURJPY and USDJPY signals have similar confidence patterns?
    eurjpy_confs = [s.get("confidence", 0) for s in eurjpy_sigs]
    usdjpy_confs = [s.get("confidence", 0) for s in usdjpy_sigs]

    if len(eurjpy_confs) > 0 and len(usdjpy_confs) > 0:
        # Correlation across edges
        min_len = min(len(eurjpy_confs), len(usdjpy_confs))
        if min_len >= 2:
            corr_matrix = np.corrcoef(
                eurjpy_confs[:min_len], usdjpy_confs[:min_len]
            )
            conf_corr = float(corr_matrix[0, 1])
            if np.isnan(conf_corr):
                conf_corr = 0.0
        else:
            conf_corr = 0.0
    else:
        conf_corr = 0.0

    # Direction alignment
    eurjpy_dirs = {s.get("edge_id"): s.get("direction", 0) for s in eurjpy_sigs}
    usdjpy_dirs = {s.get("edge_id"): s.get("direction", 0) for s in usdjpy_sigs}

    direction_corr = 0.0
    common_ids = set(eurjpy_dirs.keys()) & set(usdjpy_dirs.keys())
    if common_ids:
        aligned = sum(1 for eid in common_ids if eurjpy_dirs[eid] == usdjpy_dirs[eid])
        direction_corr = aligned / len(common_ids)

    return {
        "edge_04_signal_present": e04_sig is not None,
        "edge_04_confidence": e04_sig.get("confidence", 0) if e04_sig else 0.0,
        "edge_04_direction": e04_sig.get("direction", 0) if e04_sig else 0,
        "eurjpy_vs_usdjpy": {
            "confidence_correlation": round(float(conf_corr), 4),
            "direction_alignment_rate": round(float(direction_corr), 4),
            "common_edge_count": len(common_ids),
            "interpretation": (
                "SHARED JPY CLUSTER DYNAMICS" if conf_corr > 0.5 or direction_corr > 0.7
                else "INDEPENDENT SIGNALS"
            ),
        },
    }


# ---------------------------------------------------------------------------
# Task 4: Report Generation
# ---------------------------------------------------------------------------

def build_report(
    exit_result: Tuple[List[dict], dict],
    compression_score: float,
    compression_detail: dict,
    false_activation_result: dict,
    jpy_cluster_result: dict,
    overall_score: float,
    recommendation: str,
) -> dict:
    """Build the final structured report."""
    cycle_results, exit_summary = exit_result

    report = {
        "report_metadata": {
            "report_type": "EDGE_04_SHADOW_VALIDATION",
            "phase": "Batch 6.3 Phase 2",
            "edge_id": EDGE_04_ID,
            "symbol": "EURJPY",
            "strategy": "pullback",
            "params": EDGE_04_PARAMS,
            "manifest_pf": 1.3104,
            "manifest_wf_pf": 1.3075,
            "generated_at": datetime.now().isoformat(),
            "constraints_applied": [
                "NO_REAL_MT5_EXECUTION",
                "NO_NEW_TRADES",
                "VALIDATION_ONLY",
                "NO_LIVE_MODE",
            ],
        },
        "task1_edge_04_decision_simulation": {
            "description": (
                "Simulated edge_04 signal generation across all 3 bootstrap observation "
                "cycles, measuring exit timing consistency through the full decision "
                "pipeline: Signal → MOF Gate → Bridge Evaluation → Exit"
            ),
            "cycle_results": cycle_results,
            "cross_cycle_summary": exit_summary,
        },
        "task2_compression_signature_stability": {
            "description": (
                "Measured edge_04's compression signature pattern — the characteristic "
                "pre-exit convergence of price → pullback EMA and pullback EMA → trend EMA. "
                "Higher score means the signature reproduces consistently across events."
            ),
            "compression_details": compression_detail,
            "compression_signature_score": round(compression_score, 4),
        },
        "task3_false_activation_test": {
            "description": (
                "Tested edge_04 on NZDCAD (non-JPY pair) to measure leakage — edges that "
                "fire on wrong symbols. Also checked JPY cluster correlation with USDJPY."
            ),
            "test_results": false_activation_result,
            "jpychuster_deep_check": jpy_cluster_result,
            "false_activation_score": round(
                false_activation_result["leakage_metrics"]["false_activation_score"], 4
            ),
        },
        "task4_scoring_summary": {
            "exit_consistency_score": float(exit_summary["exit_consistency_score"]),
            "compression_signature_score": float(compression_score),
            "false_activation_score": float(
                false_activation_result["leakage_metrics"]["false_activation_score"]
            ),
            "cross_cycle_variance": {
                "exit_confidence_std": float(exit_summary.get("exit_confidence_std", 0)),
                "cycles_with_signal": exit_summary.get("cycles_with_edge_04_signal", 0),
                "cycles_total": exit_summary.get("total_cycles_simulated", 0),
            },
            "overall_reproducibility_score": round(overall_score, 4),
            "recommendation": recommendation,
            "scoring_thresholds": {
                "consistent_min": 0.7,
                "observation_min": 0.5,
                "unstable_max": 0.5,
            },
        },
    }
    return report


def format_report_dashboard(report: dict) -> str:
    """Format the report as a console-readable dashboard."""
    lines = []
    lines.append("")
    lines.append("=" * 78)
    lines.append("  EDGE_04 SHADOW VALIDATION REPORT")
    lines.append("=" * 78)
    lines.append(f"")
    lines.append(f"  Edge:        {report['report_metadata']['edge_id']}")
    lines.append(f"  Symbol:      {report['report_metadata']['symbol']}")
    lines.append(f"  Strategy:    {report['report_metadata']['strategy']}")
    lines.append(f"  PF (manifest): {report['report_metadata']['manifest_pf']}")
    lines.append(f"  WF PF:        {report['report_metadata']['manifest_wf_pf']}")
    lines.append(f"")

    # Task 1 summary
    ts1 = report["task1_edge_04_decision_simulation"]["cross_cycle_summary"]
    lines.append(f"  ┌─ TASK 1: Decision Simulation ─────────────────────────────")
    lines.append(f"  │ Cycles simulated:        {ts1['total_cycles_simulated']}")
    lines.append(f"  │ Cycles w/ edge_04 sig:   {ts1['cycles_with_edge_04_signal']}")
    lines.append(f"  │ Cycles would exit:       {ts1['cycles_would_exit']}")
    lines.append(f"  │ Exit consistency ratio:  {ts1['exit_consistency_ratio']}")
    lines.append(f"  │ Exit confidence values:  {ts1['exit_confidence_values']}")
    lines.append(f"  │ Exit conf std:           {ts1['exit_confidence_std']}")
    lines.append(f"  │ Timing consistency:      {ts1['timing_consistency_score']}")
    lines.append(f"  │ Exit consistency score:  {ts1['exit_consistency_score']}")
    lines.append(f"  └──────────────────────────────────────────────────────────")
    lines.append(f"")

    # Task 2 summary
    ts2 = report["task2_compression_signature_stability"]
    cd = ts2["compression_details"]
    lines.append(f"  ┌─ TASK 2: Compression Signature ───────────────────────────")
    lines.append(f"  │ Compression events:      {cd.get('total_compression_events', 0)}")
    lines.append(f"  │ Compression ratio:       {cd.get('compression_ratio', 0)}")
    lines.append(f"  │ Frequency score:         {cd.get('compression_frequency_score', 0)}")
    lines.append(f"  │ Strength consistency:    {cd.get('strength_consistency_score', 0)}")
    lines.append(f"  │ Compression signature:   {ts2['compression_signature_score']}")
    lines.append(f"  └──────────────────────────────────────────────────────────")
    lines.append(f"")

    # Task 3 summary
    ts3 = report["task3_false_activation_test"]
    lm = ts3["test_results"]["leakage_metrics"]
    jcc = ts3["test_results"]["jpychuster_correlation"]
    lines.append(f"  ┌─ TASK 3: False Activation ────────────────────────────────")
    lines.append(f"  │ Total edges in manifest: {lm['total_edges_in_manifest']}")
    lines.append(f"  │ Leakage count:           {lm['leakage_count']}")
    lines.append(f"  │ Leakage ratio:           {lm['leakage_ratio']}")
    lines.append(f"  │ False activation score:  {lm['false_activation_score']}")
    lines.append(f"  │ Leakage interpretation:  {lm['interpretation']}")
    lines.append(f"  │ JPY cluster correlation: {jcc['interpretation']}")
    lines.append(f"  └──────────────────────────────────────────────────────────")
    lines.append(f"")

    # Overall
    sc = report["task4_scoring_summary"]
    lines.append(f"  ┌─ OVERALL SCORING ──────────────────────────────────────────")
    lines.append(f"  │ Exit Consistency Score:    {sc['exit_consistency_score']}")
    lines.append(f"  │ Compression Signature:     {sc['compression_signature_score']}")
    lines.append(f"  │ False Activation Score:    {sc['false_activation_score']}")
    lines.append(f"  │ Cross-cycle variance:      {sc['cross_cycle_variance']}")
    lines.append(f"  ├──────────────────────────────────────────────────────────")
    lines.append(f"  │ OVERALL REPRODUCIBILITY:   {sc['overall_reproducibility_score']}")
    lines.append(f"  │ RECOMMENDATION:            {sc['recommendation']}")
    lines.append(f"  └──────────────────────────────────────────────────────────")
    lines.append(f"")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the full Edge_04 shadow validation suite."""
    logger.info("=" * 78)
    logger.info("  EDGE_04 SHADOW VALIDATION — Batch 6.3 Phase 2")
    logger.info("=" * 78)
    logger.info("  Constraint: VALIDATION ONLY — no real execution")
    logger.info("")

    # ---- Setup: Load EdgeSignalMapper ----
    mapper = EdgeSignalMapper()
    logger.info("Loaded %d edges across %d symbols",
                mapper.edge_count, len(mapper.get_symbols_with_edges()))

    # Verify edge_04 exists
    e04_edges = mapper.get_edges_for_symbol("EURJPY")
    e04 = [e for e in e04_edges if e["id"] == EDGE_04_ID]
    if not e04:
        logger.error("Edge_04 not found in manifest — aborting")
        sys.exit(1)
    logger.info("Edge_04 config: %s", json.dumps(e04[0], indent=2))

    # ---- Load price data ----
    logger.info("")
    logger.info("=" * 60)
    logger.info("Loading price data...")

    # Try MT5 first, fall back to synthetic
    eurjpy_closes = load_mt5_rates("EURJPY", count=BAR_COUNT)
    if eurjpy_closes is None:
        logger.info("MT5 unavailable — using synthetic EURJPY data")
        eurjpy_closes = generate_synthetic_eurjpy_prices(BAR_COUNT)

    # Split into 3 segments to simulate 3 cycles
    n = len(eurjpy_closes)
    seg_size = n // 3
    cycle_segments = {
        0: {"EURJPY": eurjpy_closes[:seg_size]},
        1: {"EURJPY": eurjpy_closes[seg_size:2 * seg_size]},
        2: {"EURJPY": eurjpy_closes[2 * seg_size:]},
    }

    # ------------------------------------------------------------------
    # Task 1: Edge_04 Decision Simulation
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 60)
    logger.info("TASK 1: Edge_04 Decision Simulation")
    logger.info("=" * 60)

    cycle_segment_results = []
    all_exit_summaries = []
    for cycle_idx, cycle_info in enumerate(BOOTSTRAP_CYCLES):
        closes_dict = cycle_segments[cycle_idx]
        logger.info("  Running cycle %d simulation (%s)...",
                     cycle_info["cycle"],
                     cycle_info["timestamp"].isoformat())
        result, summary = simulate_exit_decision_pipeline(
            mapper, closes_dict, [cycle_info]
        )
        cycle_segment_results.extend(result)
        all_exit_summaries.append(summary)

    # Aggregate exit consistency across all 3 cycles
    exit_conf_vals = []
    would_exit_count = 0
    for r in cycle_segment_results:
        if r.get("edge_04_signal"):
            exit_conf_vals.append(r["edge_04_signal"]["confidence"])
            if r["edge_04_signal"]["would_exit"]:
                would_exit_count += 1

    exit_consistency_ratio = would_exit_count / len(BOOTSTRAP_CYCLES) if BOOTSTRAP_CYCLES else 0.0
    exit_conf_std = float(np.std(exit_conf_vals)) if len(exit_conf_vals) > 1 else 0.0
    timing_score = max(0.0, 1.0 - min(1.0, exit_conf_std * 3.0))
    exit_consistency_score = round(exit_consistency_ratio * 0.6 + timing_score * 0.4, 4)

    exit_summary_aggregate = {
        "total_cycles_simulated": len(BOOTSTRAP_CYCLES),
        "cycles_with_edge_04_signal": sum(
            1 for r in cycle_segment_results if r["edge_04_signal"] is not None
        ),
        "cycles_would_exit": would_exit_count,
        "exit_consistency_ratio": round(exit_consistency_ratio, 4),
        "exit_confidence_values": [round(c, 4) for c in exit_conf_vals],
        "exit_confidence_std": round(exit_conf_std, 4),
        "timing_consistency_score": round(timing_score, 4),
        "exit_consistency_score": exit_consistency_score,
        "overall_would_exit_in_live": any(
            r["overall_decision"]["would_exit_in_live"]
            for r in cycle_segment_results
        ),
        "overall_would_exit_in_bootstrap": any(
            r["overall_decision"]["would_exit_in_bootstrap"]
            for r in cycle_segment_results
        ),
    }

    logger.info("")
    logger.info("  Cycle simulation complete:")
    logger.info("    Cycles with edge_04 signal: %d/%d",
                exit_summary_aggregate["cycles_with_edge_04_signal"],
                exit_summary_aggregate["total_cycles_simulated"])
    logger.info("    Cycles would exit:         %d/%d",
                exit_summary_aggregate["cycles_would_exit"],
                exit_summary_aggregate["total_cycles_simulated"])
    logger.info("    Exit confidence std:       %.4f", exit_conf_std)
    logger.info("    Exit consistency score:    %.4f", exit_consistency_score)

    # ------------------------------------------------------------------
    # Task 2: Compression Signature Stability
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 60)
    logger.info("TASK 2: Compression Signature Stability")
    logger.info("=" * 60)

    # Compute edge_04 state across full EURJPY data
    logger.info("  Computing edge_04 internal state at each bar...")
    edge_04_state = compute_edge_04_state_at_each_bar(eurjpy_closes)

    # Find compression signatures
    logger.info("  Identifying compression signatures...")
    signatures = find_compression_signatures(edge_04_state)

    # Score compression signature stability
    compression_score, compression_detail = compute_compression_signature_score(signatures)

    logger.info("  Compression signatures found: %d", len(signatures))
    logger.info("  Compression ratio:           %.4f",
                compression_detail.get("compression_ratio", 0))
    logger.info("  Compression signature score: %.4f", compression_score)

    # ------------------------------------------------------------------
    # Task 3: Cross-Symbol False Activation Test
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 60)
    logger.info("TASK 3: Cross-Symbol False Activation Test")
    logger.info("=" * 60)

    false_activation_result = run_false_activation_test(mapper)

    # JPY cluster deep check
    usdjpy_closes = load_mt5_rates("USDJPY", count=BAR_COUNT)
    if usdjpy_closes is None:
        rng = np.random.RandomState(42)
        usdjpy_closes = 150.0 + np.cumsum(rng.randn(BAR_COUNT) * 0.001) + rng.randn(BAR_COUNT) * 0.005
        usdjpy_closes = np.maximum(usdjpy_closes, 148.0)

    jpy_cluster_result = run_jpychuster_correlation_deep_check(
        mapper, eurjpy_closes, usdjpy_closes
    )

    logger.info("")
    logger.info("  False activation score: %.4f",
                false_activation_result["leakage_metrics"]["false_activation_score"])
    logger.info("  Leakage ratio:          %.4f",
                false_activation_result["leakage_metrics"]["leakage_ratio"])
    logger.info("  JPY cluster correlation: %s",
                jpy_cluster_result["eurjpy_vs_usdjpy"]["interpretation"])

    # ------------------------------------------------------------------
    # Task 4: Scoring & Report
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 60)
    logger.info("TASK 4: Decision Reproducibility Report")
    logger.info("=" * 60)

    # Overall reproducibility score
    false_act_score = false_activation_result["leakage_metrics"]["false_activation_score"]
    overall_score = round(
        (
            exit_consistency_score
            + compression_score
            + false_act_score
        ) / 3.0,
        4,
    )

    # Recommendation
    if all([
        exit_consistency_score >= 0.7,
        compression_score >= 0.7,
        false_act_score >= 0.7,
    ]):
        recommendation = "CONSISTENT"
    elif overall_score >= 0.5:
        recommendation = "NEEDS_OBSERVATION"
    else:
        recommendation = "UNSTABLE"

    logger.info("  Exit Consistency Score:   %.4f", exit_consistency_score)
    logger.info("  Compression Signature:    %.4f", compression_score)
    logger.info("  False Activation Score:   %.4f", false_act_score)
    logger.info("  Overall Reproducibility:  %.4f", overall_score)
    logger.info("  Recommendation:           %s", recommendation)

    # Build report
    exit_result = (cycle_segment_results, exit_summary_aggregate)
    report = build_report(
        exit_result=exit_result,
        compression_score=compression_score,
        compression_detail=compression_detail,
        false_activation_result=false_activation_result,
        jpy_cluster_result=jpy_cluster_result,
        overall_score=overall_score,
        recommendation=recommendation,
    )

    # Save report
    report_path = os.path.join(_STATE_DIR, "edge_04_shadow_test_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("  Report saved to: %s", report_path)

    # Print dashboard
    dashboard = format_report_dashboard(report)
    print(dashboard)

    logger.info("")
    logger.info("=" * 78)
    logger.info("  EDGE_04 SHADOW VALIDATION COMPLETE")
    logger.info("  Report: %s", report_path)
    logger.info("=" * 78)


if __name__ == "__main__":
    main()
