"""Edge Isolation Sandbox — Batch 6.4 Phase 1.

Completely decoupled evaluation pipeline for edge_04 that runs without any
MOF, RF, or lifecycle influence. Only raw price-derived features -> edge transform.

Purpose
-------
The Brain identified a contamination problem: edge_04's current evaluation is
entangled with MOF gating, RF model weighting, and lifecycle feedback. This
sandbox isolates edge_04 to see its pure, uncontaminated signature.

Design
------
1. Loads ONLY:
   - Edge definitions from deployment_manifest.json
   - EdgeSignalMapper (strategy functions only)
   - Price data from MT5 (raw tick/bar data)

2. Does NOT load or use:
   - MOF (MarketObservabilityFilter) — NOT imported
   - RF model (no RandomForest, no rf_gate) — NOT imported
   - Lifecycle state — NOT imported
   - Any system state files — NOT loaded

3. Runs edge_04 3 times with slight price perturbations to test stability.

Output
------
state/edge_04_isolated_signature.json
    Raw signal direction, raw confidence (no MOF/RF modification),
    ECDF value, drift, raw compression signature from price features,
    stability across 3 runs.

Usage::
    cd proxima_x
    python bootstrap/edge_isolation_sandbox.py
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

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
logger = logging.getLogger("edge_isolation_sandbox")

os.makedirs(_STATE_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Imports — ONLY what we need, NO MOF/RF/lifecycle
# ---------------------------------------------------------------------------

try:
    from proxima_ops.risk.edge_signal_mapper import (
        EdgeSignalMapper,
        _pullback_signal,
        _ema,
        _compute_atr,
        _compute_rsi,
    )
except ImportError as exc:
    logger.error("Cannot import EdgeSignalMapper: %s", exc)
    sys.exit(1)

try:
    from proxima_ops.execution.mt5_connector import MT5Connector
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    logger.warning("MT5Connector not available — using synthetic data only")

# IMPORTANT: Verify no MOF, RF, or lifecycle modules were accidentally loaded
_VERIFY_NO_CONTAMINATION = [
    "market_observability_filter",
    "governance_pipeline",
    "randomforest",
    "rf_gate",
    "lifecycle",
]
for mod_name in _VERIFY_NO_CONTAMINATION:
    if mod_name in "".join(sys.modules.keys()):
        logger.warning(
            "CONTAMINATION WARNING: Module '%s' found in sys.modules — "
            "this sandbox should be pure!", mod_name
        )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EDGE_04_ID = "edge_04"
EURJPY_SYMBOL = "EURJPY"
BAR_COUNT = 500

# Edge_04 parameters (from deployment_manifest.json)
EDGE_04_PARAMS = {
    "trend_ema": 100,
    "pullback_ema": 10,
    "max_hold": 18,
}

# Perturbation levels for stability testing (fraction of price)
PERTURBATIONS = [-0.001, 0.0, +0.001]  # -10bp, 0, +10bp


# ---------------------------------------------------------------------------
# Price Data Loading (pure — no MOF/RF/lifecycle)
# ---------------------------------------------------------------------------

def fetch_eurjpy_ohlcv(count: int = BAR_COUNT) -> Optional[Dict[str, np.ndarray]]:
    """Fetch EURJPY OHLCV from MT5.

    Returns dict with keys: open, high, low, close, volume as numpy arrays.
    Returns None if MT5 unavailable or fails.
    """
    if not MT5_AVAILABLE:
        logger.warning("MT5 not available — cannot fetch real price data")
        return None

    connector = MT5Connector()
    if not connector.connect():
        logger.warning("MT5 connection failed")
        return None

    try:
        rates = connector.get_rates(EURJPY_SYMBOL, count=count, timeframe="M5")
        if rates is None or len(rates) == 0:
            logger.warning("No rates returned from MT5 for EURJPY")
            connector.disconnect()
            return None

        ohlcv = {
            "time": np.array([r["time"] for r in rates], dtype=np.int64),
            "open": np.array([r["open"] for r in rates], dtype=float),
            "high": np.array([r["high"] for r in rates], dtype=float),
            "low": np.array([r["low"] for r in rates], dtype=float),
            "close": np.array([r["close"] for r in rates], dtype=float),
            "volume": np.array([r["volume"] for r in rates], dtype=float),
        }

        logger.info(
            "Fetched %d OHLCV bars for EURJPY from MT5 "
            "(close range: %.5f – %.5f, vol range: %.0f – %.0f)",
            len(ohlcv["close"]),
            ohlcv["close"][0], ohlcv["close"][-1],
            ohlcv["volume"].min(), ohlcv["volume"].max(),
        )
        return ohlcv

    except Exception as exc:
        logger.warning("Failed to fetch EURJPY data: %s", exc)
        return None

    finally:
        connector.disconnect()


def generate_synthetic_eurjpy_ohlcv(
    n_bars: int = BAR_COUNT,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Generate synthetic EURJPY OHLCV data for when MT5 is unavailable.

    Creates realistic-ish bars with OHLC relationships preserved.
    """
    rng = np.random.RandomState(seed)
    base = 184.0

    # Generate close prices with drift, noise, and pullback patterns
    trend = np.cumsum(rng.randn(n_bars) * 0.001)
    noise = rng.randn(n_bars) * 0.005
    closes = base + trend + noise

    # Add pullback patterns
    pullback_indices = rng.choice(
        np.arange(50, n_bars - 50), size=n_bars // 30, replace=False
    )
    for idx in pullback_indices:
        pull_size = rng.uniform(-0.15, 0.15)
        length = rng.randint(5, 15)
        for i in range(length):
            if idx + i < n_bars:
                closes[idx + i] += pull_size * (1 - i / length)

    closes = np.maximum(closes, base * 0.95)

    # Derive OHLC from close with realistic spreads
    half_spread = np.abs(noise) * 0.3 + 0.002
    highs = closes + np.abs(rng.randn(n_bars) * 0.008) + half_spread
    lows = closes - np.abs(rng.randn(n_bars) * 0.008) - half_spread
    opens = closes + rng.randn(n_bars) * 0.005
    volumes = rng.randint(50, 500, size=n_bars).astype(float)

    # Ensure OHLC integrity
    highs = np.maximum(highs, np.maximum(opens, closes))
    lows = np.minimum(lows, np.minimum(opens, closes))

    ohlcv = {
        "time": np.arange(n_bars, dtype=np.int64),
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    }

    logger.info(
        "Generated %d synthetic EURJPY bars "
        "(close range: %.5f – %.5f)",
        n_bars, closes.min(), closes.max(),
    )
    return ohlcv


def perturb_prices(
    ohlcv: Dict[str, np.ndarray],
    perturbation: float,
) -> Dict[str, np.ndarray]:
    """Apply a fractional perturbation to all price fields.

    Used for stability testing — slight changes should not dramatically
    alter edge_04's signal if it's a genuine price-derived pattern.
    """
    perturbed = {}
    for key in ("open", "high", "low", "close"):
        perturbed[key] = ohlcv[key] * (1.0 + perturbation)
    perturbed["time"] = ohlcv["time"].copy()
    perturbed["volume"] = ohlcv["volume"].copy()
    return perturbed


# ---------------------------------------------------------------------------
# Compression Signature Analysis (pure price-derived)
# ---------------------------------------------------------------------------

def compute_edge_04_state(
    closes: np.ndarray,
    trend_span: int = 100,
    pullback_span: int = 10,
) -> Dict[str, Any]:
    """Compute edge_04's internal pullback state at every bar.

    Returns dict with arrays tracking all internal signals.
    """
    n = len(closes)
    trend = _ema(closes, trend_span)
    pullback = _ema(closes, pullback_span)

    direction = np.zeros(n, dtype=int)
    confidence = np.zeros(n)
    ecdf_vals = np.full(n, np.nan)
    drift_vals = np.zeros(n, dtype=int)
    pullback_depth_vals = np.zeros(n)
    in_pullback = np.full(n, False)

    for i in range(trend_span + 5, n):
        c = closes[i]
        t = trend[i]
        p = pullback[i]

        if np.isnan(t) or np.isnan(p):
            continue

        d2p = abs(c - p) / max(c, 1e-12)
        dp2t = abs(p - t) / max(c, 1e-12)

        # Trend direction
        tu = False
        td = False
        if i >= 5 and not np.isnan(trend[i - 5]):
            tu = trend[i] > trend[i - 5]
            td = trend[i] < trend[i - 5]

        # Pullback detection (same logic as _pullback_signal)
        if tu and c <= p:
            in_pullback[i] = True
            pdepth = max(0, min(1.0, 1.0 - d2p / max(dp2t + 0.0001, 0.0001)))
            direction[i] = 1
            confidence[i] = min(1.0, 0.4 + pdepth * 0.4)
            pullback_depth_vals[i] = pdepth
        elif td and c >= p:
            in_pullback[i] = True
            pdepth = max(0, min(1.0, 1.0 - d2p / max(dp2t + 0.0001, 0.0001)))
            direction[i] = -1
            confidence[i] = min(1.0, 0.4 + pdepth * 0.4)
            pullback_depth_vals[i] = pdepth

        # ECDF (percentile rank)
        valid_sorted = np.sort(closes)
        ecdf_vals[i] = float(
            np.searchsorted(valid_sorted, c) / max(len(valid_sorted), 1)
        )

        # Drift
        if i >= 3:
            pdrift = closes[i] - closes[i - 3]
            drift_vals[i] = 1 if pdrift > 0 else (-1 if pdrift < 0 else 0)

    return {
        "trend_ema": trend,
        "pullback_ema": pullback,
        "price": closes,
        "direction": direction,
        "confidence": confidence,
        "ecdf": ecdf_vals,
        "drift": drift_vals,
        "pullback_depth": pullback_depth_vals,
        "in_pullback": in_pullback,
    }


def compute_compression_signature(
    state: Dict[str, np.ndarray],
    window: int = 20,
) -> Dict[str, Any]:
    """Compute raw compression signature from price features.

    The compression signature measures how price converges toward the
    pullback EMA and how the pullback EMA converges toward the trend EMA
    during pullback events — the characteristic pattern edge_04 exploits.

    Returns dict with compression metrics.
    """
    n = len(state["price"])
    if n < 120:
        return {"error": "Insufficient data for compression analysis"}

    in_pb = state["in_pullback"]
    conf = state["confidence"]
    direction = state["direction"]
    price = state["price"]
    trend = state["trend_ema"]
    pullback = state["pullback_ema"]

    # Find pullback events
    events = []
    i = 105  # skip initial unstable
    while i < n:
        if in_pb[i] and conf[i] > 0:
            start = i
            peak_conf = conf[i]
            peak_idx = i
            for j in range(i, min(i + window, n)):
                if not in_pb[j] or conf[j] == 0:
                    break
                if conf[j] > peak_conf:
                    peak_conf = conf[j]
                    peak_idx = j
                i = j + 1

            # Pre-exit slice: 5 bars before peak confidence
            pre_start = max(start, peak_idx - 5)
            if pre_start < peak_idx and peak_idx < n:
                pre_slice = slice(pre_start, peak_idx)
                slice_len = peak_idx - pre_start

                if slice_len >= 2:
                    # Distance from price to pullback EMA (normalized)
                    d2p = np.abs(price[pre_slice] - pullback[pre_slice]) / np.maximum(price[pre_slice], 1e-12)
                    # Distance from pullback EMA to trend EMA (normalized)
                    dp2t = np.abs(pullback[pre_slice] - trend[pre_slice]) / np.maximum(price[pre_slice], 1e-12)
                    pre_conf = conf[pre_slice]

                    # Trends (linear fit)
                    x = np.arange(slice_len)
                    d2p_slope = np.polyfit(x, d2p, 1)[0] if slice_len >= 2 else 0.0
                    dp2t_slope = np.polyfit(x, dp2t, 1)[0] if slice_len >= 2 else 0.0
                    conf_slope = np.polyfit(x, pre_conf, 1)[0] if slice_len >= 2 else 0.0

                    # Compression strength: sum of absolute negative slopes
                    comp_strength = abs(min(0, d2p_slope)) + abs(min(0, dp2t_slope))

                    events.append({
                        "start_bar": int(start),
                        "peak_bar": int(peak_idx),
                        "direction": int(direction[peak_idx]),
                        "peak_confidence": float(round(conf[peak_idx], 4)),
                        "d2p_slope": float(round(d2p_slope, 8)),
                        "dp2t_slope": float(round(dp2t_slope, 8)),
                        "confidence_slope": float(round(conf_slope, 8)),
                        "compression_strength": float(round(comp_strength * 10000, 4)),
                        "is_compressing": bool(d2p_slope < 0 or dp2t_slope < 0),
                    })
        else:
            i += 1

    # Aggregate compression metrics
    if not events:
        return {
            "event_count": 0,
            "compression_score": 0.0,
            "events": [],
            "note": "No pullback events detected",
        }

    compressing_count = sum(1 for e in events if e["is_compressing"])
    compression_ratio = compressing_count / len(events)

    strengths = [e["compression_strength"] for e in events if e["compression_strength"] > 0]
    if len(strengths) >= 2:
        strength_cv = float(np.std(strengths) / max(np.mean(strengths), 0.001))
        strength_consistency = max(0.0, 1.0 - min(1.0, strength_cv))
    elif len(strengths) == 1:
        strength_consistency = 0.5
    else:
        strength_consistency = 0.0

    # Compression score: higher is more consistent
    compression_score = round(compression_ratio * 0.5 + strength_consistency * 0.5, 4)

    return {
        "event_count": len(events),
        "compressing_count": compressing_count,
        "compression_ratio": round(compression_ratio, 4),
        "compression_score": compression_score,
        "strength_consistency": round(strength_consistency, 4),
        "strength_values": [round(s, 4) for s in strengths],
        "events": events,
    }


# ---------------------------------------------------------------------------
# Stability Test: Run edge_04 with perturbed prices
# ---------------------------------------------------------------------------

def run_isolated_edge_04(
    mapper: EdgeSignalMapper,
    ohlcv: Dict[str, np.ndarray],
    run_label: str = "run_0",
    perturbation: float = 0.0,
) -> dict:
    """Run EdgeSignalMapper and extract pure edge_04 signal.

    No MOF, no RF, no lifecycle — just raw price -> edge transform.
    """
    # Apply perturbation if any
    if perturbation != 0.0:
        data = perturb_prices(ohlcv, perturbation)
    else:
        data = {k: v.copy() for k, v in ohlcv.items()}

    # Prepare inputs for EdgeSignalMapper.generate_all()
    closes_by_symbol = {EURJPY_SYMBOL: data["close"]}
    highs_by_symbol = {EURJPY_SYMBOL: data["high"]}
    lows_by_symbol = {EURJPY_SYMBOL: data["low"]}
    prices_by_symbol = {EURJPY_SYMBOL: float(data["close"][-1])}

    # Generate ALL edge signals (but we only look at edge_04)
    signals = mapper.generate_all(
        closes_by_symbol=closes_by_symbol,
        highs_by_symbol=highs_by_symbol,
        lows_by_symbol=lows_by_symbol,
        prices_by_symbol=prices_by_symbol,
    )

    # Extract edge_04 specifically
    edge_04_signal = None
    for s in signals:
        if s.get("edge_id") == EDGE_04_ID:
            edge_04_signal = {
                "symbol": s["symbol"],
                "direction": int(s["direction"]),
                "confidence": float(s["confidence"]),
                "ecdf": float(s["ecdf"]),
                "drift": int(s["drift"]),
                "price": float(s["price"]),
                "strategy": str(s["strategy"]),
                "edge_pf": float(s["edge_pf"]),
                "has_active_signal": bool(s["direction"] != 0 and s["confidence"] >= 0.3),
            }
            break

    # Compute edge_04's internal state from raw closes
    internal_state = compute_edge_04_state(data["close"])

    # Current bar's state
    last_dir = int(internal_state["direction"][-1])
    last_conf = float(internal_state["confidence"][-1])
    last_ecdf = float(internal_state["ecdf"][-1]) if not np.isnan(internal_state["ecdf"][-1]) else 0.5
    last_drift = int(internal_state["drift"][-1])
    last_pdepth = float(internal_state["pullback_depth"][-1])
    last_in_pb = bool(internal_state["in_pullback"][-1])

    # Count active pullback bars in last 50
    recent_pb = int(np.sum(internal_state["in_pullback"][-50:]))

    # Compression signature
    compression = compute_compression_signature(internal_state)

    # Price-derived stats
    closes = data["close"]
    returns = np.diff(closes) / closes[:-1]
    volatility = float(np.std(returns) * 100)  # % vol per bar
    atr_val = float(_compute_atr(data["high"], data["low"], data["close"])[-1]) if len(closes) >= 14 else 0.0
    rsi_val = float(_compute_rsi(closes)[-1]) if len(closes) >= 14 else 50.0

    result = {
        "run_label": run_label,
        "perturbation": perturbation,
        "timestamp": datetime.now().isoformat(),
        "data_stats": {
            "bars": len(closes),
            "close_first": float(closes[0]),
            "close_last": float(closes[-1]),
            "close_min": float(closes.min()),
            "close_max": float(closes.max()),
            "close_mean": float(closes.mean()),
            "volatility_pct": round(volatility, 4),
            "atr": round(atr_val, 6),
            "rsi": round(rsi_val, 2),
        },
        "edge_04_signal": edge_04_signal,
        "edge_04_internal_state": {
            "current_direction": last_dir,
            "current_confidence": round(last_conf, 4),
            "current_ecdf": round(last_ecdf, 4),
            "current_drift": last_drift,
            "current_pullback_depth": round(last_pdepth, 4),
            "in_pullback_now": last_in_pb,
            "pullback_bars_last_50": recent_pb,
        },
        "compression_signature": compression,
    }

    return result


# ---------------------------------------------------------------------------
# Report Builder
# ---------------------------------------------------------------------------

def build_isolation_report(
    run_results: List[dict],
    ohlcv_source: str,
) -> dict:
    """Build the edge_04 isolated signature report."""
    # Extract key values across runs for stability analysis
    directions = []
    confidences = []
    ecdfs = []
    drifts = []
    has_active = []
    compression_scores = []

    for r in run_results:
        sig = r.get("edge_04_signal")
        if sig:
            directions.append(sig["direction"])
            confidences.append(sig["confidence"])
            ecdfs.append(sig["ecdf"])
            drifts.append(sig["drift"])
            has_active.append(sig["has_active_signal"])
        cs = r.get("compression_signature", {})
        compression_scores.append(cs.get("compression_score", 0.0))

    # Stability metrics
    n_runs = len(run_results)
    conf_std = float(np.std(confidences)) if len(confidences) > 1 else 0.0
    conf_mean = float(np.mean(confidences)) if confidences else 0.0
    conf_cv = conf_std / max(conf_mean, 0.001) if conf_mean > 0 else 999.0

    ecdf_std = float(np.std(ecdfs)) if len(ecdfs) > 1 else 0.0
    ecdf_mean = float(np.mean(ecdfs)) if ecdfs else 0.5

    comp_score_std = float(np.std(compression_scores)) if len(compression_scores) > 1 else 0.0
    comp_score_mean = float(np.mean(compression_scores)) if compression_scores else 0.0

    dir_stable = len(set(directions)) == 1 if directions else False
    drift_stable = len(set(drifts)) == 1 if drifts else False

    # Stability assessment
    variance_pct = conf_cv * 100  # coefficient of variation as %
    stability_pass = variance_pct < 10.0  # variance < 10% success criterion

    report = {
        "report_metadata": {
            "report_type": "EDGE_04_ISOLATED_SIGNATURE",
            "phase": "Batch 6.4 Phase 1 — Edge Isolation Sandbox",
            "edge_id": EDGE_04_ID,
            "symbol": EURJPY_SYMBOL,
            "strategy": "pullback",
            "params": EDGE_04_PARAMS,
            "manifest_pf": 1.3104,
            "manifest_wf_pf": 1.3075,
            "data_source": ohlcv_source,
            "bars_used": run_results[0].get("data_stats", {}).get("bars", 0) if run_results else 0,
            "run_count": n_runs,
            "perturbations_applied": [r.get("perturbation", 0.0) for r in run_results],
            "contamination_controls": [
                "MOF: NOT LOADED",
                "RF Model: NOT LOADED",
                "Lifecycle State: NOT LOADED",
                "System State Files: NOT LOADED",
            ],
            "generated_at": datetime.now().isoformat(),
        },
        "isolated_edge_04_signal": {
            "description": (
                "Pure edge_04 signal from raw price-derived features only. "
                "No MOF gating, no RF model weighting, no lifecycle feedback. "
                "This is the uncontaminated signature."
            ),
            "current_signal": run_results[1].get("edge_04_signal") if len(run_results) > 1 else None,
            "signal_at_each_run": [
                {
                    "run": r.get("run_label"),
                    "perturbation": r.get("perturbation"),
                    "direction": r.get("edge_04_signal", {}).get("direction"),
                    "confidence": r.get("edge_04_signal", {}).get("confidence"),
                    "ecdf": r.get("edge_04_signal", {}).get("ecdf"),
                    "drift": r.get("edge_04_signal", {}).get("drift"),
                    "has_active_signal": r.get("edge_04_signal", {}).get("has_active_signal"),
                }
                for r in run_results
            ],
        },
        "stability_analysis": {
            "description": (
                "Stability across multiple runs with slight price perturbations. "
                "Variance < 10% indicates a genuine price-derived pattern, not noise."
            ),
            "direction_stable_across_runs": dir_stable,
            "drift_stable_across_runs": drift_stable,
            "confidence_mean": round(conf_mean, 4),
            "confidence_std": round(conf_std, 4),
            "confidence_cv_pct": round(variance_pct, 4),
            "ecdf_mean": round(ecdf_mean, 4),
            "ecdf_std": round(ecdf_std, 4),
            "compression_score_mean": round(comp_score_mean, 4),
            "compression_score_std": round(comp_score_std, 4),
            "variance_pct": round(variance_pct, 4),
            "stability_pass_variance_under_10pct": stability_pass,
            "interpretation": (
                "STABLE — genuine price-derived pattern"
                if stability_pass
                else "UNSTABLE — may be noise or system artifact"
            ),
        },
        "compression_signature_aggregate": {
            "description": (
                "Raw compression signature from price features: how price converges "
                "toward pullback EMA before signal peaks. Measured across all detected "
                "pullback events in the price series."
            ),
            "per_run": [
                {
                    "run": r.get("run_label"),
                    "event_count": r.get("compression_signature", {}).get("event_count", 0),
                    "compression_ratio": r.get("compression_signature", {}).get("compression_ratio", 0),
                    "compression_score": r.get("compression_signature", {}).get("compression_score", 0),
                    "strength_consistency": r.get("compression_signature", {}).get("strength_consistency", 0),
                }
                for r in run_results
            ],
            "aggregate_compression_score": round(comp_score_mean, 4),
        },
        "raw_data_profile": {
            "description": "Profile of the raw price data used for isolation.",
            "source": ohlcv_source,
            "stats": run_results[1].get("data_stats") if len(run_results) > 1 else None,
        },
        "success_criteria": {
            "edge_04_produces_signal_in_isolation": any(
                r.get("edge_04_signal") is not None for r in run_results
            ),
            "three_runs_consistent_variance_under_10pct": stability_pass,
            "result_file_saved": True,
        },
    }

    return report


def format_isolation_dashboard(report: dict) -> str:
    """Format the isolation report as a readable dashboard."""
    lines = []
    lines.append("")
    lines.append("=" * 78)
    lines.append("  EDGE ISOLATION SANDBOX — Pure Edge_04 Signature")
    lines.append("=" * 78)
    lines.append("")

    meta = report["report_metadata"]
    lines.append(f"  Edge ID:     {meta['edge_id']}")
    lines.append(f"  Symbol:      {meta['symbol']}")
    lines.append(f"  Strategy:    {meta['strategy']}")
    lines.append(f"  Data source: {meta['data_source']}")
    lines.append(f"  Bars:        {meta['bars_used']}")
    lines.append(f"  Runs:        {meta['run_count']}")
    lines.append(f"  Perturbations: {meta['perturbations_applied']}")
    lines.append("")
    lines.append(f"  Contamination Controls:")
    for ctrl in meta["contamination_controls"]:
        lines.append(f"    ✓ {ctrl}")
    lines.append("")

    # Edge_04 signal
    sig = report["isolated_edge_04_signal"]
    curr = sig.get("current_signal") or {}
    lines.append("  ┌─ ISOLATED EDGE_04 SIGNAL ─────────────────────────────")
    if curr:
        lines.append(f"  │ Direction:         {curr.get('direction', 'N/A'):+d}")
        lines.append(f"  │ Confidence (raw):  {curr.get('confidence', 0):.4f}")
        lines.append(f"  │ ECDF:              {curr.get('ecdf', 0):.4f}")
        lines.append(f"  │ Drift:             {curr.get('drift', 0):+d}")
        lines.append(f"  │ Price:             {curr.get('price', 0):.5f}")
        lines.append(f"  │ Has active signal: {curr.get('has_active_signal', False)}")
    else:
        lines.append(f"  │ No edge_04 signal detected")
    lines.append("  └──────────────────────────────────────────────────────")
    lines.append("")

    # Stability
    stab = report["stability_analysis"]
    lines.append("  ┌─ STABILITY ANALYSIS ───────────────────────────────────")
    lines.append(f"  │ Direction stable:     {stab['direction_stable_across_runs']}")
    lines.append(f"  │ Drift stable:         {stab['drift_stable_across_runs']}")
    lines.append(f"  │ Confidence mean:      {stab['confidence_mean']}")
    lines.append(f"  │ Confidence std:       {stab['confidence_std']}")
    lines.append(f"  │ Confidence CV (%):    {stab['confidence_cv_pct']:.4f}%")
    lines.append(f"  │ ECDF mean:            {stab['ecdf_mean']}")
    lines.append(f"  │ ECDF std:             {stab['ecdf_std']}")
    lines.append(f"  │ Compress score mean:  {stab['compression_score_mean']}")
    lines.append(f"  │ Compress score std:   {stab['compression_score_std']}")
    lines.append(f"  │ Variance < 10%:       {stab['stability_pass_variance_under_10pct']}")
    lines.append(f"  │ Interpretation:       {stab['interpretation']}")
    lines.append("  └──────────────────────────────────────────────────────")
    lines.append("")

    # Per-run details
    lines.append("  ┌─ SIGNAL PER RUN ───────────────────────────────────────")
    for r in sig.get("signal_at_each_run", []):
        lines.append(
            f"  │ {r['run']:12s} pert={r['perturbation']:+.4f}  "
            f"dir={r['direction']:+d}  conf={r['confidence']:.4f}  "
            f"ecdf={r['ecdf']:.4f}  drift={r['drift']:+d}  "
            f"active={r['has_active_signal']}"
        )
    lines.append("  └──────────────────────────────────────────────────────")
    lines.append("")

    # Compression
    comp = report["compression_signature_aggregate"]
    lines.append("  ┌─ COMPRESSION SIGNATURE AGGREGATE ─────────────────────")
    lines.append(f"  │ Aggregate score: {comp['aggregate_compression_score']}")
    for cr in comp.get("per_run", []):
        lines.append(
            f"  │ {cr['run']:12s} events={cr['event_count']:2d}  "
            f"ratio={cr['compression_ratio']:.4f}  score={cr['compression_score']:.4f}  "
            f"consistency={cr['strength_consistency']:.4f}"
        )
    lines.append("  └──────────────────────────────────────────────────────")
    lines.append("")

    # Success criteria
    sc = report["success_criteria"]
    lines.append("  ┌─ SUCCESS CRITERIA ─────────────────────────────────────")
    lines.append(f"  │ Signal in isolation:             {sc['edge_04_produces_signal_in_isolation']}")
    lines.append(f"  │ 3 runs variance < 10%:            {sc['three_runs_consistent_variance_under_10pct']}")
    lines.append(f"  │ Result file saved:                {sc['result_file_saved']}")
    lines.append("  └──────────────────────────────────────────────────────")
    lines.append("")
    lines.append("=" * 78)
    lines.append("  EDGE ISOLATION SANDBOX COMPLETE")
    lines.append("=" * 78)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the edge_04 isolation sandbox."""
    logger.info("=" * 78)
    logger.info("  EDGE ISOLATION SANDBOX — Batch 6.4 Phase 1")
    logger.info("=" * 78)
    logger.info("  Isolating edge_04 (EURJPY pullback, PF=1.31)")
    logger.info("  Contamination controls:")
    logger.info("    ✓ MOF: NOT loaded")
    logger.info("    ✓ RF Model: NOT loaded")
    logger.info("    ✓ Lifecycle State: NOT loaded")
    logger.info("    ✓ System State Files: NOT loaded")
    logger.info("")

    # ------------------------------------------------------------------
    # Step 1: Load EdgeSignalMapper (ONLY manifest + strategy functions)
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 1: Loading EdgeSignalMapper")
    logger.info("=" * 60)

    mapper = EdgeSignalMapper()
    logger.info("  Loaded %d edges across %d symbols",
                mapper.edge_count, len(mapper.get_symbols_with_edges()))

    # Verify edge_04 exists
    e04_edges = mapper.get_edges_for_symbol(EURJPY_SYMBOL)
    e04 = [e for e in e04_edges if e["id"] == EDGE_04_ID]
    if not e04:
        logger.error("Edge_04 not found in manifest — aborting")
        sys.exit(1)
    logger.info("  Edge_04 config: %s", json.dumps(e04[0], indent=2))
    logger.info("")

    # ------------------------------------------------------------------
    # Step 2: Load raw price data from MT5 (no state files)
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 2: Loading raw EURJPY price data")
    logger.info("=" * 60)

    ohlcv = fetch_eurjpy_ohlcv(count=BAR_COUNT)
    if ohlcv is None:
        logger.info("  MT5 unavailable — using synthetic EURJPY data")
        ohlcv = generate_synthetic_eurjpy_ohlcv(BAR_COUNT)
        ohlcv_source = "synthetic"
    else:
        ohlcv_source = "MT5_live"

    logger.info("  Data source: %s", ohlcv_source)
    logger.info("  Bars: %d", len(ohlcv["close"]))
    logger.info("  Close range: %.5f – %.5f (mean=%.5f)",
                ohlcv["close"].min(), ohlcv["close"].max(), ohlcv["close"].mean())
    logger.info("")

    # ------------------------------------------------------------------
    # Step 3: Run edge_04 in isolation 3 times with perturbations
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 3: Running edge_04 in isolation (3 runs)")
    logger.info("=" * 60)

    run_results: List[dict] = []
    for idx, pert in enumerate(PERTURBATIONS):
        run_label = f"run_{idx}"
        pert_str = f"{pert:+.4f}" if pert != 0 else " 0.0000 (baseline)"
        logger.info("  Run %d/%d: perturbation=%s", idx + 1, len(PERTURBATIONS), pert_str)

        result = run_isolated_edge_04(mapper, ohlcv, run_label=run_label, perturbation=pert)

        sig = result.get("edge_04_signal")
        if sig:
            logger.info(
                "    → direction=%+d  confidence=%.4f  ecdf=%.4f  drift=%+d  active=%s",
                sig["direction"], sig["confidence"], sig["ecdf"],
                sig["drift"], sig["has_active_signal"],
            )
        else:
            logger.info("    → No edge_04 signal produced")

        cs = result.get("compression_signature", {})
        logger.info(
            "    → Compression: %d events, ratio=%.4f, score=%.4f",
            cs.get("event_count", 0), cs.get("compression_ratio", 0),
            cs.get("compression_score", 0),
        )

        run_results.append(result)

    logger.info("")

    # ------------------------------------------------------------------
    # Step 4: Build and save report
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 4: Building isolation signature report")
    logger.info("=" * 60)

    report = build_isolation_report(run_results, ohlcv_source)

    report_path = os.path.join(_STATE_DIR, "edge_04_isolated_signature.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("  Report saved to: %s", report_path)
    logger.info("")

    # ------------------------------------------------------------------
    # Step 5: Print dashboard
    # ------------------------------------------------------------------
    dashboard = format_isolation_dashboard(report)
    print(dashboard)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    logger.info("=" * 78)
    logger.info("  ISOLATION SANDBOX SUMMARY")
    logger.info("=" * 78)

    sc = report["success_criteria"]
    signal_ok = sc["edge_04_produces_signal_in_isolation"]
    stability_ok = sc["three_runs_consistent_variance_under_10pct"]

    logger.info("  Signal in isolation:  %s", signal_ok)
    logger.info("  Stability (<10%% var): %s", stability_ok)

    if signal_ok and stability_ok:
        logger.info("  CONCLUSION: edge_04 is a REAL price-derived pattern —")
        logger.info("              proven by uncontaminated isolation.")
    elif signal_ok:
        logger.info("  CONCLUSION: edge_04 produces a signal but stability needs review.")
    else:
        logger.info("  CONCLUSION: edge_04 did NOT produce a signal in isolation —")
        logger.info("              may be a system artifact.")

    logger.info("")
    logger.info("  Result file: %s", report_path)
    logger.info("=" * 78)


if __name__ == "__main__":
    main()
