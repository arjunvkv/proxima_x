#!/usr/bin/env python3
"""
Batch 6.3 Phase 1 — Tasks 2 & 3: RF Variance Stability + MOF Shock Simulation

Task 2 — RF Variance Stability Analysis:
  1. Load RF model from research/models/edge_state_rf.joblib
  2. For each of 28 symbols, run RF inference 10 times with feature perturbations
  3. Measure mean probability, variance, collapse to 0.0
  4. Apply synthetic volatility shock (+2%, -2%, +5%) and re-run
  5. Check bounded range (0.3-0.9)

Task 3 — MOF Synthetic Volatility Shock:
  1. Create MOF with bootstrap_mode=False
  2. Run MOF evaluate with current signals
  3. Apply shock: signal confidences * 0.5, 2.0, 5.0
  4. Check if MOF stays in STRUCTURE_LIMITED or degrades
  5. Report shock boundaries

Usage:
    python -m proxima_x.bootstrap.rf_stress_test

Output:
    - Full console report with tables and pass/fail verdicts
"""

from __future__ import annotations

import copy
import json
import logging
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rf_stress_test")

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_STATE_DIR = os.path.join(_PROJECT_ROOT, "state")
_MODEL_DIR = os.path.join(_PROJECT_ROOT, "research", "models")
_LIFECYCLE_PATH = os.path.join(_STATE_DIR, "lifecycle_state.json")

ALL_SYMBOLS_28: List[str] = [
    "EURJPY", "USDJPY", "EURUSD", "GBPUSD", "AUDUSD", "USDCAD", "USDCHF",
    "NZDUSD", "GBPJPY", "AUDJPY", "CADJPY", "CHFJPY", "NZDJPY", "EURGBP",
    "EURCHF", "EURAUD", "EURCAD", "EURNZD", "GBPCHF", "GBPAUD", "GBPCAD",
    "GBPNZD", "AUDNZD", "AUDCAD", "AUDCHF", "NZDCAD", "NZDCHF", "CADCHF",
]

FEAT_ORDER: List[str] = [
    "fft_low", "fft_mid", "fft_high", "spectral_entropy", "corr",
    "lag_corr", "max_corr", "zero_sync", "tpi_flip", "bfd_burst",
    "bfd_density", "tpi_mean", "tpi_std", "tpi_skew",
]

WINDOW = 2000
READINESS_THRESHOLD = 0.50

SYMBOL_PROFILES_V2: Dict[str, Dict[str, float]] = {
    "EURUSD": {"drift": 0.0035, "vol": 1.15, "bfd_scale": 1.0},
    "GBPUSD": {"drift": 0.0032, "vol": 1.20, "bfd_scale": 1.1},
    "USDJPY": {"drift": 0.0030, "vol": 1.10, "bfd_scale": 1.0},
    "EURJPY": {"drift": 0.0038, "vol": 1.10, "bfd_scale": 1.0},
    "AUDUSD": {"drift": 0.0025, "vol": 0.95, "bfd_scale": 0.9},
    "USDCAD": {"drift": 0.0028, "vol": 1.00, "bfd_scale": 0.9},
    "USDCHF": {"drift": 0.0025, "vol": 0.95, "bfd_scale": 0.9},
    "NZDUSD": {"drift": 0.0020, "vol": 0.90, "bfd_scale": 0.8},
    "GBPJPY": {"drift": 0.0040, "vol": 1.30, "bfd_scale": 1.2},
    "AUDJPY": {"drift": 0.0030, "vol": 1.00, "bfd_scale": 0.9},
    "CADJPY": {"drift": 0.0028, "vol": 1.00, "bfd_scale": 0.9},
    "CHFJPY": {"drift": 0.0025, "vol": 0.90, "bfd_scale": 0.9},
    "NZDJPY": {"drift": 0.0022, "vol": 0.85, "bfd_scale": 0.8},
    "EURGBP": {"drift": 0.0025, "vol": 0.90, "bfd_scale": 0.9},
    "EURCHF": {"drift": 0.0020, "vol": 0.85, "bfd_scale": 0.8},
    "EURAUD": {"drift": 0.0028, "vol": 0.95, "bfd_scale": 0.9},
    "EURCAD": {"drift": 0.0030, "vol": 1.00, "bfd_scale": 0.9},
    "EURNZD": {"drift": 0.0025, "vol": 0.90, "bfd_scale": 0.8},
    "GBPCHF": {"drift": 0.0032, "vol": 1.10, "bfd_scale": 1.0},
    "GBPAUD": {"drift": 0.0028, "vol": 1.00, "bfd_scale": 0.9},
    "GBPCAD": {"drift": 0.0030, "vol": 1.05, "bfd_scale": 1.0},
    "GBPNZD": {"drift": 0.0028, "vol": 1.00, "bfd_scale": 0.9},
    "AUDNZD": {"drift": 0.0018, "vol": 0.80, "bfd_scale": 0.8},
    "AUDCAD": {"drift": 0.0022, "vol": 0.90, "bfd_scale": 0.8},
    "AUDCHF": {"drift": 0.0020, "vol": 0.85, "bfd_scale": 0.8},
    "NZDCAD": {"drift": 0.0018, "vol": 0.80, "bfd_scale": 0.8},
    "NZDCHF": {"drift": 0.0015, "vol": 0.80, "bfd_scale": 0.8},
    "CADCHF": {"drift": 0.0020, "vol": 0.85, "bfd_scale": 0.8},
}

# ═══════════════════════════════════════════════════════════════════════════════
# Part 1: RF Feature Extraction (from rf_rehydration.py)
# ═══════════════════════════════════════════════════════════════════════════════


def extract_features(tpi: np.ndarray, bfd: np.ndarray) -> Dict[str, float]:
    """Extract the 14 RF features from TPI and BFD arrays."""
    from scipy import signal as scipy_signal

    Nl = len(tpi)
    f: Dict[str, float] = {}

    # FFT spectral features
    fft = np.fft.rfft(tpi - np.mean(tpi))
    fp = np.abs(fft) ** 2
    n = len(fp)
    l1, l2 = n // 3, 2 * n // 3

    f["fft_low"] = float(np.sum(fp[1:l1]) / max(np.sum(fp[1:]), 1))
    f["fft_mid"] = float(np.sum(fp[l1:l2]) / max(np.sum(fp[1:]), 1))
    f["fft_high"] = float(np.sum(fp[l2:]) / max(np.sum(fp[1:]), 1))

    # Spectral entropy
    p = fp[1:] / max(np.sum(fp[1:]), 1e-12)
    p = p[p > 0]
    f["spectral_entropy"] = float(-np.sum(p * np.log2(p))) if len(p) > 0 else 0.0

    # Cross-correlation TPI ↔ BFD
    if np.std(tpi) > 1e-12 and np.std(bfd) > 1e-12:
        f["corr"] = float(np.corrcoef(tpi, bfd)[0, 1])
        cc = scipy_signal.correlate(
            tpi - np.mean(tpi), bfd - np.mean(bfd), mode="same"
        )
        cc = cc / (Nl * np.std(tpi) * np.std(bfd) + 1e-12)
        mi = int(np.argmax(np.abs(cc)))
        f["lag_corr"] = float(mi - Nl // 2)
        f["max_corr"] = float(cc[mi])
    else:
        f["corr"] = f["lag_corr"] = f["max_corr"] = 0.0

    # Zero-crossing sync
    ts = np.sign(tpi)
    bs = np.sign(bfd)
    f["zero_sync"] = float(np.mean((ts != 0) & (bs != 0) & (ts == bs)))

    # TPI flip rate
    tn = tpi[np.abs(tpi) > 1e-12]
    if len(tn) > 1:
        flips = np.sum(np.abs(np.diff(np.sign(tn)))) / 2
        f["tpi_flip"] = float(flips / max(len(tn), 1))
    else:
        f["tpi_flip"] = 0.0

    # BFD burst / density
    burst_indices = np.where(bfd > 0.05)[0]
    if len(burst_indices) > 1:
        inter = np.diff(burst_indices)
        f["bfd_burst"] = float(np.std(inter) / max(np.mean(inter), 1e-12))
    else:
        f["bfd_burst"] = 0.0
    f["bfd_density"] = float(np.mean(bfd > 0.05))

    # TPI distribution
    f["tpi_mean"] = float(np.mean(tpi))
    f["tpi_std"] = float(np.std(tpi))
    if np.std(tpi) > 1e-12:
        f["tpi_skew"] = float(np.mean((tpi - np.mean(tpi)) ** 3) / max(np.std(tpi) ** 3, 1e-12))
    else:
        f["tpi_skew"] = 0.0

    return f


def generate_synthetic_tpi_bfd(
    symbol: str,
    n: int = WINDOW,
    seed: Optional[int] = None,
    volatility_shock: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic TPI and BFD with optional volatility shock multiplier."""
    if seed is not None:
        np.random.seed(seed)

    profile = SYMBOL_PROFILES_V2.get(symbol, {"drift": 0.003, "vol": 1.0, "bfd_scale": 1.0})
    drift = profile["drift"]
    vol = profile["vol"] * volatility_shock  # Apply volatility shock
    bfd_scale = profile["bfd_scale"]

    # TPI generation via random walk with drift
    steps = np.random.randn(n) * 0.008 * vol + drift

    # Add occasional directional momentum bursts
    n_bursts = max(1, n // 250)
    for _ in range(n_bursts):
        pos = np.random.randint(50, n - 50)
        burst_len = np.random.randint(10, 40)
        burst_dir = 1.0 if np.random.rand() > 0.3 else -1.0
        burst_mag = np.random.uniform(0.015, 0.035)
        steps[pos:pos + burst_len] += burst_dir * burst_mag

    tpi = np.cumsum(steps)
    tpi = tpi - np.mean(tpi[:200])

    # Add slow oscillatory component
    n_cycles = np.random.uniform(1.0, 4.0)
    slow_osc = np.sin(np.linspace(0, np.pi * n_cycles, n)) * 0.01
    tpi = tpi + slow_osc

    target_mean = 0.020
    target_std = 0.058
    current_std = max(np.std(tpi), 1e-10)
    tpi = (tpi - np.mean(tpi)) / current_std * target_std + target_mean
    tpi = np.clip(tpi, -0.12, 0.30)

    # BFD generation
    bfd_base = np.random.randn(n) * 0.012 * bfd_scale + 0.24
    n_dips = max(1, int(n * 0.03))
    dip_positions = np.random.randint(0, n - 1, size=n_dips)
    bfd_base[dip_positions] = np.random.rand(n_dips) * 0.03
    bfd = bfd_base + tpi * 0.02
    bfd = np.clip(bfd, 0.0, 0.40)

    return tpi, bfd


def perturb_features(features: Dict[str, float], noise_scale: float = 0.01) -> Dict[str, float]:
    """Add small Gaussian perturbations to features for variance testing."""
    perturbed = {}
    for k, v in features.items():
        if k in ("lag_corr",):
            # lag_corr is integer-like, perturb with discrete noise
            perturbed[k] = v + np.random.randint(-3, 4)
        else:
            noise = np.random.randn() * noise_scale * max(abs(v), 0.001)
            perturbed[k] = v + noise
    return perturbed


def infer_readiness(model, tpi: np.ndarray, bfd: np.ndarray) -> float:
    """Run RF inference and return probability of class 1."""
    fv = extract_features(tpi, bfd)
    df = pd.DataFrame([fv])[FEAT_ORDER]
    prob = float(model.predict_proba(df)[0, 1])
    return prob


def infer_readiness_from_features(model, features: Dict[str, float]) -> float:
    """Run RF inference from pre-computed features."""
    df = pd.DataFrame([features])[FEAT_ORDER]
    prob = float(model.predict_proba(df)[0, 1])
    return prob


def load_rf_model() -> Tuple[Any, float]:
    """Load the RF model."""
    model_path = os.path.join(_MODEL_DIR, "edge_state_rf.joblib")
    if not os.path.exists(model_path):
        logger.error(f"RF model not found at {model_path}")
        return None, 0.0

    try:
        import joblib
        mdl = joblib.load(model_path)
        model_obj = mdl["model"]
        logger.info(f"Loaded RF model from {model_path}")
        logger.info(f"  n_estimators={model_obj.n_estimators}, "
                     f"max_depth={model_obj.max_depth}, "
                     f"n_features={model_obj.n_features_in_}")
        return model_obj, 0.60
    except Exception as e:
        logger.error(f"Failed to load RF model: {e}")
        return None, 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Task 2: RF Variance Stability Analysis
# ═══════════════════════════════════════════════════════════════════════════════


def task2_rf_stability(model) -> Dict[str, Any]:
    """Run RF variance stability analysis across all 28 symbols.

    For each symbol:
    - Generate data, run inference (baseline)
    - Run 10 perturbed inferences
    - Measure mean, variance, min
    - Check for collapse to 0.0

    Then apply volatility shocks (+2%, -2%, +5%) to the price process
    and re-run to check bounded output.
    """
    print()
    print("=" * 72)
    print("  TASK 2: RF VARIANCE STABILITY ANALYSIS")
    print("=" * 72)
    print()

    results: Dict[str, Dict[str, Any]] = {}
    all_stable = True
    any_collapse = False

    for idx, symbol in enumerate(ALL_SYMBOLS_28):
        # --- Baseline inference ---
        base_seed = 42 + idx
        tpi, bfd = generate_synthetic_tpi_bfd(symbol, seed=base_seed)
        baseline_prob = infer_readiness(model, tpi, bfd)

        # --- 10 perturbed inferences ---
        perturbed_probs = []
        base_features = extract_features(tpi, bfd)

        for run_i in range(10):
            np.random.seed(base_seed * 100 + run_i * 7 + 13)
            perturbed_feats = perturb_features(base_features, noise_scale=0.02)
            p = infer_readiness_from_features(model, perturbed_feats)
            perturbed_probs.append(p)

        perturbed_arr = np.array(perturbed_probs)
        mean_prob = float(np.mean(perturbed_arr))
        variance = float(np.var(perturbed_arr))
        std_prob = float(np.std(perturbed_arr))
        min_prob = float(np.min(perturbed_arr))
        max_prob = float(np.max(perturbed_arr))

        collapsed = min_prob == 0.0
        if collapsed:
            any_collapse = True

        # Stable if variance is low (threshold: < 0.001) AND no collapse
        stable = variance < 0.001 and not collapsed
        if not stable:
            all_stable = False

        results[symbol] = {
            "baseline": round(baseline_prob, 6),
            "mean": round(mean_prob, 6),
            "variance": variance,
            "std": round(std_prob, 6),
            "min": round(min_prob, 6),
            "max": round(max_prob, 6),
            "collapsed": collapsed,
            "stable": stable,
        }

    # --- Print results table ---
    hdr = f"  {'Symbol':<8s} {'Baseline':>10s} {'Mean':>8s} {'Std':>8s} "
    hdr += f"{'Min':>8s} {'Max':>8s} {'Var':>10s} {'Collapsed':>10s} {'Stable':>8s}"
    print(hdr)
    sep = "  " + "-"*8 + " " + "-"*10 + " " + "-"*8 + " " + "-"*8 + " "
    sep += "-"*8 + " " + "-"*10 + " " + "-"*10 + " " + "-"*8
    print(sep)

    stable_count = 0
    collapse_count = 0
    variance_sum = 0.0
    for sym in ALL_SYMBOLS_28:
        r = results[sym]
        collapsed_str = "⚠️ YES" if r["collapsed"] else "NO"
        stable_str = "✓" if r["stable"] else "✗"
        print(f"  {sym:<8s} {r['baseline']:>10.4f} {r['mean']:>8.4f} {r['std']:>8.4f} "
              f"{r['min']:>8.4f} {r['max']:>8.4f} {r['variance']:>10.6f} "
              f"{collapsed_str:>10s} {stable_str:>8s}")
        if r["stable"]:
            stable_count += 1
        if r["collapsed"]:
            collapse_count += 1
        variance_sum += r["variance"]

    avg_variance = variance_sum / len(ALL_SYMBOLS_28)

    print()
    print(f"  Summary:")
    print(f"    Total symbols:      {len(ALL_SYMBOLS_28)}")
    print(f"    Variance < 0.001:   {stable_count}/{len(ALL_SYMBOLS_28)}")
    print(f"    Collapsed symbols:  {collapse_count}/{len(ALL_SYMBOLS_28)}")
    print(f"    Avg variance:       {avg_variance:.8f}")
    print(f"    Max variance:       {max(r['variance'] for r in results.values()):.8f}")
    print(f"    Any collapse:       {'YES' if any_collapse else 'NO'}")

    # Success criteria: NO collapse to 0.0 under perturbation
    no_collapse = not any_collapse
    print(f"\n  >>> No collapse to 0.0: {'✅ PASS' if no_collapse else '❌ FAIL'}")
    print(f"  >>> Variance bounded:      {'✅ LOW (' + str(round(avg_variance, 6)) + ')' if avg_variance < 0.005 else '⚠️ ELEVATED'}")

    # === Volatility Shock Tests ===
    print()
    print(f"  {'─' * 72}")
    print(f"  Volatility Shock Test:")
    print()

    shock_results: Dict[str, Any] = {}
    all_bounded = True

    for shock_label, shock_mult in [("+2%", 1.02), ("-2%", 0.98), ("+5%", 1.05)]:
        shock_probs = []
        for idx, symbol in enumerate(ALL_SYMBOLS_28):
            base_seed = 42 + idx
            tpi_shock, bfd_shock = generate_synthetic_tpi_bfd(
                symbol, seed=base_seed, volatility_shock=shock_mult
            )
            prob = infer_readiness(model, tpi_shock, bfd_shock)
            shock_probs.append(prob)

        shock_arr = np.array(shock_probs)
        min_p = float(np.min(shock_arr))
        max_p = float(np.max(shock_arr))
        mean_p = float(np.mean(shock_arr))

        # Check bounded range: 0.3-0.9
        out_of_bounds_low = int(np.sum(shock_arr < 0.3))
        out_of_bounds_high = int(np.sum(shock_arr > 0.9))
        out_of_bounds = out_of_bounds_low + out_of_bounds_high
        bounded = out_of_bounds == 0

        if not bounded:
            all_bounded = False

        shock_results[shock_label] = {
            "multiplier": shock_mult,
            "min": round(min_p, 4),
            "max": round(max_p, 4),
            "mean": round(mean_p, 4),
            "out_of_bounds_low": out_of_bounds_low,
            "out_of_bounds_high": out_of_bounds_high,
            "out_of_bounds_total": out_of_bounds,
            "bounded": bounded,
        }

        bounded_str = "✓ BOUNDED" if bounded else "✗ OUT OF BOUNDS"
        print(f"    Shock {shock_label:<5s} (x{shock_mult:.2f}):  "
              f"min={min_p:.4f}  max={max_p:.4f}  mean={mean_p:.4f}  "
              f"OOB={out_of_bounds:2d}/{len(ALL_SYMBOLS_28):2d}  {bounded_str}")

    print()
    print(f"  >>> All shocks bounded (0.3-0.9): {'✅ PASS' if all_bounded else '❌ FAIL'}")

    return {
        "per_symbol": results,
        "stable_count": stable_count,
        "collapse_count": collapse_count,
        "avg_variance": round(avg_variance, 8),
        "all_stable": all_stable,
        "any_collapse": any_collapse,
        "no_collapse": not any_collapse,
        "passed": not any_collapse,
        "shock_results": shock_results,
        "all_shocks_bounded": all_bounded,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Task 3: MOF Synthetic Volatility Shock
# ═══════════════════════════════════════════════════════════════════════════════

# Build minimal cluster_states for MOF evaluation
def build_cluster_states() -> Dict[str, Dict[str, Any]]:
    """Build minimal cluster states for MOF testing."""
    return {
        "cluster_1": {
            "coherence": 0.45,
            "active_symbols": 5,
            "net_direction": 1,
            "divergence": 0.1,
            "net_pressure": "BULLISH",
        },
        "cluster_2": {
            "coherence": 0.50,
            "active_symbols": 4,
            "net_direction": -1,
            "divergence": 0.2,
            "net_pressure": "BEARISH",
        },
        "cluster_3": {
            "coherence": 0.35,
            "active_symbols": 3,
            "net_direction": 0,
            "divergence": 0.05,
            "net_pressure": "NEUTRAL",
        },
        "cluster_4": {
            "coherence": 0.55,
            "active_symbols": 2,
            "net_direction": 1,
            "divergence": 0.08,
            "net_pressure": "BULLISH",
        },
    }


def build_base_signals() -> List[Dict[str, Any]]:
    """Build a representative set of OSS signals for MOF evaluation."""
    signals = []
    # Create signals for the 5 open positions
    position_symbols = [
        ("NZDCAD", 1, 0.45),
        ("EURUSD", 1, 0.50),
        ("EURJPY", -1, 0.40),
        ("GBPUSD", -1, 0.45),
        ("NZDCAD", -1, 0.35),
    ]
    for sym, direction, confidence in position_symbols:
        signals.append({
            "symbol": sym,
            "direction": direction,
            "confidence": confidence,
            "ecdf": 0.5,
            "drift": 0,
        })

    # Add diversified signals across more symbols
    extra_symbols = [
        ("USDJPY", 1, 0.50),
        ("AUDUSD", -1, 0.45),
        ("USDCAD", 1, 0.40),
        ("GBPJPY", -1, 0.55),
        ("EURGBP", 1, 0.50),
        ("NZDUSD", -1, 0.35),
        ("AUDJPY", 1, 0.45),
        ("CADJPY", -1, 0.40),
        ("EURAUD", 1, 0.50),
        ("GBPAUD", -1, 0.45),
        ("NZDCAD", 1, 0.30),
        ("NZDCHF", -1, 0.40),
        ("CADCHF", 1, 0.35),
    ]
    for sym, direction, confidence in extra_symbols:
        signals.append({
            "symbol": sym,
            "direction": direction,
            "confidence": confidence,
            "ecdf": 0.5,
            "drift": 0,
        })

    return signals


def task3_mof_shock_simulation() -> Dict[str, Any]:
    """Run MOF synthetic volatility shock simulation.

    1. Create MOF with bootstrap_mode=False
    2. Run MOF evaluate with current signals
    3. Apply shock: signal confidences * 0.5x, 2.0x, 5.0x
    4. Check if MOF stays in STRUCTURE_LIMITED or degrades
    """
    print()
    print("=" * 72)
    print("  TASK 3: MOF SYNTHETIC VOLATILITY SHOCK")
    print("=" * 72)
    print()

    # Import MOF
    sys.path.insert(0, _PROJECT_ROOT)
    try:
        from proxima_ops.risk.market_observability_filter import (
            MarketObservabilityFilter,
        )
    except ImportError as e:
        logger.error(f"Cannot import MOF: {e}")
        # Fall back to direct import
        import importlib.util
        try:
            mof_path = os.path.join(
                _PROJECT_ROOT, "proxima_x", "proxima_ops", "risk",
                "market_observability_filter.py",
            )
            spec = importlib.util.spec_from_file_location(
                "market_observability_filter", mof_path
            )
            mof_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mof_mod)
            MarketObservabilityFilter = mof_mod.MarketObservabilityFilter
        except Exception as e2:
            logger.error(f"Cannot load MOF via importlib: {e2}")
            return {"error": str(e2)}

    # 1. Create MOF with bootstrap_mode=False
    mof = MarketObservabilityFilter(bootstrap_mode=False)
    logger.info("MOF created with bootstrap_mode=False")

    # 2. Build cluster states and base signals
    cluster_states = build_cluster_states()
    base_signals = build_base_signals()

    print(f"  Cluster states: {len(cluster_states)} clusters")
    print(f"  Base signals:   {len(base_signals)} signals")
    print()

    # 3. Run baseline MOF evaluation
    baseline_result = mof.evaluate(cluster_states, base_signals)
    baseline_state = baseline_result["observability_state"]
    baseline_score = baseline_result["observability_score"]
    baseline_permission = baseline_result["action_permission"]
    baseline_components = baseline_result["components"]

    print(f"  ── Baseline MOF (no shock) ──")
    print(f"    State:      {baseline_state}")
    print(f"    Score:      {baseline_score:.4f}")
    print(f"    Permission: {baseline_permission}")
    print(f"    Components: coherence={baseline_components['coherence_quality']:.4f}, "
          f"confidence={baseline_components['oss_confidence_quality']:.4f}, "
          f"stability={baseline_components['stability_quality']:.4f}")
    print()

    shock_multipliers = [0.5, 2.0, 5.0]
    shock_results_list = []

    for mult in shock_multipliers:
        # Apply confidence multiplier shock
        shocked_signals = copy.deepcopy(base_signals)
        for sig in shocked_signals:
            sig["confidence"] = min(1.0, sig.get("confidence", 0.5) * mult)

        # Also apply a coherence shock for the 5x multiplier to simulate
        # extreme volatility
        shocked_clusters = copy.deepcopy(cluster_states)
        if mult >= 5.0:
            for cname in shocked_clusters:
                shocked_clusters[cname]["coherence"] *= 0.7  # Reduce coherence

        shocked_result = mof.evaluate(shocked_clusters, shocked_signals)
        shocked_state = shocked_result["observability_state"]
        shocked_score = shocked_result["observability_score"]
        shocked_components = shocked_result["components"]
        force_triggers = shocked_result.get("force_triggers", [])

        # Check if MOF stays in STRUCTURE_LIMITED (or degrades)
        # Currently the system is in STRUCTURE_LIMITED with score ~0.46
        if shocked_state == "INFORMATION_DEGRADED":
            degraded_by_shock = True
            remained_limited = False
            is_rich = False
        elif shocked_state == "STRUCTURE_LIMITED":
            degraded_by_shock = False
            remained_limited = True
            is_rich = False
        else:  # INFORMATION_RICH
            degraded_by_shock = False
            remained_limited = False
            is_rich = True

        shock_results_list.append({
            "multiplier": mult,
            "state": shocked_state,
            "score": round(shocked_score, 4),
            "remained_limited": remained_limited,
            "degraded": degraded_by_shock,
            "became_rich": is_rich,
            "components": shocked_components,
            "force_triggers": force_triggers,
        })

        print(f"  ── Shock x{mult:.1f} ──")
        print(f"    State:      {shocked_state}")
        print(f"    Score:      {shocked_score:.4f}")
        print(f"    Components: coherence={shocked_components['coherence_quality']:.4f}, "
              f"confidence={shocked_components['oss_confidence_quality']:.4f}, "
              f"stability={shocked_components['stability_quality']:.4f}")
        if force_triggers:
            print(f"    Force trigs: {force_triggers}")
        if degraded_by_shock:
            print(f"    ⚠️  DEGRADED by shock!")
        elif remained_limited:
            print(f"    ✓ Remained in STRUCTURE_LIMITED")
        elif is_rich:
            print(f"    ↑ Upgraded to INFORMATION_RICH")
        print()

    # Determine shock boundaries
    print(f"  ── Shock Boundary Analysis ──")

    # Find the multiplier range where MOF stays in STRUCTURE_LIMITED
    limited_multipliers = [r["multiplier"] for r in shock_results_list if r["remained_limited"]]
    degraded_multipliers = [r["multiplier"] for r in shock_results_list if r["degraded"]]

    print(f"    STRUCTURE_LIMITED at multipliers: {limited_multipliers}")
    print(f"    INFORMATION_DEGRADED at multipliers: {degraded_multipliers}")

    baseline_ok = baseline_state == "STRUCTURE_LIMITED" or baseline_state == "INFORMATION_RICH"
    print(f"    Baseline state OK (not DEGRADED): {'YES' if baseline_ok else 'NO'}")

    # Check moderate shock tolerance
    moderate_shock_ok = True
    for r in shock_results_list:
        if r["multiplier"] <= 2.0 and r["degraded"]:
            moderate_shock_ok = False
        if r["multiplier"] >= 5.0 and r["degraded"]:
            pass  # 5x may be expected to degrade

    # Determine the maximum safe multiplier
    safe_multiplier = max([r["multiplier"] for r in shock_results_list if not r["degraded"]] + [1.0])

    moderate_shock_ok = safe_multiplier >= 2.0
    print(f"    Max safe multiplier: x{safe_multiplier:.1f}")
    print(f"    Moderate shock tolerance (>= 2x): {'✅ PASS' if moderate_shock_ok else '❌ FAIL'}")
    print()

    return {
        "baseline": {
            "state": baseline_state,
            "score": round(baseline_score, 4),
            "permission": baseline_permission,
            "components": baseline_components,
        },
        "shocks": shock_results_list,
        "safe_multiplier": safe_multiplier,
        "moderate_shock_tolerance_ok": moderate_shock_ok,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> int:
    print()
    print("=" * 72)
    print("  BATCH 6.3 PHASE 1 — STRUCTURAL STRESS VALIDATION")
    print("  RF Variance Stability + MOF Shock Simulation")
    print("=" * 72)
    print()

    # Load RF model
    model, threshold = load_rf_model()
    if model is None:
        logger.error("Cannot proceed without RF model")
        return 1

    # ── Task 2: RF Variance Stability ──
    t2_results = task2_rf_stability(model)
    # Success criteria: no collapse to 0.0 AND all shocks bounded in [0.3, 0.9]
    t2_passed = (not t2_results.get("any_collapse", True)
                 and t2_results.get("all_shocks_bounded", False))

    # ── Task 3: MOF Shock Simulation ──
    t3_results = task3_mof_shock_simulation()
    t3_passed = t3_results.get("moderate_shock_tolerance_ok", False)

    # ── Final Verdict ──
    print()
    print("=" * 72)
    print("  FINAL VERDICT")
    print("=" * 72)
    print()
    print(f"  Task 2 — RF Variance Stability:")
    print(f"    Collapsed symbols:     {t2_results.get('collapse_count', 0)}/{len(ALL_SYMBOLS_28)}")
    print(f"    No collapse to 0.0:   {'YES' if not t2_results.get('any_collapse', True) else 'NO'}")
    print(f"    Avg variance:         {t2_results.get('avg_variance', 0):.8f}")
    print(f"    All shocks bounded:   {'YES' if t2_results.get('all_shocks_bounded', False) else 'NO'}")
    print(f"    Overall:              {'✅ PASS' if t2_passed else '❌ FAIL'}")
    print()
    print(f"  Task 3 — MOF Shock Simulation:")
    print(f"    Baseline state:       {t3_results.get('baseline', {}).get('state', 'N/A')}")
    print(f"    Baseline score:       {t3_results.get('baseline', {}).get('score', 0):.4f}")
    print(f"    Max safe multiplier:  x{t3_results.get('safe_multiplier', 0):.1f}")
    print(f"    Moderate shock safe:  {'YES' if t3_results.get('moderate_shock_tolerance_ok', False) else 'NO'}")
    print(f"    Overall:              {'✅ PASS' if t3_passed else '❌ FAIL'}")
    print()
    print(f"  Combined: {'✅ ALL PASS' if (t2_passed and t3_passed) else '❌ SOME FAILURES'}")
    print()
    print("=" * 72)
    print()

    return 0 if (t2_passed and t3_passed) else 1


if __name__ == "__main__":
    sys.exit(main())
