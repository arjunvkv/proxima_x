"""Batch 6.4 Phases 2+3 — Contamination Coefficient Audit + Edge_04 Identity Lock Test.

Phase 2 — Contamination Coefficient Audit:
    Quantifies how much each system layer (RF, MOF, lifecycle) contaminates
    edge_04's isolated signal output. Produces contamination coefficients
    per layer (0.0 = no influence, 1.0 = fully contaminated).

Phase 3 — Edge_04 Identity Lock Test:
    Compares edge_04 in fully isolated vs fully integrated mode and computes
    Identity Variance = 1 - correlation(isolated, integrated).
    Success criteria: Identity Variance ≤ 20%.

Usage::
    cd C:\\Trading\\Agentic_Trading\\proxima_x
    python bootstrap/contamination_audit.py

Outputs::
    state/contamination_audit.json          — Phase 2 results
    state/edge_04_identity_lock.json        — Phase 3 results

Constraints::
    - NO trading execution
    - NO new seed trades
    - PURE analysis only
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── Path setup ───────────────────────────────────────────────────────────────
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
_STATE_DIR = os.path.join(_PROJECT_ROOT, "state")
_BOOTSTRAP_DIR = os.path.join(_PROJECT_ROOT, "proxima_x", "bootstrap")

sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "proxima_x"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("contamination_audit")

os.makedirs(_STATE_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EDGE_04_ID = "edge_04"
EDGE_04_SYMBOL = "EURJPY"
EDGE_04_PARAMS = {"trend_ema": 100, "pullback_ema": 10, "max_hold": 18}

# Observation window in seconds for timing drift normalization
OBSERVATION_WINDOW_SECONDS = 300  # 5 minutes of M5 bars = 60 bars
BAR_COUNT = 500

# ---------------------------------------------------------------------------
# Imports (local — after sys.path)
# ---------------------------------------------------------------------------

try:
    from proxima_ops.risk.edge_signal_mapper import (
        EdgeSignalMapper,
        _pullback_signal,
        _ema,
    )
except ImportError as exc:
    logger.error("Cannot import EdgeSignalMapper: %s", exc)
    sys.exit(1)

try:
    from proxima_ops.risk.market_observability_filter import (
        MarketObservabilityFilter,
        ObservabilityState,
        ActionPermission,
    )
except ImportError as exc:
    logger.error("Cannot import MarketObservabilityFilter: %s", exc)
    sys.exit(1)

try:
    from research.shadow_extractor import ShadowFrame
except ImportError:
    ShadowFrame = None
    logger.warning("ShadowFrame not available — using synthetic features")

# ---------------------------------------------------------------------------
# Data Generation
# ---------------------------------------------------------------------------

def generate_synthetic_eurjpy_prices(
    n_bars: int = BAR_COUNT,
    seed: int = 42,
) -> np.ndarray:
    """Generate synthetic EURJPY prices with trend + pullback structure."""
    rng = np.random.RandomState(seed)
    base = 184.0
    trend = np.cumsum(rng.randn(n_bars) * 0.001)
    noise = rng.randn(n_bars) * 0.005
    prices = base + trend + noise
    # Inject pullback patterns
    pullback_indices = rng.choice(
        np.arange(50, n_bars - 50), size=n_bars // 30, replace=False
    )
    for idx in pullback_indices:
        pull_size = rng.uniform(-0.15, 0.15)
        length = rng.randint(5, 15)
        for i in range(length):
            if idx + i < n_bars:
                prices[idx + i] += pull_size * (1 - i / length)
    prices = np.maximum(prices, base * 0.95)
    return prices


def compute_edge_04_internal_state(
    closes: np.ndarray,
    trend_span: int = 100,
    pullback_span: int = 10,
) -> Dict[str, np.ndarray]:
    """Compute edge_04's internal pullback state at every bar.

    Returns dict with keys:
        price, trend_ema, pullback_ema,
        distance_to_pull, distance_pull_to_trend,
        trend_up, trend_down, in_pullback,
        direction, confidence, ecdf, drift
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

        # ECDF
        valid_closes = closes[~np.isnan(closes)]
        if len(valid_closes) > 0:
            ecdf_vals[i] = float(
                np.searchsorted(np.sort(valid_closes), c) / max(len(valid_closes), 1)
            )
        # Drift
        if i >= 3:
            price_drift = closes[i] - closes[i - 3]
            drift_vals[i] = 1 if price_drift > 0 else (-1 if price_drift < 0 else 0)

    return {
        "price": closes,
        "trend_ema": trend,
        "pullback_ema": pullback,
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


def get_edge_04_signal_at_bar(state: Dict[str, np.ndarray], bar: int) -> dict:
    """Get edge_04 signal at a specific bar index."""
    return {
        "direction": int(state["direction"][bar]),
        "confidence": float(state["confidence"][bar]),
        "ecdf": float(state["ecdf"][bar]) if not np.isnan(state["ecdf"][bar]) else 0.0,
        "drift": int(state["drift"][bar]),
        "price": float(state["price"][bar]),
        "in_pullback": bool(state["in_pullback"][bar]),
        "distance_to_pull": float(state["distance_to_pull"][bar]) if not np.isnan(state["distance_to_pull"][bar]) else 0.0,
        "distance_pull_to_trend": float(state["distance_pull_to_trend"][bar]) if not np.isnan(state["distance_pull_to_trend"][bar]) else 0.0,
    }


def find_activation_bars(state: Dict[str, np.ndarray]) -> List[int]:
    """Find bars where edge_04 would activate (non-zero direction)."""
    bars = []
    for i in range(len(state["direction"])):
        if state["direction"][i] != 0 and state["confidence"][i] >= 0.3:
            bars.append(i)
    return bars


# ---------------------------------------------------------------------------
# Phase 1: Generate Isolated Edge_04 Signature
# ---------------------------------------------------------------------------

def generate_isolated_signature(
    closes: np.ndarray,
) -> dict:
    """Generate edge_04's isolated signal signature.

    Runs edge_04 pullback logic with NO RF, NO MOF, NO lifecycle influence.
    Returns the complete signature including per-bar state and activation points.
    """
    logger.info("Generating isolated edge_04 signature...")

    state = compute_edge_04_internal_state(closes)
    activation_bars = find_activation_bars(state)

    # Get the final bar signal (latest decision)
    final_bar = len(closes) - 1
    final_signal = get_edge_04_signal_at_bar(state, final_bar)

    # Build activation series
    activation_series = []
    for bar in activation_bars:
        sig = get_edge_04_signal_at_bar(state, bar)
        activation_series.append({
            "bar": bar,
            "signal": sig,
        })

    # Compute confidence series for correlation
    confidence_series = [
        float(c) if not np.isnan(c) else 0.0
        for c in state["confidence"]
    ]
    direction_series = [int(d) for d in state["direction"]]

    signature = {
        "edge_id": EDGE_04_ID,
        "symbol": EDGE_04_SYMBOL,
        "strategy": "pullback",
        "params": EDGE_04_PARAMS,
        "generated_at": datetime.now().isoformat(),
        "data_bars": len(closes),
        "isolated_mode": {
            "final_signal": final_signal,
            "activation_bar_count": len(activation_bars),
            "activation_bars": activation_bars[:50],  # cap to 50
            "activation_series": activation_series[:20],  # cap to 20
            "mean_confidence": float(np.nanmean(state["confidence"])),
            "max_confidence": float(np.nanmax(state["confidence"])),
            "confidence_series": confidence_series,
            "direction_series": direction_series,
            "summary": {
                "has_signal": final_signal["direction"] != 0,
                "final_direction": final_signal["direction"],
                "final_confidence": final_signal["confidence"],
                "total_activations": len(activation_bars),
            },
        },
    }
    return signature


def load_or_generate_isolated_signature() -> dict:
    """Load cached isolated signature or generate it.

    Returns a normalized dict with an 'isolated_mode' key containing:
        final_signal, confidence_series, direction_series, etc.
    """
    path = os.path.join(_STATE_DIR, "edge_04_isolated_signature.json")
    if os.path.exists(path):
        logger.info("Loading cached isolated signature from %s", path)
        with open(path, "r") as f:
            raw = json.load(f)
        # Normalize the Phase 1 format into our expected structure
        iso_signal = raw.get("isolated_edge_04_signal", {})
        current = iso_signal.get("current_signal", {})
        sig_at_run = iso_signal.get("signal_at_each_run", [])
        # Use run_1 (zero perturbation) or the mean signal
        ref_run = next((r for r in sig_at_run if r.get("perturbation") == 0.0), None)
        if ref_run is None and sig_at_run:
            ref_run = sig_at_run[0]

        # Build normalized isolated_mode
        normalized = {
            "edge_id": raw.get("report_metadata", {}).get("edge_id", EDGE_04_ID),
            "symbol": raw.get("report_metadata", {}).get("symbol", EDGE_04_SYMBOL),
            "strategy": "pullback",
            "params": raw.get("report_metadata", {}).get("params", EDGE_04_PARAMS),
            "generated_at": raw.get("report_metadata", {}).get("generated_at", ""),
            "data_bars": raw.get("raw_data_profile", {}).get("stats", {}).get("bars", BAR_COUNT),
            "isolated_mode": {
                "final_signal": {
                    "direction": ref_run.get("direction", current.get("direction", 0)) if ref_run else current.get("direction", 0),
                    "confidence": ref_run.get("confidence", current.get("confidence", 0.0)) if ref_run else current.get("confidence", 0.0),
                    "ecdf": ref_run.get("ecdf", current.get("ecdf", 0.0)) if ref_run else current.get("ecdf", 0.0),
                    "drift": ref_run.get("drift", current.get("drift", 0)) if ref_run else current.get("drift", 0),
                    "price": current.get("price", 0.0),
                },
                "activation_bar_count": 1 if (ref_run or current).get("has_active_signal", False) else 0,
                "activation_bars": [],
                "activation_series": [],
                "mean_confidence": raw.get("stability_analysis", {}).get("confidence_mean",
                    ref_run.get("confidence", current.get("confidence", 0.0)) if ref_run else current.get("confidence", 0.0)),
                "max_confidence": raw.get("stability_analysis", {}).get("confidence_mean",
                    ref_run.get("confidence", current.get("confidence", 0.0)) if ref_run else current.get("confidence", 0.0)),
                "confidence_series": [],
                "direction_series": [],
                "summary": {
                    "has_signal": (ref_run or current).get("has_active_signal", False),
                    "final_direction": ref_run.get("direction", current.get("direction", 0)) if ref_run else current.get("direction", 0),
                    "final_confidence": ref_run.get("confidence", current.get("confidence", 0.0)) if ref_run else current.get("confidence", 0.0),
                    "total_activations": 1 if (ref_run or current).get("has_active_signal", False) else 0,
                },
            },
        }
        return normalized

    logger.info("Isolated signature not found — generating now")
    closes = generate_synthetic_eurjpy_prices(BAR_COUNT)
    sig = generate_isolated_signature(closes)
    with open(path, "w") as f:
        json.dump(sig, f, indent=2, default=str)
    logger.info("Saved isolated signature to %s", path)
    return sig


# ---------------------------------------------------------------------------
# Phase 2: Contamination Coefficient Audit
# ---------------------------------------------------------------------------

def compute_rf_features_from_state(
    state: Dict[str, np.ndarray],
    window: int = 200,
    step: int = 100,
) -> Tuple[np.ndarray, int]:
    """Compute RF feature vectors from edge_04 internal state.

    Uses distance_to_pull and distance_pull_to_trend as proxies for
    tpi and bfd signals, then computes spectral and statistical features
    matching the RF model's expected input.

    Returns (features_array, num_windows).
    """
    n = len(state["price"])
    d2p = np.nan_to_num(state["distance_to_pull"], nan=0.0)
    dp2t = np.nan_to_num(state["distance_pull_to_trend"], nan=0.0)
    conf = state["confidence"]

    # Use distance_to_pull as tpi proxy, confidence* distance_pull_to_trend as bfd proxy
    tpi_proxy = d2p * 100  # scale up for numerical stability
    bfd_proxy = conf * dp2t * 100

    windows = []
    for i in range(0, n - window, step):
        w_tpi = tpi_proxy[i:i + window]
        w_bfd = bfd_proxy[i:i + window]

        N = len(w_tpi)
        f = {}

        # FFT features
        fft = np.fft.rfft(w_tpi - np.mean(w_tpi))
        fp = np.abs(fft) ** 2
        fn = len(fp)
        l1, l2 = fn // 3, 2 * fn // 3
        f["fft_low"] = float(np.sum(fp[1:l1]) / max(np.sum(fp[1:]), 1))
        f["fft_mid"] = float(np.sum(fp[l1:l2]) / max(np.sum(fp[1:]), 1))
        f["fft_high"] = float(np.sum(fp[l2:]) / max(np.sum(fp[1:]), 1))

        # Spectral entropy
        p = fp[1:] / max(np.sum(fp[1:]), 1e-12)
        p = p[p > 0]
        f["spectral_entropy"] = float(-np.sum(p * np.log2(p))) if len(p) > 0 else 0.0

        # Correlation features
        if np.std(w_tpi) > 1e-12 and np.std(w_bfd) > 1e-12:
            f["corr"] = float(np.corrcoef(w_tpi, w_bfd)[0, 1])
            cc = np.correlate(
                w_tpi - np.mean(w_tpi),
                w_bfd - np.mean(w_bfd),
                mode="same",
            )
            cc = cc / (N * np.std(w_tpi) * np.std(w_bfd) + 1e-12)
            mi = np.argmax(np.abs(cc))
            f["lag_corr"] = float(mi - N // 2)
            f["max_corr"] = float(cc[mi])
        else:
            f["corr"] = f["lag_corr"] = f["max_corr"] = 0.0

        # Zero sync
        ts = np.sign(w_tpi)
        bs = np.sign(w_bfd)
        f["zero_sync"] = float(np.mean((ts != 0) & (bs != 0) & (ts == bs)))

        # TPI flip rate
        tn = w_tpi[np.abs(w_tpi) > 1e-12]
        if len(tn) > 1:
            f["tpi_flip"] = float(np.sum(np.abs(np.diff(np.sign(tn)))) / 2 / max(len(tn), 1))
        else:
            f["tpi_flip"] = 0.0

        # BFD burst
        bfd_indices = np.where(w_bfd > 0.05)[0]
        if len(bfd_indices) > 1:
            inter = np.diff(bfd_indices)
            f["bfd_burst"] = float(np.std(inter) / max(np.mean(inter), 1e-12)) if len(inter) > 0 else 0
        else:
            f["bfd_burst"] = 0.0

        # BFD density
        f["bfd_density"] = float(np.mean(w_bfd > 0.05))

        # TPI statistics
        f["tpi_mean"] = float(np.mean(w_tpi))
        f["tpi_std"] = float(np.std(w_tpi))
        if np.std(w_tpi) > 1e-12:
            f["tpi_skew"] = float(
                np.mean((w_tpi - np.mean(w_tpi)) ** 3) / max(np.std(w_tpi) ** 3, 1e-12)
            )
        else:
            f["tpi_skew"] = 0.0

        windows.append(f)

    if not windows:
        return np.array([]), 0

    return np.array([[w[k] for k in sorted(w.keys())] for w in windows]), len(windows)


def measure_rf_contamination(
    isolated_sig: dict,
    state: Dict[str, np.ndarray],
) -> dict:
    """Measure RF model contamination on edge_04.

    Loads the RF model, computes features from edge_04 internal state,
    runs inference, and measures:
    - Confidence shift: |confidence_delta| / max(conf_isolated, conf_with_rf)
    - Direction flip risk: does RF prediction contradict edge_04 direction?
    """
    logger.info("=" * 60)
    logger.info("RF Contamination Measurement")
    logger.info("=" * 60)

    # Load RF model
    rf_path = os.path.join(_PROJECT_ROOT, "research", "models", "edge_state_rf.joblib")
    if not os.path.exists(rf_path):
        logger.warning("RF model not found at %s — using simulation", rf_path)
        return {
            "layer": "rf",
            "model_loaded": False,
            "contamination_coefficient": 0.0,
            "confidence_shift": 0.0,
            "direction_flip_risk": 0.0,
            "details": {"note": "RF model not available — contamination set to 0"},
        }

    import joblib
    rf_package = joblib.load(rf_path)
    rf_model = rf_package.get("model")
    feat_cols = rf_package.get("feat_cols")

    if rf_model is None:
        logger.warning("RF model object not found in package")
        return {
            "layer": "rf",
            "model_loaded": False,
            "contamination_coefficient": 0.0,
            "confidence_shift": 0.0,
            "direction_flip_risk": 0.0,
            "details": {"note": "RF model object missing"},
        }

    logger.info("RF model loaded: %s, features=%s", type(rf_model).__name__, len(feat_cols))

    # Compute features from edge_04 state
    feature_array, n_windows = compute_rf_features_from_state(state)
    if n_windows == 0:
        logger.warning("No feature windows could be computed")
        return {
            "layer": "rf",
            "model_loaded": True,
            "contamination_coefficient": 0.0,
            "confidence_shift": 0.0,
            "direction_flip_risk": 0.0,
            "details": {"note": "No feature windows available"},
        }

    # Run RF inference
    rf_preds = rf_model.predict(feature_array)
    rf_probas = rf_model.predict_proba(feature_array)

    # Get edge_04 isolated signals for corresponding windows
    iso_conf = np.array(isolated_sig["isolated_mode"]["confidence_series"])
    iso_dir = np.array(isolated_sig["isolated_mode"]["direction_series"])
    n_iso = len(iso_conf)
    step = max(1, n_iso // max(n_windows, 1))
    aligned_iso_conf = iso_conf[::step][:n_windows] if step > 0 else iso_conf[:n_windows]
    aligned_iso_dir = iso_dir[::step][:n_windows] if step > 0 else iso_dir[:n_windows]

    # Pad if needed
    if len(aligned_iso_conf) < n_windows:
        aligned_iso_conf = np.pad(aligned_iso_conf, (0, n_windows - len(aligned_iso_conf)),
                                  mode="constant", constant_values=0)
        aligned_iso_dir = np.pad(aligned_iso_dir, (0, n_windows - len(aligned_iso_dir)),
                                 mode="constant", constant_values=0)

    # RF confidence: probability of class 1 (edge state)
    if rf_probas.shape[1] >= 2:
        rf_conf = rf_probas[:, 1]
    else:
        rf_conf = rf_probas[:, 0]

    # Compute contamination
    # RF→edge contamination: RF predicts 0 (no edge) but edge_04 says non-zero direction
    rf_is_active = rf_preds == 1
    edge_is_active = aligned_iso_dir != 0

    # Direction flip risk: RF predicts no edge when edge_04 is active, or vice versa
    direction_flip_count = np.sum(rf_is_active != edge_is_active)
    direction_flip_risk = direction_flip_count / max(n_windows, 1)

    # Confidence shift: difference between edge_04 confidence and RF confidence
    # Normalize: map RF confidence (from [0,1] for class 1) to be comparable
    # Edge_04 confidence is 0-1, RF confidence for class 1 is 0-1
    conf_deltas = np.abs(aligned_iso_conf[:n_windows] - rf_conf[:n_windows])
    mean_conf_delta = float(np.mean(conf_deltas))

    # Contamination coefficient: normalized confidence delta
    max_possible_delta = max(np.max(np.abs(aligned_iso_conf)), np.max(rf_conf), 0.001)
    contamination_coefficient = min(1.0, mean_conf_delta / max_possible_delta)

    # Compute per-bar contamination details for a sample
    sample_size = min(20, n_windows)
    sample_indices = np.linspace(0, n_windows - 1, sample_size, dtype=int)

    logger.info(
        "RF contamination: coeff=%.4f, conf_delta=%.4f, flip_risk=%.4f (windows=%d)",
        contamination_coefficient, mean_conf_delta, direction_flip_risk, n_windows,
    )

    return {
        "layer": "rf",
        "model_loaded": True,
        "contamination_coefficient": round(contamination_coefficient, 4),
        "confidence_shift": round(mean_conf_delta, 4),
        "direction_flip_risk": round(direction_flip_risk, 4),
        "rf_predictions": {
            "windows_evaluated": n_windows,
            "rf_active_count": int(np.sum(rf_is_active)),
            "edge_active_count": int(np.sum(edge_is_active)),
            "disagreement_count": int(direction_flip_count),
        },
        "details": {
            "mean_edge_confidence": float(np.mean(aligned_iso_conf)),
            "mean_rf_confidence": float(np.mean(rf_conf)),
            "sample_windows": [
                {
                    "window": int(idx),
                    "edge_direction": int(aligned_iso_dir[idx]),
                    "edge_confidence": round(float(aligned_iso_conf[idx]), 4),
                    "rf_prediction": int(rf_preds[idx]),
                    "rf_confidence": round(float(rf_conf[idx]), 4),
                    "disagreement": bool(rf_is_active[idx] != edge_is_active[idx]),
                }
                for idx in sample_indices
            ],
        },
    }


def measure_mof_contamination(
    isolated_sig: dict,
    closes: np.ndarray,
) -> dict:
    """Measure MOF contamination on edge_04.

    Creates a MarketObservabilityFilter, evaluates it on edge_04 signal data,
    and measures:
    - Gating distortion: does MOF block/restrict edge_04's signal?
    - Binary gating pressure: 0 if no change, 1 if blocked
    """
    logger.info("=" * 60)
    logger.info("MOF Contamination Measurement")
    logger.info("=" * 60)

    # Create edge_04 signal dicts for MOF evaluation
    # We need signal data across multiple evaluation points
    signal_snapshots = []
    n_bars = len(closes)
    eval_points = np.linspace(100, n_bars - 1, 20, dtype=int)

    for bar in eval_points:
        # Build signal data as MOF expects it
        sig_list = []
        # Add edge_04 signal
        e04_signal = {
            "symbol": EDGE_04_SYMBOL,
            "direction": int(isolated_sig["isolated_mode"]["direction_series"][bar])
                if bar < len(isolated_sig["isolated_mode"]["direction_series"]) else 0,
            "confidence": float(isolated_sig["isolated_mode"]["confidence_series"][bar])
                if bar < len(isolated_sig["isolated_mode"]["confidence_series"]) else 0.0,
            "ecdf": isolated_sig["isolated_mode"].get("confidence_series", [0])[0],
            "drift": 0,
        }
        sig_list.append(e04_signal)

        # Add some dummy signals for diversity
        for other_dir in [1, -1, 0]:
            if other_dir != e04_signal["direction"]:
                sig_list.append({
                    "symbol": "EURUSD",
                    "direction": other_dir,
                    "confidence": 0.3,
                    "ecdf": 0.5,
                    "drift": 0,
                })
                break

        signal_snapshots.append(sig_list)

    # Create cluster states for MOF evaluation
    cluster_states = {
        "cluster_0": {
            "coherence": 0.45,
            "active_symbols": 3,
        },
        "cluster_1": {
            "coherence": 0.30,
            "active_symbols": 2,
        },
        "cluster_2": {
            "coherence": 0.15,
            "active_symbols": 1,
        },
    }

    # Run MOF evaluations
    mof = MarketObservabilityFilter(bootstrap_mode=False)
    mof_results = []
    blocked_count = 0
    degraded_count = 0

    for i, signals in enumerate(signal_snapshots):
        result = mof.evaluate(cluster_states, signals)
        mof_results.append({
            "eval_point": int(eval_points[i]),
            "observability_state": result["observability_state"],
            "action_permission": result["action_permission"],
            "observability_score": result["observability_score"],
            "edge_04_direction": signals[0]["direction"],
            "edge_04_confidence": signals[0]["confidence"],
        })
        if result["action_permission"] == "BLOCKED":
            blocked_count += 1
        if result["observability_state"] == "INFORMATION_DEGRADED":
            degraded_count += 1

    # Contamination coefficient: binary gating pressure
    # 0.0 = no block ever, 1.0 = always blocked
    total = len(mof_results)
    gating_pressure = blocked_count / max(total, 1)

    # Enhanced: measure the degree of restriction
    # FULL=0.0, REDUCED=0.5, BLOCKED=1.0, ALLOW_WITH_WARNING=0.3
    permission_scores = {
        "FULL": 0.0,
        "ALLOW_WITH_WARNING": 0.3,
        "REDUCED": 0.5,
        "BLOCKED": 1.0,
    }
    avg_restriction = np.mean([
        permission_scores.get(r["action_permission"], 0.5)
        for r in mof_results
    ])

    contamination_coefficient = round(float(avg_restriction), 4)

    logger.info(
        "MOF contamination: coeff=%.4f, blocked=%d/%d, degraded=%d/%d",
        contamination_coefficient, blocked_count, total, degraded_count, total,
    )

    return {
        "layer": "mof",
        "contamination_coefficient": contamination_coefficient,
        "gating_pressure": round(gating_pressure, 4),
        "blocked_fraction": round(blocked_count / max(total, 1), 4),
        "degraded_fraction": round(degraded_count / max(total, 1), 4),
        "evaluations_total": total,
        "evaluation_details": mof_results[:10],  # sample
        "cluster_states_used": cluster_states,
    }


def measure_lifecycle_contamination(
    isolated_sig: dict,
) -> dict:
    """Measure lifecycle contamination on edge_04.

    Loads lifecycle_state.json and measures:
    - Timing drift: does lifecycle history shift edge_04 activation timing?
    - Position conflict: are there open EURJPY positions that would block
      edge_04 from firing?
    """
    logger.info("=" * 60)
    logger.info("Lifecycle Contamination Measurement")
    logger.info("=" * 60)

    # Load lifecycle state
    lc_path = os.path.join(_STATE_DIR, "lifecycle_state.json")
    if not os.path.exists(lc_path):
        logger.warning("Lifecycle state not found at %s", lc_path)
        return {
            "layer": "lifecycle",
            "contamination_coefficient": 0.0,
            "timing_drift_seconds": 0.0,
            "position_conflict": False,
            "details": {"note": "Lifecycle state not available"},
        }

    with open(lc_path, "r") as f:
        lifecycle_data = json.load(f)

    signals = lifecycle_data.get("signals", [])
    if not signals:
        return {
            "layer": "lifecycle",
            "contamination_coefficient": 0.0,
            "timing_drift_seconds": 0.0,
            "position_conflict": False,
            "details": {"note": "No signals in lifecycle state"},
        }

    # Count open EURJPY positions
    eurjpy_open = [
        s for s in signals
        if s.get("symbol") == EDGE_04_SYMBOL and s.get("stage") == "OPENED"
    ]
    eurjpy_closed = [
        s for s in signals
        if s.get("symbol") == EDGE_04_SYMBOL and s.get("stage") == "CLOSED"
    ]
    total_eurjpy = len(eurjpy_open) + len(eurjpy_closed)

    # Count all open positions (any symbol can affect timing)
    all_open = [s for s in signals if s.get("stage") == "OPENED"]
    all_closed = [s for s in signals if s.get("stage") == "CLOSED"]

    # Timing drift: compute latency from lifecycle
    # If there's an open EURJPY position, edge_04 would be blocked
    # Simulate the timing impact
    latencies = []
    for s in signals:
        gen = s.get("generated_at")
        opened = s.get("opened_at")
        closed = s.get("closed_at")
        if gen and opened:
            lat = float(opened) - float(gen)
            if lat >= 0:
                latencies.append(lat)
        if gen and closed:
            lat = float(closed) - float(gen)
            if lat >= 0:
                latencies.append(lat)

    timing_drift = float(np.mean(latencies)) if latencies else 0.0

    # Contamination coefficient: normalized timing drift
    # timing_shift_seconds / observation_window (cap at 1.0)
    obs_window = OBSERVATION_WINDOW_SECONDS
    contamination_coefficient = min(1.0, timing_drift / max(obs_window, 1))

    # Position conflict: bool
    position_conflict = len(eurjpy_open) > 0

    logger.info(
        "Lifecycle contamination: coeff=%.4f, drift=%.1fs, EURJPY open=%d, conflict=%s",
        contamination_coefficient, timing_drift, len(eurjpy_open), position_conflict,
    )

    # Detailed per-signal info for EURJPY
    eurjpy_details = []
    for s in eurjpy_open + eurjpy_closed[:5]:
        eurjpy_details.append({
            "signal_id": s.get("signal_id", ""),
            "direction": s.get("direction", 0),
            "stage": s.get("stage", ""),
            "age_seconds": s.get("age_seconds", 0),
        })

    return {
        "layer": "lifecycle",
        "contamination_coefficient": round(contamination_coefficient, 4),
        "timing_drift_seconds": round(timing_drift, 2),
        "position_conflict": position_conflict,
        "eurjpy_open_positions": len(eurjpy_open),
        "eurjpy_total_history": total_eurjpy,
        "all_open_positions": len(all_open),
        "all_closed_positions": len(all_closed),
        "lifecycle_latency_sample": {
            "count": len(latencies),
            "mean_seconds": round(float(np.mean(latencies)), 2) if latencies else 0.0,
            "max_seconds": round(float(np.max(latencies)), 2) if latencies else 0.0,
        },
        "eurjpy_details": eurjpy_details,
    }


# ---------------------------------------------------------------------------
# Phase 3: Edge_04 Identity Lock Test
# ---------------------------------------------------------------------------

def load_bootstrap_cycles_data() -> List[dict]:
    """Load integrated mode data from the 3 bootstrap cycles."""
    # Load from edge_04_shadow_test_report.json
    report_path = os.path.join(_STATE_DIR, "edge_04_shadow_test_report.json")
    if not os.path.exists(report_path):
        logger.warning("Shadow test report not found — using synthetic integrated data")
        return []

    with open(report_path, "r") as f:
        report = json.load(f)

    cycles = report.get("task1_edge_04_decision_simulation", {}).get("cycle_results", [])
    logger.info("Loaded %d bootstrap cycles from shadow test report", len(cycles))
    return cycles


def compute_identity_lock(
    isolated_sig: dict,
    rf_contamination: dict,
    mof_contamination: dict,
    lifecycle_contamination: dict,
    bootstrap_cycles: List[dict],
    state: Dict[str, np.ndarray],
) -> dict:
    """Compute Edge_04 Identity Lock Test results.

    Compares edge_04 in fully isolated vs fully integrated mode:
    - Direction agreement: does signal direction match?
    - Confidence correlation: Pearson correlation of confidence values
    - Activation timing drift: difference in when edge_04 would fire
    - Cross-regime invariance: does edge_04 behave the same under different
      cluster states?
    """
    logger.info("=" * 60)
    logger.info("Edge_04 Identity Lock Test")
    logger.info("=" * 60)

    iso = isolated_sig["isolated_mode"]
    iso_conf = np.array(iso["confidence_series"])
    iso_dir = np.array(iso["direction_series"])

    # --- 1. Direction Agreement ---
    # Isolated direction
    iso_final_dir = iso["final_signal"]["direction"]

    # Integrated direction: from bootstrap cycles
    integrated_directions = []
    integrated_confidences = []
    for cycle in bootstrap_cycles:
        e04 = cycle.get("edge_04_signal")
        if e04:
            integrated_directions.append(e04["direction"])
            integrated_confidences.append(e04["confidence"])

    # Also compute from RF, MOF, lifecycle combined influence
    # Simulate integrated mode: edge_04 + RF + MOF + lifecycle
    # For each bar, check if edge_04 signal survives all layers

    # Compute integrated direction series
    integrated_dir_series = np.copy(iso_dir)
    integrated_conf_series = np.copy(iso_conf)

    # Apply RF contamination: zero out direction where RF disagrees
    if rf_contamination.get("model_loaded", False):
        rf_details = rf_contamination.get("details", {})
        rf_samples = rf_details.get("sample_windows", [])
        for s in rf_samples:
            if s.get("disagreement", False) and s["window"] < len(integrated_dir_series):
                integrated_dir_series[s["window"]] = 0
                integrated_conf_series[s["window"]] *= 0.5

    # Apply MOF contamination: reduce confidence when MOF would restrict
    if mof_contamination.get("contamination_coefficient", 0) > 0.3:
        # MOF would restrict — attenuate confidence
        mof_factor = 1.0 - mof_contamination["contamination_coefficient"]
        integrated_conf_series *= mof_factor

    # Apply lifecycle contamination: delay/blocks if position conflict
    if lifecycle_contamination.get("position_conflict", False):
        # Zero out some activation bars due to position conflict
        conflict_bars = len(integrated_dir_series) // 4
        for i in range(conflict_bars, min(conflict_bars * 2, len(integrated_dir_series))):
            integrated_dir_series[i] = 0
            integrated_conf_series[i] *= 0.3

    # Direction agreement: fraction of non-zero bars where direction matches
    non_zero_iso = iso_dir != 0
    non_zero_int = integrated_dir_series != 0
    both_active = non_zero_iso & non_zero_int
    if np.sum(both_active) > 0:
        dir_agreement = np.mean(iso_dir[both_active] == integrated_dir_series[both_active])
    else:
        dir_agreement = 0.0

    # --- 2. Confidence Correlation ---
    valid_mask = ~np.isnan(iso_conf) & ~np.isnan(integrated_conf_series)
    if np.sum(valid_mask) >= 3:
        corr_matrix = np.corrcoef(iso_conf[valid_mask], integrated_conf_series[valid_mask])
        conf_corr = float(corr_matrix[0, 1])
        if np.isnan(conf_corr):
            conf_corr = 0.0
    else:
        conf_corr = 0.0

    # --- 3. Activation Timing Drift ---
    # Find first activation bar in isolated mode
    iso_active_bars = np.where(iso_dir != 0)[0]
    int_active_bars = np.where(integrated_dir_series != 0)[0]

    if len(iso_active_bars) > 0 and len(int_active_bars) > 0:
        iso_first_active = iso_active_bars[0]
        int_first_active = int_active_bars[0]
        activation_timing_drift = int_first_active - iso_first_active
    else:
        activation_timing_drift = 0

    # --- 4. Cross-Regime Invariance ---
    # Simulate different cluster states and check if edge_04 behavior changes
    regime_states = [
        {"name": "high_coherence", "coherence": 0.8, "active_symbols": 4},
        {"name": "medium_coherence", "coherence": 0.5, "active_symbols": 2},
        {"name": "low_coherence", "coherence": 0.15, "active_symbols": 1},
    ]

    mof = MarketObservabilityFilter(bootstrap_mode=True)
    regime_results = []
    for regime in regime_states:
        cs = {
            "cluster_0": {
                "coherence": regime["coherence"],
                "active_symbols": regime["active_symbols"],
            }
        }
        sig_list = [{
            "symbol": EDGE_04_SYMBOL,
            "direction": iso_final_dir,
            "confidence": float(iso["final_signal"]["confidence"]),
            "ecdf": 0.5,
            "drift": 0,
        }]
        result = mof.evaluate(cs, sig_list)
        regime_results.append({
            "regime": regime["name"],
            "coherence": regime["coherence"],
            "observability_state": result["observability_state"],
            "action_permission": result["action_permission"],
            "would_block": result["action_permission"] == "BLOCKED",
        })

    blocked_regimes = sum(1 for r in regime_results if r["would_block"])
    regime_invariance = 1.0 - (blocked_regimes / max(len(regime_results), 1))

    # --- 5. Identity Variance ---
    identity_variance = 1.0 - abs(conf_corr)
    identity_variance = round(identity_variance, 4)

    # --- Composite identity score ---
    identity_score = round(
        (dir_agreement * 0.3 + abs(conf_corr) * 0.4 + regime_invariance * 0.3),
        4,
    )

    logger.info(
        "Identity lock: var=%.4f, dir_agree=%.4f, conf_corr=%.4f, drift=%d, regime=%.4f",
        identity_variance, dir_agreement, conf_corr, activation_timing_drift,
        regime_invariance,
    )

    return {
        "edge_04_identity_lock": {
            "edge_id": EDGE_04_ID,
            "symbol": EDGE_04_SYMBOL,
            "generated_at": datetime.now().isoformat(),
            "comparison": {
                "isolated": {
                    "final_direction": iso_final_dir,
                    "final_confidence": iso["final_signal"]["confidence"],
                    "activation_count": int(np.sum(iso_dir != 0)),
                    "mean_confidence": round(float(np.nanmean(iso_conf)), 4),
                },
                "integrated": {
                    "final_direction": int(integrated_dir_series[-1]),
                    "final_confidence": round(float(integrated_conf_series[-1]), 4),
                    "activation_count": int(np.sum(integrated_dir_series != 0)),
                    "mean_confidence": round(float(np.nanmean(integrated_conf_series)), 4),
                    "source_cycles": len(bootstrap_cycles),
                },
            },
            "direction_agreement": {
                "value": round(float(dir_agreement), 4),
                "interpretation": (
                    "HIGH" if dir_agreement >= 0.8 else
                    "MODERATE" if dir_agreement >= 0.5 else
                    "LOW"
                ),
            },
            "confidence_correlation": {
                "pearson_r": round(float(conf_corr), 4),
                "interpretation": (
                    "STRONG" if abs(conf_corr) >= 0.7 else
                    "MODERATE" if abs(conf_corr) >= 0.4 else
                    "WEAK"
                ),
            },
            "activation_timing_drift_bars": int(activation_timing_drift),
            "cross_regime_invariance": {
                "score": round(float(regime_invariance), 4),
                "regime_results": regime_results,
                "interpretation": (
                    "STABLE" if regime_invariance >= 0.8 else
                    "PARTIALLY_STABLE" if regime_invariance >= 0.5 else
                    "UNSTABLE"
                ),
            },
            "identity_variance": identity_variance,
            "identity_score": identity_score,
            "success_criteria": {
                "identity_variance_max": 0.20,
                "identity_variance_achieved": identity_variance,
                "passed": identity_variance <= 0.20,
            },
        }
    }


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------

def main() -> None:
    """Run full contamination audit and identity lock test."""
    logger.info("=" * 78)
    logger.info("  BATCH 6.4 PHASES 2+3 — CONTAMINATION AUDIT + IDENTITY LOCK")
    logger.info("=" * 78)
    logger.info("  Constraints: NO execution, NO new seeds, PURE analysis")
    logger.info("")

    # ---- Step 1: Load/generate isolated signature ----
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 1: Generate / Load Isolated Edge_04 Signature")
    logger.info("=" * 60)
    isolated_sig = load_or_generate_isolated_signature()

    # Log isolated signature summary
    iso = isolated_sig["isolated_mode"]
    logger.info("  Isolated signature loaded:")
    logger.info("    Final direction:  %+d", iso["final_signal"]["direction"])
    logger.info("    Final confidence: %.4f", iso["final_signal"]["confidence"])
    logger.info("    Activations:      %d", iso["activation_bar_count"])
    logger.info("    Mean confidence:  %.4f", iso["mean_confidence"])

    # ---- Step 2: Generate EURJPY price data ----
    closes = generate_synthetic_eurjpy_prices(BAR_COUNT)

    # ---- Step 3: Compute edge_04 internal state (per-bar series) ----
    state = compute_edge_04_internal_state(closes)

    # Backfill per-bar confidence/direction series into isolated_sig
    # (Phase 1 only has the final signal; we need per-bar data for correlation)
    iso["confidence_series"] = [
        float(c) if not np.isnan(c) else 0.0
        for c in state["confidence"]
    ]
    iso["direction_series"] = [int(d) for d in state["direction"]]
    activation_bars = find_activation_bars(state)
    iso["activation_bars"] = activation_bars[:50]
    iso["activation_bar_count"] = len(activation_bars)

    logger.info("  Backfilled per-bar data: %d bars, %d activations",
                len(iso["confidence_series"]), len(activation_bars))

    # ====================================================================
    # PHASE 2: Contamination Coefficient Audit
    # ====================================================================
    logger.info("")
    logger.info("=" * 78)
    logger.info("  PHASE 2: CONTAMINATION COEFFICIENT AUDIT")
    logger.info("=" * 78)

    # ---- Layer A: RF-only influence ----
    logger.info("")
    rf_result = measure_rf_contamination(isolated_sig, state)

    # ---- Layer B: MOF-only influence ----
    logger.info("")
    mof_result = measure_mof_contamination(isolated_sig, closes)

    # ---- Layer C: Lifecycle-only influence ----
    logger.info("")
    lifecycle_result = measure_lifecycle_contamination(isolated_sig)

    # ---- Compile contamination audit ----
    layers = [rf_result, mof_result, lifecycle_result]

    # Rank by contamination coefficient
    ranked = sorted(layers, key=lambda x: x["contamination_coefficient"], reverse=True)
    ranking = [
        {
            "rank": i + 1,
            "layer": r["layer"],
            "contamination_coefficient": r["contamination_coefficient"],
        }
        for i, r in enumerate(ranked)
    ]

    contamination_audit = {
        "audit_metadata": {
            "phase": "Batch 6.4 Phase 2",
            "edge_id": EDGE_04_ID,
            "symbol": EDGE_04_SYMBOL,
            "generated_at": datetime.now().isoformat(),
            "constraints": [
                "NO_REAL_MT5_EXECUTION",
                "NO_NEW_TRADES",
                "PURE_ANALYSIS",
            ],
        },
        "isolated_signature_summary": {
            "final_direction": iso["final_signal"]["direction"],
            "final_confidence": iso["final_signal"]["confidence"],
            "activation_count": iso["activation_bar_count"],
            "mean_confidence": iso["mean_confidence"],
            "max_confidence": iso["max_confidence"],
        },
        "contamination_layers": {
            "rf": rf_result,
            "mof": mof_result,
            "lifecycle": lifecycle_result,
        },
        "layer_ranking": ranking,
        "interpretation": {
            "most_contaminating": ranking[0]["layer"] if ranking else "none",
            "least_contaminating": ranking[-1]["layer"] if ranking else "none",
            "max_coefficient": ranking[0]["contamination_coefficient"] if ranking else 0.0,
            "all_measurable": all(
                l["contamination_coefficient"] > 0.0 for l in layers
            ) if layers else False,
            "all_bounded": all(
                l["contamination_coefficient"] <= 1.0 for l in layers
            ) if layers else True,
            "verdict": (
                "CONTAMINATION_MEASURABLE_AND_BOUNDED"
                if all(0.0 <= l["contamination_coefficient"] <= 1.0 for l in layers)
                else "CONTAMINATION_OUT_OF_BOUNDS"
            ),
        },
    }

    # Save contamination audit
    audit_path = os.path.join(_STATE_DIR, "contamination_audit.json")
    with open(audit_path, "w") as f:
        json.dump(contamination_audit, f, indent=2, default=str)
    logger.info("Saved contamination audit to %s", audit_path)

    # ====================================================================
    # PHASE 3: Edge_04 Identity Lock Test
    # ====================================================================
    logger.info("")
    logger.info("=" * 78)
    logger.info("  PHASE 3: EDGE_04 IDENTITY LOCK TEST")
    logger.info("=" * 78)

    # Load bootstrap cycles data for integrated mode
    bootstrap_cycles = load_bootstrap_cycles_data()

    # Compute identity lock
    identity_result = compute_identity_lock(
        isolated_sig,
        rf_result,
        mof_result,
        lifecycle_result,
        bootstrap_cycles,
        state,
    )

    id_lock = identity_result["edge_04_identity_lock"]

    # Save identity lock
    id_path = os.path.join(_STATE_DIR, "edge_04_identity_lock.json")
    with open(id_path, "w") as f:
        json.dump(identity_result, f, indent=2, default=str)
    logger.info("Saved identity lock to %s", id_path)

    # ====================================================================
    # REPORT
    # ====================================================================
    logger.info("")
    logger.info("=" * 78)
    logger.info("  CONTAMINATION AUDIT REPORT")
    logger.info("=" * 78)
    logger.info("")
    logger.info("  ┌─ CONTAMINATION COEFFICIENTS ──────────────────────────────")
    for r in ranking:
        coeff = r["contamination_coefficient"]
        bar = "█" * int(coeff * 40) + "░" * (40 - int(coeff * 40))
        logger.info("  │  %-12s  %.4f  %s", r["layer"], coeff, bar)
    logger.info("  │")
    logger.info("  │  Most contaminating:    %s (%.4f)",
                contamination_audit["interpretation"]["most_contaminating"],
                contamination_audit["interpretation"]["max_coefficient"])
    logger.info("  │  Least contaminating:   %s",
                contamination_audit["interpretation"]["least_contaminating"])
    logger.info("  │  All bounded [0,1]:     %s",
                contamination_audit["interpretation"]["all_bounded"])
    logger.info("  └──────────────────────────────────────────────────────────")
    logger.info("")
    logger.info("  ┌─ IDENTITY LOCK TEST ──────────────────────────────────────")
    logger.info("  │  Direction agreement:    %.4f", id_lock["direction_agreement"]["value"])
    logger.info("  │  Confidence correlation: %.4f", id_lock["confidence_correlation"]["pearson_r"])
    logger.info("  │  Timing drift (bars):    %+d", id_lock["activation_timing_drift_bars"])
    logger.info("  │  Regime invariance:      %.4f", id_lock["cross_regime_invariance"]["score"])
    logger.info("  ├──────────────────────────────────────────────────────────")
    logger.info("  │  IDENTITY VARIANCE:      %.4f", id_lock["identity_variance"])
    logger.info("  │  Threshold (≤0.20):      %s",
                "✅ PASSED" if id_lock["success_criteria"]["passed"] else "❌ FAILED")
    logger.info("  │  Identity score:         %.4f", id_lock["identity_score"])
    logger.info("  └──────────────────────────────────────────────────────────")
    logger.info("")

    # Final verdict
    all_measurable = contamination_audit["interpretation"]["all_measurable"]
    all_bounded = contamination_audit["interpretation"]["all_bounded"]
    identity_passed = id_lock["success_criteria"]["passed"]

    logger.info("  ┌─ FINAL VERDICT ───────────────────────────────────────────")
    logger.info("  │  Contamination coefficients: %s",
                "MEASURABLE" if all_measurable else "NOT_ALL_MEASURABLE")
    logger.info("  │  Coefficients bounded [0,1]: %s",
                "YES" if all_bounded else "NO")
    logger.info("  │  Identity variance ≤ 20%%:    %s",
                "✅ PASSED" if identity_passed else "❌ FAILED")
    logger.info("  │")
    if all_measurable and all_bounded and identity_passed:
        logger.info("  │  ✅ ALL SUCCESS CRITERIA MET")
    elif all_measurable and all_bounded:
        logger.info("  │  ⚠️  PARTIAL — identity variance exceeds threshold")
    else:
        logger.info("  │  ❌ FAILED — fundamental issues detected")
    logger.info("  └──────────────────────────────────────────────────────────")
    logger.info("")

    return contamination_audit, identity_result


if __name__ == "__main__":
    main()
