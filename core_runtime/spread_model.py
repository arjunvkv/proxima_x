"""
Spread Model — Replace hard cutoff with probabilistic sigmoid penalty.

The problem:
  Hard cutoff (if spread > threshold → reject) is an ALPHA_DESTROYER.
  It rejects structurally valid signals due to miscalibrated thresholds.

The fix:
  penalty = sigmoid(spread - threshold, steepness)
  adjusted_score = raw_score * penalty

This allows signals with slightly elevated spreads to still execute,
with gradually decreasing weight instead of a hard wall.

Also computes spread distribution metrics for calibration.
"""
import os
import json
import math
import time
import logging
from collections import defaultdict, deque
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

import numpy as np

logger = logging.getLogger("spread_model")


def sigmoid_penalty(x: float, center: float = 0.0, steepness: float = 1.0) -> float:
    """
    Sigmoid penalty function.

    Args:
        x: The input value (e.g., spread - threshold)
        center: Center point (typically spread - threshold, so 0)
        steepness: How sharp the cutoff is (lower = softer)

    Returns:
        penalty: 0.0 to 1.0 (1.0 = no penalty when x << 0)
    """
    return 1.0 / (1.0 + math.exp(steepness * (x - center)))


def linear_decay_penalty(spread: float, soft_threshold: float,
                         hard_threshold: float) -> float:
    """
    Linear decay penalty between soft and hard thresholds.

    spread <= soft_threshold: penalty = 1.0
    soft < spread < hard: penalty linearly decays 1.0 -> 0.0
    spread >= hard_threshold: penalty = 0.0
    """
    if spread <= soft_threshold:
        return 1.0
    if spread >= hard_threshold:
        return 0.0
    return 1.0 - (spread - soft_threshold) / (hard_threshold - soft_threshold)


class SpreadModel:
    """
    Probabilistic spread filter with adaptive threshold calibration.

    Two modes:
      1. sigmoid: smooth penalty via sigmoid(spread - threshold)
      2. linear_decay: linear decay between soft and hard thresholds

    Tracks spread distribution per symbol for automatic threshold calibration.
    """

    def __init__(self,
                 mode: str = "sigmoid",
                 default_steepness: float = 2.0,
                 calibration_percentile: float = 85.0,
                 max_history: int = 10000):
        """
        Args:
            mode: "sigmoid" or "linear_decay"
            default_steepness: Sigmoid steepness (higher = sharper cutoff)
            calibration_percentile: Percentile for auto-calibrating threshold
            max_history: Max spread samples to keep per symbol
        """
        assert mode in ("sigmoid", "linear_decay"), f"Unknown mode: {mode}"
        self._mode = mode
        self._default_steepness = default_steepness
        self._calibration_percentile = calibration_percentile
        self._max_history = max_history

        # Per-symbol spread history
        self._spread_histories: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=max_history)
        )

        # Per-symbol calibrated thresholds (auto-updating)
        self._calibrated_thresholds: Dict[str, float] = {}

        # Per-symbol stats
        self._symbol_stats: Dict[str, dict] = defaultdict(
            lambda: {"samples": 0, "min": float('inf'), "max": 0.0,
                     "sum": 0.0, "sum_sq": 0.0}
        )

        # Rejection tracking
        self._total_evaluations = 0
        self._total_penalized = 0  # signals where penalty < 1.0
        self._total_hard_rejected = 0  # signals where penalty ≈ 0

        # History of computed penalties for debugging
        self._penalty_history: deque = deque(maxlen=5000)

        logger.info(
            f"[SPREAD_MODEL] Initialized mode={mode} "
            f"calibration_percentile={calibration_percentile}"
        )

    def observe_spread(self, symbol: str, spread: float):
        """Record a spread observation for calibration."""
        if spread <= 0:
            return
        self._spread_histories[symbol].append(spread)

        stats = self._symbol_stats[symbol]
        stats["samples"] += 1
        stats["min"] = min(stats["min"], spread)
        stats["max"] = max(stats["max"], spread)
        stats["sum"] += spread
        stats["sum_sq"] += spread * spread

        # Auto-calibrate threshold
        if len(self._spread_histories[symbol]) >= 100:
            arr = np.array(self._spread_histories[symbol])
            self._calibrated_thresholds[symbol] = float(
                np.percentile(arr, self._calibration_percentile)
            )

    def compute_penalty(self, symbol: str, spread: float,
                        custom_threshold: Optional[float] = None,
                        custom_steepness: Optional[float] = None) -> float:
        """
        Compute penalty for a given spread.

        Args:
            symbol: Trading symbol
            spread: Current spread value
            custom_threshold: Override auto-calibrated threshold
            custom_steepness: Override default steepness

        Returns:
            penalty: 0.0 to 1.0
        """
        self._total_evaluations += 1

        if spread <= 0:
            return 1.0  # No penalty for zero/negative spread

        threshold = custom_threshold or self._calibrated_thresholds.get(symbol)
        steepness = custom_steepness or self._default_steepness

        if threshold is None:
            # No calibration yet — use statistical outlier approach
            stats = self._symbol_stats.get(symbol)
            if stats and stats["samples"] >= 10:
                mean = stats["sum"] / stats["samples"]
                std = math.sqrt(stats["sum_sq"] / stats["samples"] - mean * mean)
                threshold = mean + 2.0 * std if std > 0 else spread * 2
            else:
                # Default generous threshold
                threshold = 50.0

        if self._mode == "sigmoid":
            penalty = sigmoid_penalty(spread - threshold, center=0.0, steepness=steepness)
        else:  # linear_decay
            hard_threshold = threshold * 2.0  # 2x threshold = hard reject
            penalty = linear_decay_penalty(spread, threshold, hard_threshold)

        # Track
        if penalty < 1.0:
            self._total_penalized += 1
        if penalty < 0.01:
            self._total_hard_rejected += 1

        self._penalty_history.append({
            "ts": time.time(),
            "symbol": symbol,
            "spread": spread,
            "threshold": threshold,
            "penalty": round(penalty, 4),
            "mode": self._mode,
        })

        return penalty

    def adjusted_score(self, raw_score: float, symbol: str, spread: float,
                       custom_threshold: Optional[float] = None,
                       custom_steepness: Optional[float] = None) -> Tuple[float, float]:
        """
        Compute adjusted score = raw_score * penalty.

        Returns:
            (adjusted_score, penalty)
        """
        penalty = self.compute_penalty(symbol, spread, custom_threshold, custom_steepness)
        return raw_score * penalty, penalty

    def should_reject(self, symbol: str, spread: float,
                      min_penalty_threshold: float = 0.05) -> bool:
        """
        Whether to hard-reject based on penalty.

        Default min_penalty_threshold = 0.05 means
        only reject if penalty < 5% (extreme spread).
        """
        penalty = self.compute_penalty(symbol, spread)
        return penalty < min_penalty_threshold

    def symbol_spread_stats(self, symbol: str) -> dict:
        """Get spread distribution stats for a symbol."""
        hist = list(self._spread_histories.get(symbol, []))
        if not hist:
            return {"samples": 0}

        arr = np.array(hist)
        return {
            "samples": len(arr),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "std": float(arr.std()),
            "p10": float(np.percentile(arr, 10)),
            "p25": float(np.percentile(arr, 25)),
            "p75": float(np.percentile(arr, 75)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "calibrated_threshold": self._calibrated_thresholds.get(symbol),
        }

    def summary(self) -> dict:
        """Return aggregate spread model summary."""
        all_spreads = []
        for symbol, hist in self._spread_histories.items():
            all_spreads.extend(hist)

        if all_spreads:
            arr = np.array(all_spreads)
            global_stats = {
                "min": float(arr.min()),
                "max": float(arr.max()),
                "mean": float(arr.mean()),
                "median": float(np.median(arr)),
                "std": float(arr.std()),
                "p95": float(np.percentile(arr, 95)),
                "p99": float(np.percentile(arr, 99)),
            }
        else:
            global_stats = {"min": 0, "max": 0, "mean": 0, "median": 0,
                            "std": 0, "p95": 0, "p99": 0}

        return {
            "mode": self._mode,
            "default_steepness": self._default_steepness,
            "calibration_percentile": self._calibration_percentile,
            "total_evaluations": self._total_evaluations,
            "total_penalized": self._total_penalized,
            "total_hard_rejected": self._total_hard_rejected,
            "penalty_rate": round(self._total_penalized / max(self._total_evaluations, 1) * 100, 2),
            "hard_rejection_rate": round(self._total_hard_rejected / max(self._total_evaluations, 1) * 100, 2),
            "symbols_calibrated": len(self._calibrated_thresholds),
            "global_spread_stats": global_stats,
            "symbols": {
                sym: self.symbol_spread_stats(sym)
                for sym in sorted(self._spread_histories.keys())
            },
        }


# Singleton for global use
_INSTANCE: Optional[SpreadModel] = None


def get_spread_model(mode: str = "sigmoid") -> SpreadModel:
    """Get or create the global SpreadModel instance."""
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = SpreadModel(mode=mode)
    return _INSTANCE
