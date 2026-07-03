"""
Microstructure Calibrator — Measure alignment between model assumptions and broker reality.

The problem:
  Model assumptions (spread, latency, tick frequency) may not match
  the live broker environment. This causes miscalibrated gates.

The fix:
  1. Compute real spread distribution from live MT5 data
  2. Measure latency histogram (signal → submission → confirmation)
  3. Analyze tick volatility clustering patterns
  4. Compute MAI (Microstructure Alignment Index) score

Output: Calibration parameters that can be fed back into the gate system.
"""
import os
import json
import math
import time
import logging
from collections import defaultdict, deque
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple, Callable

import numpy as np

logger = logging.getLogger("micro_calibrator")


class MicrostructureCalibrator:
    """
    Calibrates system parameters to live broker microstructure.

    Tracks and analyzes:
      - Spread distribution per symbol
      - Latency from signal generation to broker confirmation
      - Tick frequency and volatility clustering
      - Order rejection patterns

    Produces MAI (Microstructure Alignment Index) score.
    """

    def __init__(self,
                 max_samples: int = 10000,
                 calibration_window_hours: float = 24.0):
        """
        Args:
            max_samples: Max samples to keep per metric
            calibration_window_hours: Rolling calibration window
        """
        self._max_samples = max_samples
        self._window_seconds = calibration_window_hours * 3600.0

        # Spread samples: symbol -> [(timestamp, spread)]
        self._spread_samples: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=max_samples)
        )

        # Latency samples: [(timestamp, latency_ms)]
        self._latency_samples: deque = deque(maxlen=max_samples)

        # Tick interval samples: symbol -> [(timestamp, interval_ms)]
        self._tick_intervals: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=max_samples)
        )

        # Rejection samples: [(timestamp, reason)]
        self._rejection_samples: deque = deque(maxlen=max_samples)

        # Order success tracking
        self._order_submissions = 0
        self._order_accepts = 0
        self._order_rejects = 0
        self._reject_reasons: Dict[str, int] = defaultdict(int)

        # Computed calibration parameters
        self._calibration_params: Dict[str, dict] = {}

        # MAI score history
        self._mai_history: deque = deque(maxlen=100)

        # Persistence
        self._persist_path = os.path.join(
            os.getcwd(), "state", "micro_calibration.json"
        )
        self._load_persisted()

        logger.info(
            f"[MICRO_CALIB] Initialized window={calibration_window_hours}h "
            f"max_samples={max_samples}"
        )

    def _load_persisted(self):
        """Load persisted calibration data."""
        if not os.path.exists(self._persist_path):
            return
        try:
            with open(self._persist_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._calibration_params = data.get("calibration_params", {})
            self._order_submissions = data.get("order_submissions", 0)
            self._order_accepts = data.get("order_accepts", 0)
            self._order_rejects = data.get("order_rejects", 0)
            logger.info(f"[MICRO_CALIB] Loaded persisted calibration")
        except Exception as e:
            logger.warning(f"[MICRO_CALIB] Could not load persisted: {e}")

    def _persist(self):
        """Persist calibration data."""
        try:
            os.makedirs(os.path.dirname(self._persist_path), exist_ok=True)
            data = {
                "calibration_params": self._calibration_params,
                "order_submissions": self._order_submissions,
                "order_accepts": self._order_accepts,
                "order_rejects": self._order_rejects,
            }
            with open(self._persist_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception:
            pass

    # --- Observers ---

    def observe_spread(self, symbol: str, spread: float):
        """Record a spread observation."""
        if spread <= 0:
            return
        self._spread_samples[symbol].append((time.time(), spread))

    def observe_latency(self, latency_ms: float):
        """Record a latency observation (signal → broker confirmation)."""
        self._latency_samples.append((time.time(), latency_ms))

    def observe_tick_interval(self, symbol: str, interval_ms: float):
        """Record tick interval for a symbol."""
        if interval_ms <= 0:
            return
        self._tick_intervals[symbol].append((time.time(), interval_ms))

    def observe_order_result(self, accepted: bool, reason: str = ""):
        """Record order submission result."""
        self._order_submissions += 1
        if accepted:
            self._order_accepts += 1
        else:
            self._order_rejects += 1
            if reason:
                self._reject_reasons[reason] += 1

    def observe_rejection(self, reason: str):
        """Record a gate rejection for analysis."""
        self._rejection_samples.append((time.time(), reason))

    # --- Analysis ---

    def _trim_window(self, samples: deque) -> list:
        """Keep only samples within the calibration window."""
        if not samples:
            return []
        cutoff = time.time() - self._window_seconds
        # deque is ordered newest-last typically, but might be mixed
        # Filter in-place conceptually
        result = [(ts, val) for ts, val in samples if ts >= cutoff]
        return result

    def _compute_spread_calibration(self) -> Dict[str, dict]:
        """Compute per-symbol spread calibration parameters."""
        result = {}
        for symbol, samples in self._spread_samples.items():
            recent = self._trim_window(samples)
            if len(recent) < 10:
                continue

            spreads = np.array([s[1] for s in recent])

            result[symbol] = {
                "samples": len(spreads),
                "min": float(spreads.min()),
                "max": float(spreads.max()),
                "mean": float(spreads.mean()),
                "median": float(np.median(spreads)),
                "std": float(spreads.std()),
                "p10": float(np.percentile(spreads, 10)),
                "p25": float(np.percentile(spreads, 25)),
                "p75": float(np.percentile(spreads, 75)),
                "p90": float(np.percentile(spreads, 90)),
                "p95": float(np.percentile(spreads, 95)),
                "p99": float(np.percentile(spreads, 99)),
                "cv": float(spreads.std() / spreads.mean()) if spreads.mean() > 0 else 0,
                "recommended_threshold": float(np.percentile(spreads, 85)),
                "recommended_soft_threshold": float(np.percentile(spreads, 75)),
                "recommended_hard_threshold": float(np.percentile(spreads, 95)),
            }
        return result

    def _compute_latency_calibration(self) -> dict:
        """Compute latency distribution."""
        recent = self._trim_window(self._latency_samples)
        if len(recent) < 5:
            return {"samples": 0}

        latencies = np.array([s[1] for s in recent])
        return {
            "samples": len(latencies),
            "min_ms": float(latencies.min()),
            "max_ms": float(latencies.max()),
            "mean_ms": float(latencies.mean()),
            "median_ms": float(np.median(latencies)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "p99_ms": float(np.percentile(latencies, 99)),
            "std_ms": float(latencies.std()),
        }

    def _compute_tick_calibration(self) -> Dict[str, dict]:
        """Compute tick frequency calibration per symbol."""
        result = {}
        for symbol, samples in self._tick_intervals.items():
            recent = self._trim_window(samples)
            if len(recent) < 10:
                continue

            intervals = np.array([s[1] for s in recent])
            # Tick frequency in Hz
            mean_interval_ms = float(intervals.mean())
            frequency_hz = 1000.0 / mean_interval_ms if mean_interval_ms > 0 else 0

            result[symbol] = {
                "samples": len(intervals),
                "mean_interval_ms": mean_interval_ms,
                "median_interval_ms": float(np.median(intervals)),
                "frequency_hz": round(frequency_hz, 2),
                "p10_interval_ms": float(np.percentile(intervals, 10)),
                "p90_interval_ms": float(np.percentile(intervals, 90)),
                "cv": float(intervals.std() / intervals.mean()) if intervals.mean() > 0 else 0,
            }
        return result

    def _compute_order_quality(self) -> dict:
        """Compute order submission success metrics."""
        total = self._order_submissions
        if total == 0:
            return {"total": 0}

        accept_rate = self._order_accepts / total
        reject_rate = self._order_rejects / total

        return {
            "total_submissions": total,
            "accepts": self._order_accepts,
            "rejects": self._order_rejects,
            "accept_rate": round(accept_rate, 4),
            "reject_rate": round(reject_rate, 4),
            "reject_reasons": dict(self._reject_reasons),
        }

    # --- MAI Score ---

    def compute_mai(self) -> float:
        """
        Microstructure Alignment Index.

        Measures how well our model assumptions match broker reality.

        Components:
          1. Spread alignment: Do our spread thresholds match real distribution?
             (Higher = thresholds are reasonable)
          2. Latency alignment: Is latency within expected bounds?
          3. Order quality: Are orders being accepted at reasonable rate?
          4. Tick alignment: Is tick frequency as expected?

        Returns:
            MAI score: 0.0 (complete mismatch) to 1.0 (perfect alignment)
        """
        # 1. Spread alignment score
        spread_cal = self._compute_spread_calibration()
        if spread_cal:
            spread_scores = []
            for sym, cal in spread_cal.items():
                # If CV is low (< 1.0), spreads are stable -> good alignment
                cv = cal.get("cv", 0)
                spread_scores.append(max(0.0, min(1.0, 1.0 - cv / 2.0)))
            spread_alignment = np.mean(spread_scores) if spread_scores else 0.5
        else:
            spread_alignment = 0.5  # Neutral — no data yet

        # 2. Latency alignment score
        latency_cal = self._compute_latency_calibration()
        if latency_cal.get("samples", 0) >= 5:
            mean_latency = latency_cal.get("mean_ms", 100)
            # Latency < 1000ms = good, > 5000ms = bad
            latency_alignment = max(0.0, min(1.0, 1.0 - (mean_latency - 50) / 5000.0))
        else:
            latency_alignment = 0.5

        # 3. Order quality score
        order_quality = self._compute_order_quality()
        if order_quality.get("total", 0) >= 10:
            accept_rate = order_quality.get("accept_rate", 0)
            # Accept rate > 0.8 = good, < 0.3 = bad
            order_alignment = max(0.0, min(1.0, (accept_rate - 0.3) / 0.5))
        else:
            order_alignment = 0.5

        # 4. Tick alignment score
        tick_cal = self._compute_tick_calibration()
        if tick_cal:
            tick_scores = []
            for sym, cal in tick_cal.items():
                freq = cal.get("frequency_hz", 0)
                # Higher frequency = better for microstructure analysis
                # Score: 1.0 at >10Hz, 0.0 at <0.1Hz
                tick_scores.append(max(0.0, min(1.0, math.log10(freq + 0.1) / 2.0 + 0.5)))
            tick_alignment = np.mean(tick_scores) if tick_scores else 0.5
        else:
            tick_alignment = 0.5

        # Weighted MAI
        mai = (
            0.35 * spread_alignment +
            0.20 * latency_alignment +
            0.30 * order_alignment +
            0.15 * tick_alignment
        )

        mai = round(max(0.0, min(1.0, mai)), 4)
        self._mai_history.append((time.time(), mai))

        # Store calibration params
        self._calibration_params = {
            "mai": mai,
            "mai_components": {
                "spread_alignment": round(spread_alignment, 4),
                "latency_alignment": round(latency_alignment, 4),
                "order_alignment": round(order_alignment, 4),
                "tick_alignment": round(tick_alignment, 4),
            },
            "spread": spread_cal,
            "latency": latency_cal,
            "order_quality": order_quality,
            "tick_intervals": tick_cal,
            "last_updated": time.time(),
        }
        self._persist()

        logger.info(f"[MICRO_CALIB] MAI={mai} "
                     f"(spread={spread_alignment:.3f} lat={latency_alignment:.3f} "
                     f"order={order_alignment:.3f} tick={tick_alignment:.3f})")
        return mai

    def get_calibration_params(self) -> dict:
        """Get current calibration parameters."""
        # Auto-refresh if stale
        if not self._calibration_params or \
           time.time() - self._calibration_params.get("last_updated", 0) > 300:
            self.compute_mai()
        return self._calibration_params

    def get_spread_recommendations(self) -> Dict[str, dict]:
        """
        Get per-symbol spread threshold recommendations.
        Returns {symbol: {soft, hard, critical}} thresholds.
        """
        spread_cal = self._compute_spread_calibration()
        recommendations = {}
        for sym, cal in spread_cal.items():
            recommendations[sym] = {
                "soft_threshold": cal.get("recommended_soft_threshold", 30),
                "hard_threshold": cal.get("recommended_hard_threshold", 50),
                "critical_threshold": cal.get("p99", 100),
                "current_median": cal.get("median", 20),
                "confidence": min(1.0, cal.get("samples", 0) / 1000.0),
            }
        return recommendations

    def summary(self) -> dict:
        """Return full calibration summary."""
        mai = self.compute_mai()
        return {
            "mai": mai,
            "calibration_params": self._calibration_params,
            "sample_counts": {
                "spread": sum(len(s) for s in self._spread_samples.values()),
                "latency": len(self._latency_samples),
                "tick_intervals": sum(len(s) for s in self._tick_intervals.values()),
                "rejections": len(self._rejection_samples),
                "orders": self._order_submissions,
            },
            "spread_recommendations": self.get_spread_recommendations(),
            "rejection_profile": dict(self._reject_reasons),
        }


# Singleton
_INSTANCE: Optional[MicrostructureCalibrator] = None


def get_micro_calibrator() -> MicrostructureCalibrator:
    """Get or create the global MicrostructureCalibrator instance."""
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = MicrostructureCalibrator()
    return _INSTANCE
