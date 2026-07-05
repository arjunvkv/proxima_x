"""
system_health.py — System stability monitoring and health scoring.

Computes a single scalar representing: "Is the system stable, drifting,
or collapsing?" by aggregating entropy, stability, alignment, anomaly,
and regime-transition signals from the telemetry stream.

Score range: -1.0 (critical/collapsing) to +1.0 (healthy/stable).

Frame layout (432 bytes) — matches ``shared_memory_telemetry.py``::

    Offset   Size   Content
    0         64    Header  (<QdQQQQ2Q)
    64       184    Frame buffer 0  (<32f13f4x)
    248      184    Frame buffer 1  (<32f13f4x)

Scalar indices within the frame payload (13 floats, last is padding)::

    [0]  alignment
    [1]  stability
    [2]  entropy
    [3]  regime_state
    [4]  tpi_confidence
    [5]  shadow_alignment
    [6]  sof_score
    [7]  kill_switch_pressure
    [8]  rollout_progress
    [9]  execution_intensity
    [10] risk_exposure
    [11] system_integrity
    [12] (padding)
"""

from __future__ import annotations

import struct
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Deque, Dict, List

from intelligence.anomaly_detector import AnomalyEvent
from intelligence.regime_transition_detector import TransitionSignal

__all__ = [
    "HealthState",
    "SystemHealthScore",
    "SystemHealthMonitor",
]

# ---------------------------------------------------------------------------
# Constants — frame layout  (mirrors anomaly_detector.py)
# ---------------------------------------------------------------------------

_HEADER_FORMAT = struct.Struct("<QdQQQQ2Q")  # 64 bytes
_FRAME_FORMAT = struct.Struct("<32f13f4x")   # 184 bytes

HEADER_SIZE = 64
FRAME_SIZE = 184
BUF0_OFFSET = 64
BUF1_OFFSET = 248

# Minimum frames before scoring becomes meaningful
_MIN_FRAMES = 10

# Weights for the final composite score
_ENTROPY_WEIGHT = 0.20
_STABILITY_WEIGHT = 0.25
_ALIGNMENT_WEIGHT = 0.20
_ANOMALY_WEIGHT = 0.20
_REGIME_WEIGHT = 0.15

# Trend thresholds
_TREND_UP = 0.05
_TREND_DOWN = -0.05


# ---------------------------------------------------------------------------
# Enums & dataclasses
# ---------------------------------------------------------------------------


class HealthState(Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    CRITICAL = "CRITICAL"


@dataclass
class SystemHealthScore:
    """Aggregate health assessment for the system at a point in time.

    Attributes
    ----------
    score : float
        Composite health score in ``[-1.0, +1.0]`` (negative = bad).
    state : HealthState
        Categorical state derived from the composite score.
    components : dict[str, float]
        Individual sub-scores keyed by component name.
    trend : str
        One of ``"improving"``, ``"stable"``, ``"declining"``.
    timestamp : float
        Unix timestamp of the assessment.
    """
    score: float
    state: HealthState
    components: Dict[str, float]
    trend: str
    timestamp: float


# ---------------------------------------------------------------------------
# SystemHealthMonitor
# ---------------------------------------------------------------------------


class SystemHealthMonitor:
    """Aggregates telemetry signals into a single system-health score.

    The monitor maintains rolling windows of the key telemetry scalars
    extracted from raw 432-byte shared-memory frames, and also accepts
    external ``AnomalyEvent`` and ``TransitionSignal`` objects produced
    by the anomaly detector and regime-transition detector respectively.

    Call :meth:`compute` to obtain the current :class:`SystemHealthScore`.

    Parameters
    ----------
    window_size : int
        Maximum number of frames / events retained for rolling-window
        statistics.
    """

    def __init__(self, window_size: int = 100) -> None:
        self.window_size = window_size

        # ── Rolling scalar windows (from feed()) ──────────────────────────
        self._entropy: Deque[float] = deque(maxlen=window_size)
        self._stability: Deque[float] = deque(maxlen=window_size)
        self._alignment: Deque[float] = deque(maxlen=window_size)
        self._regime_state: Deque[float] = deque(maxlen=window_size)

        # ── 32D engine vector buffer (for alignment health) ───────────────
        self._engine_vectors: Deque[List[float]] = deque(maxlen=window_size)

        # ── Composite-score history (for trend) ──────────────────────────
        self._composite_history: Deque[float] = deque(maxlen=window_size)

        # ── External event buffers ───────────────────────────────────────
        self._anomaly_events: Deque[AnomalyEvent] = deque(maxlen=window_size)
        self._transition_signals: Deque[TransitionSignal] = deque(maxlen=window_size)

        # ── Internal state ───────────────────────────────────────────────
        self._latest_timestamp: float = 0.0
        self._frames_seen: int = 0

    # ── Public API ──────────────────────────────────────────────────────────

    def feed(self, frame: bytes) -> None:
        """Parse a raw 432-byte SHM frame and update rolling windows.

        Parameters
        ----------
        frame : bytes
            Exactly 432 bytes as written by ``TelemetryCore``.
        """
        if len(frame) < HEADER_SIZE + FRAME_SIZE:
            return  # malformed — silently ignore

        # -- Parse header ----------------------------------------------------
        hdr = _HEADER_FORMAT.unpack_from(frame, 0)
        active_idx: int = hdr[2]
        self._latest_timestamp = hdr[1]

        # -- Read the active frame buffer ------------------------------------
        offset = BUF0_OFFSET if active_idx == 0 else BUF1_OFFSET
        raw = _FRAME_FORMAT.unpack_from(frame, offset)

        # 32 floats — engine vector
        engine_vector = [float(v) for v in raw[:32]]

        # 13 floats — scalars (last is padding)
        alignment = float(raw[32])
        stability = float(raw[33])
        entropy_val = float(raw[34])
        regime_state = float(raw[35])

        # -- Update rolling windows ------------------------------------------
        self._entropy.append(entropy_val)
        self._stability.append(stability)
        self._alignment.append(alignment)
        self._regime_state.append(regime_state)
        self._engine_vectors.append(engine_vector)

        self._frames_seen += 1

    def feed_anomaly(self, anomaly: AnomalyEvent) -> None:
        """Feed an ``AnomalyEvent`` from the anomaly detector.

        Parameters
        ----------
        anomaly : AnomalyEvent
            The anomaly event produced by :class:`AnomalyDetector`.
        """
        self._anomaly_events.append(anomaly)

    def feed_transition(self, transition: TransitionSignal) -> None:
        """Feed a ``TransitionSignal`` from the regime-transition detector.

        Parameters
        ----------
        transition : TransitionSignal
            The transition signal produced by :class:`RegimeTransitionDetector`.
        """
        self._transition_signals.append(transition)

    def compute(self) -> SystemHealthScore:
        """Aggregate all signals into a single health score.

        Returns
        -------
        SystemHealthScore
            The current health assessment.
        """
        components = self.get_component_scores()

        # Weighted composite
        score = (
            _ENTROPY_WEIGHT * components.get("entropy_health", 0.0)
            + _STABILITY_WEIGHT * components.get("stability_health", 0.0)
            + _ALIGNMENT_WEIGHT * components.get("alignment_health", 0.0)
            + _ANOMALY_WEIGHT * components.get("anomaly_impact", 0.0)
            + _REGIME_WEIGHT * components.get("regime_stability", 0.0)
        )
        score = max(-1.0, min(1.0, score))

        # Record composite in history for trend calculation
        self._composite_history.append(score)

        # Trend
        trend = self._compute_trend()

        # State
        if score > 0.3:
            state = HealthState.HEALTHY
        elif score > -0.3:
            state = HealthState.DEGRADED
        else:
            state = HealthState.CRITICAL

        return SystemHealthScore(
            score=round(score, 6),
            state=state,
            components=components,
            trend=trend,
            timestamp=self._latest_timestamp or time.time(),
        )

    def get_component_scores(self) -> Dict[str, float]:
        """Return individual component scores for transparency.

        Each score is in ``[-1.0, +1.0]``.

        Returns
        -------
        dict[str, float]
            Keys: ``"entropy_health"``, ``"stability_health"``,
            ``"alignment_health"``, ``"anomaly_impact"``,
            ``"regime_stability"``.
        """
        return {
            "entropy_health": self._score_entropy_health(),
            "stability_health": self._score_stability_health(),
            "alignment_health": self._score_alignment_health(),
            "anomaly_impact": self._score_anomaly_impact(),
            "regime_stability": self._score_regime_stability(),
        }

    # ── Internal: individual scorers ────────────────────────────────────────

    def _score_entropy_health(self) -> float:
        """Score entropy health based on rolling mean of entropy.

        Returns
        -------
        float
            -1.0 to +1.0.
        """
        entropy_vals = list(self._entropy)
        if len(entropy_vals) < _MIN_FRAMES:
            return 0.0

        mean_e = sum(entropy_vals) / len(entropy_vals)

        # Healthy range [0.3, 0.7] → +0.5 to +1.0
        if 0.3 <= mean_e <= 0.7:
            # Map [0.3, 0.5] → [0.5, 1.0]  and  [0.5, 0.7] → [1.0, 0.5]
            if mean_e <= 0.5:
                return 0.5 + (mean_e - 0.3) / 0.2 * 0.5
            else:
                return 1.0 - (mean_e - 0.5) / 0.2 * 0.5

        # Too low (< 0.2) → -0.5 to -1.0
        if mean_e < 0.2:
            # Map [0.0, 0.2] → [-1.0, -0.5]
            return -1.0 + (mean_e / 0.2) * 0.5

        # Between [0.2, 0.3) → interpolate between -0.5 and +0.5
        if mean_e < 0.3:
            ratio = (mean_e - 0.2) / 0.1
            return -0.5 + ratio * 1.0

        # Too high (> 0.8) → -0.3 to -0.8
        if mean_e > 0.8:
            # Map [0.8, 1.0] → [-0.3, -0.8]
            capped = min(mean_e, 1.0)
            return -0.3 - (capped - 0.8) / 0.2 * 0.5

        # Between (0.7, 0.8] → interpolate between +0.5 and -0.3
        ratio = (mean_e - 0.7) / 0.1
        return 0.5 - ratio * 0.8

    def _score_stability_health(self) -> float:
        """Score stability health based on rolling mean of stability.

        Returns
        -------
        float
            -1.0 to +1.0.
        """
        stability_vals = list(self._stability)
        if len(stability_vals) < _MIN_FRAMES:
            return 0.0

        mean_s = sum(stability_vals) / len(stability_vals)

        # High stability (> 0.7) → +0.5 to +1.0
        if mean_s > 0.7:
            capped = min(mean_s, 1.0)
            return 0.5 + (capped - 0.7) / 0.3 * 0.5

        # Low stability (< 0.3) → -0.3 to -0.8
        if mean_s < 0.3:
            # Map [0.0, 0.3] → [-0.8, -0.3]
            return -0.8 + (mean_s / 0.3) * 0.5

        # Mid range [0.3, 0.7] → interpolate between -0.3 and +0.5
        ratio = (mean_s - 0.3) / 0.4
        return -0.3 + ratio * 0.8

    def _score_alignment_health(self) -> float:
        """Score alignment health based on rolling mean of alignment.

        Positive alignment → +0.3 to +1.0.
        Negative alignment → -0.3 to -1.0.

        Returns
        -------
        float
            -1.0 to +1.0.
        """
        align_vals = list(self._alignment)
        if len(align_vals) < _MIN_FRAMES:
            return 0.0

        mean_a = sum(align_vals) / len(align_vals)

        # Positive alignment
        if mean_a >= 0:
            # Map [0.0, 1.0] → [+0.3, +1.0]
            capped = min(mean_a, 1.0)
            return 0.3 + capped * 0.7

        # Negative alignment
        # Map [-1.0, 0.0] → [-1.0, -0.3]
        neg = max(mean_a, -1.0)
        return -0.3 + neg * 0.7  # neg is negative, so this moves toward -1.0

    def _score_anomaly_impact(self) -> float:
        """Score the impact of recent anomalies.

        No recent anomalies → +0.5.
        Each anomaly deducts based on severity, with decay over time.

        Returns
        -------
        float
            -1.0 to +1.0.
        """
        events = list(self._anomaly_events)
        if not events:
            return 0.5

        total_impact = 0.0

        for i, event in enumerate(events):
            # Determine age factor: older than half_window → half impact
            age_factor = 0.5 if i < len(events) / 2 else 1.0
            # In practice, use position in deque as proxy for recency

            if event.severity == "LOW":
                total_impact += 0.1 * age_factor
            elif event.severity == "MEDIUM":
                total_impact += 0.3 * age_factor
            elif event.severity == "HIGH":
                total_impact += 0.6 * age_factor
            elif event.severity == "CRITICAL":
                total_impact += 1.0 * age_factor

        # Start from baseline +0.5 and subtract accumulated impact
        score = 0.5 - total_impact
        return max(-1.0, min(1.0, score))

    def _score_regime_stability(self) -> float:
        """Score stability based on recent regime transitions.

        Returns
        -------
        float
            -1.0 to +1.0.
        """
        signals = list(self._transition_signals)
        if not signals:
            return 0.3

        # Count unique transitions in the window
        if len(signals) >= 3:
            # Multiple transitions → oscillating → penalty
            return -0.5

        # Single transition — score based on target regime
        last = signals[-1]
        if last.to_regime == "FULL":
            return 0.2
        elif last.to_regime == "MICRO":
            return 0.0
        elif last.to_regime == "SHADOW":
            return -0.2

        return 0.0

    def _compute_trend(self) -> str:
        """Determine the trend direction from the composite-score history.

        Returns
        -------
        str
            ``"improving"``, ``"stable"``, or ``"declining"``.
        """
        history = list(self._composite_history)
        if len(history) < 2:
            return "stable"

        slope = self._linear_slope(history)

        if slope > _TREND_UP:
            return "improving"
        elif slope < _TREND_DOWN:
            return "declining"
        return "stable"

    # ── Internal: math helpers ──────────────────────────────────────────────

    @staticmethod
    def _linear_slope(values: List[float]) -> float:
        """Least-squares linear slope over *values* at integer x-positions.

        Parameters
        ----------
        values : list[float]
            y-values at x = 0, 1, …, n-1.

        Returns
        -------
        float
            The slope coefficient (0.0 if fewer than 2 data points).
        """
        n = len(values)
        if n < 2:
            return 0.0

        x_mean = (n - 1) / 2.0
        y_mean = sum(values) / n

        num = 0.0
        den = 0.0
        for i, y in enumerate(values):
            dx = i - x_mean
            num += dx * (y - y_mean)
            den += dx * dx

        return num / den if den != 0.0 else 0.0
