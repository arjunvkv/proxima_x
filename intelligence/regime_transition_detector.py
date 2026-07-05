"""
regime_transition_detector.py — Pre-transition pressure detection.

Detects precursors to SHADOW → MICRO → FULL regime changes by analysing
entropy compression acceleration, stability divergence, and TPI curvature
spikes from the 432-byte shared-memory telemetry frame.

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

import math
import struct
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

__all__ = [
    "TransitionSignal",
    "RegimeTransitionDetector",
]

# ---------------------------------------------------------------------------
# Constants — frame layout
# ---------------------------------------------------------------------------

_HEADER_FORMAT = struct.Struct("<QdQQQQ2Q")  # 64 bytes
_FRAME_FORMAT = struct.Struct("<32f13f4x")   # 184 bytes

HEADER_SIZE = 64
FRAME_SIZE = 184
BUF0_OFFSET = 64
BUF1_OFFSET = 248

# Regime progression order
REGIME_ORDER: List[str] = ["SHADOW", "MICRO", "FULL"]

# Per-driver weight for the combined probability softmax.
# Entropy compression is the most predictive precursor, followed by
# stability divergence and TPI curvature.
DRIVER_WEIGHTS: dict = {
    "entropy_compression": 0.50,
    "stability_divergence": 0.30,
    "tpi_curvature": 0.20,
}

# Minimum frames required before detection is attempted
_MIN_FRAMES = 20

# Sigmoid parameters for probability mapping
_SIGMOID_STEEPNESS = 4.0
_SIGMOID_MIDPOINT = 0.25


# ---------------------------------------------------------------------------
# TransitionSignal dataclass
# ---------------------------------------------------------------------------


@dataclass
class TransitionSignal:
    """Signals pre-transition pressure from the current regime to a target.

    Attributes
    ----------
    from_regime : str
        The current regime ("SHADOW", "MICRO", "FULL", or "UNKNOWN").
    to_regime : str
        The predicted next regime in the progression.
    probability : float
        Combined probability of the transition, 0.0 – 1.0.
    confidence : float
        Confidence in the signal based on driver count and magnitude, 0.0 – 1.0.
    drivers : list[str]
        Names of the active detection drivers that triggered this signal.
    timestamp : float
        Unix timestamp of the latest frame that contributed to the signal.
    """
    from_regime: str
    to_regime: str
    probability: float
    confidence: float
    drivers: List[str]
    timestamp: float


# ---------------------------------------------------------------------------
# RegimeTransitionDetector
# ---------------------------------------------------------------------------


class RegimeTransitionDetector:
    """Detects pre-transition pressure before SHADOW → MICRO → FULL changes.

    The detector maintains rolling windows of key telemetry scalars extracted
    from raw 432-byte shared-memory frames fed via :meth:`feed`.  It exposes
    three independent driver detectors:

    1. **Entropy compression acceleration** — flags when the entropy slope
       drops faster than the historical mean minus two standard deviations.
    2. **Stability divergence** — flags when the absolute difference between
       stability and alignment exceeds the historical mean by more than two
       standard deviations.
    3. **TPI curvature spikes** — flags when the second derivative of the TPI
       confidence exceeds two standard deviations of the historical baseline.

    When one or more drivers are active, a weighted softmax probability and
    confidence score are computed and returned as a ``TransitionSignal``.

    Parameters
    ----------
    window_size : int
        Maximum number of frames retained for rolling-window analysis.
        Larger windows provide smoother baselines but slower adaptation.
    """

    def __init__(self, window_size: int = 100) -> None:
        self.window_size = window_size

        # Rolling windows — deque automatically drops oldest when full
        self._entropy: deque = deque(maxlen=window_size)
        self._stability: deque = deque(maxlen=window_size)
        self._alignment: deque = deque(maxlen=window_size)
        self._tpi: deque = deque(maxlen=window_size)
        self._engine_vectors: deque = deque(maxlen=window_size)

        # Cached state
        self._current_regime: str = "UNKNOWN"
        self._latest_timestamp: float = 0.0
        self._frames_seen: int = 0

    # ── Public API ──────────────────────────────────────────────────────────

    def feed(self, frame: bytes) -> None:
        """Parse a raw 432-byte SHM frame and update internal rolling windows.

        Parameters
        ----------
        frame : bytes
            Exactly 432 bytes as written by ``TelemetryCore`` (see
            ``shared_memory_telemetry.py``).
        """
        if len(frame) < HEADER_SIZE + FRAME_SIZE:
            return  # malformed — silently ignore

        # -- Parse header ----------------------------------------------------
        hdr = _HEADER_FORMAT.unpack_from(frame, 0)
        active_idx: int = hdr[2]            # 0 or 1
        self._latest_timestamp = hdr[1]

        # -- Read the active frame buffer ------------------------------------
        offset = BUF0_OFFSET if active_idx == 0 else BUF1_OFFSET
        raw = _FRAME_FORMAT.unpack_from(frame, offset)

        # 32 floats — engine vector
        engine_vector = list(raw[:32])

        # 13 floats — scalars (last is padding)
        alignment = raw[32]
        stability = raw[33]
        entropy = raw[34]
        regime_state = raw[35]
        tpi_confidence = raw[36]

        # -- Update rolling windows ------------------------------------------
        self._entropy.append(entropy)
        self._stability.append(stability)
        self._alignment.append(alignment)
        self._tpi.append(tpi_confidence)
        self._engine_vectors.append(engine_vector)

        # -- Update cached regime --------------------------------------------
        self._current_regime = self._decode_regime(regime_state)
        self._frames_seen += 1

    def detect(self) -> Optional[TransitionSignal]:
        """Analyse recent history and return a signal if pressure is detected.

        Returns
        -------
        TransitionSignal or None
            ``None`` when insufficient data are available or when no driver
            currently exceeds its threshold.
        """
        if self._frames_seen < _MIN_FRAMES:
            return None

        drivers: List[str] = []
        pressures: List[float] = []

        # Check each driver
        ec = self._check_entropy_compression()
        if ec is not None:
            drivers.append("entropy_compression")
            pressures.append(ec)

        sd = self._check_stability_divergence()
        if sd is not None:
            drivers.append("stability_divergence")
            pressures.append(sd)

        tc = self._check_tpi_curvature()
        if tc is not None:
            drivers.append("tpi_curvature")
            pressures.append(tc)

        if not drivers:
            return None

        probability = self._combined_probability(drivers, pressures)
        confidence = self._compute_confidence(drivers, pressures)
        from_regime, to_regime = self._infer_transition()

        return TransitionSignal(
            from_regime=from_regime,
            to_regime=to_regime,
            probability=probability,
            confidence=confidence,
            drivers=drivers,
            timestamp=self._latest_timestamp or time.time(),
        )

    def get_current_regime(self) -> str:
        """Return the parsed regime from the latest processed frame.

        Returns
        -------
        str
            One of ``"SHADOW"``, ``"MICRO"``, ``"FULL"``, or ``"UNKNOWN"``.
        """
        return self._current_regime

    # ── Internal: regime decoding ───────────────────────────────────────────

    @staticmethod
    def _decode_regime(regime_state: float) -> str:
        """Map a float ``regime_state`` scalar to a regime name.

        The convention (from the telemetry pipeline) is:
            - 0.0 → SHADOW
            - 1.0 → MICRO
            - 2.0 → FULL
        Any other value is mapped to ``"UNKNOWN"``.
        """
        idx = round(regime_state)
        if 0 <= idx < len(REGIME_ORDER):
            return REGIME_ORDER[idx]
        return "UNKNOWN"

    # ── Internal: driver detectors ──────────────────────────────────────────

    def _check_entropy_compression(self) -> Optional[float]:
        """Detect entropy compression acceleration.

        Computes a least-squares linear slope over the trailing quarter of the
        entropy window.  If the slope is more than two standard deviations
        *below* the rolling-mean slope (i.e. entropy is dropping abnormally
        fast), returns a normalised pressure magnitude in [0, 1].

        Returns
        -------
        float or None
            Pressure magnitude or ``None`` if the driver is inactive.
        """
        entropy = list(self._entropy)
        n = len(entropy)
        if n < 10:
            return None

        recent_n = max(5, n // 4)
        recent = entropy[-recent_n:]
        recent_slope = self._linear_slope(recent)

        # Baseline from the older portion of the window
        older = entropy[:-recent_n]
        if len(older) < 3:
            return None

        older_deltas = [older[i] - older[i - 1] for i in range(1, len(older))]
        if not older_deltas:
            return None

        mean_delta = sum(older_deltas) / len(older_deltas)
        var_delta = sum((d - mean_delta) ** 2 for d in older_deltas) / len(older_deltas)
        std_delta = math.sqrt(var_delta) if var_delta > 0 else 1e-8

        # Threshold: two sigma below the mean of historical deltas
        threshold = mean_delta - 2.0 * std_delta

        # Compression means entropy dropping faster than normal → more negative slope
        if recent_slope < threshold:
            magnitude = (threshold - recent_slope) / (abs(mean_delta) + 1e-8)
            return min(magnitude, 1.0)

        return None

    def _check_stability_divergence(self) -> Optional[float]:
        """Detect stability–alignment divergence.

        Compares the mean of stability and alignment over the trailing quarter
        of the window.  When their absolute difference exceeds the historical
        mean divergence by more than two standard deviations, returns a
        normalised pressure magnitude.

        Returns
        -------
        float or None
            Pressure magnitude or ``None`` if the driver is inactive.
        """
        stability = list(self._stability)
        alignment = list(self._alignment)
        n = min(len(stability), len(alignment))
        if n < 10:
            return None

        recent_n = max(5, n // 4)
        recent_stab = stability[-recent_n:]
        recent_align = alignment[-recent_n:]
        mean_stab = sum(recent_stab) / len(recent_stab)
        mean_align = sum(recent_align) / len(recent_align)
        divergence = abs(mean_stab - mean_align)

        # Baseline from older period
        older_stab = stability[:-recent_n]
        older_align = alignment[:-recent_n]
        if len(older_stab) < 3:
            return None

        older_divs = [abs(older_stab[i] - older_align[i]) for i in range(len(older_stab))]
        mean_old = sum(older_divs) / len(older_divs)
        var_old = sum((d - mean_old) ** 2 for d in older_divs) / len(older_divs)
        std_old = math.sqrt(var_old) if var_old > 0 else 1e-8

        threshold = mean_old + 2.0 * std_old
        if divergence > threshold:
            magnitude = (divergence - threshold) / (threshold + 1e-8)
            return min(magnitude, 1.0)

        return None

    def _check_tpi_curvature(self) -> Optional[float]:
        """Detect TPI curvature spikes via second derivative.

        Computes the mean second derivative of the TPI confidence over the
        trailing quarter of the window.  If it exceeds two standard deviations
        of the historical curvature baseline, returns a normalised pressure
        magnitude.

        Returns
        -------
        float or None
            Pressure magnitude or ``None`` if the driver is inactive.
        """
        tpi = list(self._tpi)
        n = len(tpi)
        if n < 10:
            return None

        # First derivative (rate of change)
        d1 = [tpi[i] - tpi[i - 1] for i in range(1, n)]
        # Second derivative (curvature / acceleration)
        d2 = [d1[i] - d1[i - 1] for i in range(1, len(d1))]

        if not d2:
            return None

        recent_n = max(3, len(d2) // 4)
        recent_d2 = d2[-recent_n:]
        mean_curv = sum(recent_d2) / len(recent_d2)

        # Baseline from older curvature data
        older_d2 = d2[:-recent_n] if len(d2) > recent_n else d2
        if len(older_d2) < 3:
            return None

        mean_old = sum(older_d2) / len(older_d2)
        var_old = sum((c - mean_old) ** 2 for c in older_d2) / len(older_d2)
        std_old = math.sqrt(var_old) if var_old > 0 else 1e-8

        threshold = 2.0 * std_old
        if abs(mean_curv) > threshold:
            magnitude = (abs(mean_curv) - threshold) / (threshold + 1e-8)
            return min(magnitude, 1.0)

        return None

    # ── Internal: scoring helpers ───────────────────────────────────────────

    def _combined_probability(self, drivers: List[str],
                              pressures: List[float]) -> float:
        """Combine active driver pressures into a single probability.

        Uses a weighted softmax approach: each driver contributes its pressure
        weighted by the predefined ``DRIVER_WEIGHTS``.  The weighted sum is
        passed through a sigmoid to produce a 0–1 probability.

        Parameters
        ----------
        drivers : list[str]
            Active driver names.
        pressures : list[float]
            Corresponding pressure magnitudes (same order as *drivers*).

        Returns
        -------
        float
            Combined transition probability in [0, 1].
        """
        if not drivers:
            return 0.0

        weights = [DRIVER_WEIGHTS.get(d, 0.2) for d in drivers]
        total_w = sum(weights)
        if total_w == 0:
            return 0.0

        # Normalise weights to sum to 1.0
        norm_w = [w / total_w for w in weights]
        weighted_sum = sum(p * w for p, w in zip(pressures, norm_w))

        # Sigmoid mapping: steepness controls how sharply probability rises
        # around the midpoint.
        prob = 1.0 / (1.0 + math.exp(-_SIGMOID_STEEPNESS * (weighted_sum - _SIGMOID_MIDPOINT)))
        return prob

    def _compute_confidence(self, drivers: List[str],
                            pressures: List[float]) -> float:
        """Compute confidence from the number of active drivers and their strength.

        Confidence is the equally-weighted combination of:
            - The ratio of active drivers to the maximum possible (3).
            - The average pressure magnitude across all active drivers.

        Parameters
        ----------
        drivers : list[str]
            Active driver names.
        pressures : list[float]
            Corresponding pressure magnitudes.

        Returns
        -------
        float
            Confidence score in [0, 1].
        """
        if not drivers:
            return 0.0

        driver_ratio = len(drivers) / 3.0
        avg_pressure = sum(pressures) / len(pressures)
        confidence = 0.5 * driver_ratio + 0.5 * avg_pressure
        return min(max(confidence, 0.0), 1.0)

    def _infer_transition(self) -> Tuple[str, str]:
        """Infer the transition direction based on the current regime.

        Follows the canonical progression: SHADOW → MICRO → FULL.
        If the current regime is unknown, defaults to (SHADOW, MICRO).

        Returns
        -------
        tuple[str, str]
            ``(from_regime, to_regime)``.
        """
        current = self._current_regime
        if current in REGIME_ORDER:
            idx = REGIME_ORDER.index(current)
            if idx < len(REGIME_ORDER) - 1:
                return current, REGIME_ORDER[idx + 1]
            # Already at FULL — return (FULL, FULL) since progression stalls
            return current, current
        return "SHADOW", "MICRO"

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
