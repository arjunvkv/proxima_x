"""
anomaly_detector.py — System-behaviour anomaly detection.

Detects "states that should not exist" in the trading system telemetry
stream by analysing the 32D engine vector and scalar metrics from the
432-byte shared-memory frame.

Detection strategies
--------------------
1. Entropy collapse spikes        — entropy drops below mean - 4σ
2. Contradictory engine outputs   — divergence between correlated engines
3. Alignment inversion events     — sign flips in consecutive frames
4. Shadow mirror divergence       — shadow metrics diverge from primary
5. Zero / NaN / Inf frame         — corrupted frames

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
from typing import Deque, Dict, List, Optional, Tuple

__all__ = [
    "AnomalyEvent",
    "AnomalyDetector",
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

# Minimum frames required before detection is attempted
_MIN_FRAMES = 20

# Pairs of engines known to be correlated under normal operation.
# (engine_idx_a, engine_idx_b, label)
_CORRELATED_ENGINE_PAIRS: List[Tuple[int, int, str]] = [
    (0, 7, "engine_0_7_corr"),
    (1, 8, "engine_1_8_corr"),
    (2, 6, "engine_2_6_corr"),
    (3, 11, "engine_3_11_corr"),
]


# ---------------------------------------------------------------------------
# AnomalyEvent dataclass
# ---------------------------------------------------------------------------


@dataclass
class AnomalyEvent:
    """Describes a single anomalous condition detected in the telemetry stream.

    Attributes
    ----------
    severity : str
        One of ``"LOW"``, ``"MEDIUM"``, ``"HIGH"``, ``"CRITICAL"``.
    subsystem : str
        The telemetry subsystem that exhibited the anomaly (e.g. ``"entropy"``,
        ``"engine_vector"``, ``"alignment"``, ``"stability"``, ``"tpi"``,
        ``"composite"``).
    timestamp : float
        Unix timestamp of the frame that triggered the event.
    vector_signature : list[float]
        The anomaly pattern (full 32D engine vector or a relevant subset).
    description : str
        Human-readable description of the anomaly.
    score : float
        Normalised anomaly severity score between 0.0 and 1.0.
    """
    severity: str
    subsystem: str
    timestamp: float
    vector_signature: List[float]
    description: str
    score: float


# ---------------------------------------------------------------------------
# AnomalyDetector
# ---------------------------------------------------------------------------


class AnomalyDetector:
    """Detects anomalous states in the trading-system telemetry stream.

    The detector maintains rolling windows of key telemetry scalars and engine-
    vector statistics extracted from raw 432-byte shared-memory frames fed via
    :meth:`feed`.  Each call to :meth:`detect` runs all five detection
    strategies against the current window and returns a list of
    :class:`AnomalyEvent` instances.

    Parameters
    ----------
    window_size : int
        Maximum number of frames retained for rolling-window baseline.
    zscore_threshold : float
        Z-score threshold above which a deviation is considered anomalous
        (used by the shadow-divergence and engine-correlation strategies).
    """

    def __init__(self, window_size: int = 200, zscore_threshold: float = 3.0) -> None:
        self.window_size = window_size
        self.zscore_threshold = zscore_threshold

        # ── Scalar rolling windows ──────────────────────────────────────────
        self._alignment: Deque[float] = deque(maxlen=window_size)
        self._stability: Deque[float] = deque(maxlen=window_size)
        self._entropy: Deque[float] = deque(maxlen=window_size)
        self._tpi: Deque[float] = deque(maxlen=window_size)
        self._shadow_alignment: Deque[float] = deque(maxlen=window_size)
        self._regime_state: Deque[float] = deque(maxlen=window_size)

        # ── Engine-vector rolling windows ──────────────────────────────────
        self._engine_vectors: Deque[List[float]] = deque(maxlen=window_size)

        # Rolling correlation buffers for each engine pair
        # (stores product of deviations for online correlation)
        self._pair_products: Dict[str, Deque[float]] = {
            label: deque(maxlen=window_size)
            for _, _, label in _CORRELATED_ENGINE_PAIRS
        }
        self._pair_a_mean: Dict[str, float] = {
            label: 0.0 for _, _, label in _CORRELATED_ENGINE_PAIRS
        }
        self._pair_b_mean: Dict[str, float] = {
            label: 0.0 for _, _, label in _CORRELATED_ENGINE_PAIRS
        }
        self._pair_count: Dict[str, int] = {
            label: 0 for _, _, label in _CORRELATED_ENGINE_PAIRS
        }

        # ── Alignment sign history for inversion detection ─────────────────
        self._prev_alignment_sign: Optional[int] = None

        # ── Cached state ────────────────────────────────────────────────────
        self._latest_timestamp: float = 0.0
        self._frames_seen: int = 0
        self._latest_engine_vector: List[float] = [0.0] * 32

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
        tpi_confidence = float(raw[36])
        shadow_alignment = float(raw[37])

        # -- Update rolling windows ------------------------------------------
        self._alignment.append(alignment)
        self._stability.append(stability)
        self._entropy.append(entropy_val)
        self._tpi.append(tpi_confidence)
        self._shadow_alignment.append(shadow_alignment)
        self._regime_state.append(regime_state)
        self._engine_vectors.append(engine_vector)
        self._latest_engine_vector = engine_vector

        # -- Update correlated-pair statistics -------------------------------
        for idx_a, idx_b, label in _CORRELATED_ENGINE_PAIRS:
            va = engine_vector[idx_a]
            vb = engine_vector[idx_b]
            n = self._pair_count[label]
            if n == 0:
                self._pair_a_mean[label] = va
                self._pair_b_mean[label] = vb
            else:
                # Welford online update
                old_a = self._pair_a_mean[label]
                old_b = self._pair_b_mean[label]
                self._pair_a_mean[label] = old_a + (va - old_a) / (n + 1)
                self._pair_b_mean[label] = old_b + (vb - old_b) / (n + 1)
                # Product of deviations (for covariance later)
                self._pair_products[label].append(
                    (va - self._pair_a_mean[label]) * (vb - self._pair_b_mean[label])
                )
            self._pair_count[label] += 1

        # -- Update alignment sign history -----------------------------------
        current_sign = 1 if alignment > 0 else (-1 if alignment < 0 else 0)
        self._prev_alignment_sign = current_sign

        self._frames_seen += 1

    def detect(self) -> List[AnomalyEvent]:
        """Run all five detection strategies against current telemetry window.

        Returns
        -------
        list[AnomalyEvent]
            Zero or more anomaly events, one per triggered detector.
        """
        if self._frames_seen < _MIN_FRAMES:
            return []

        events: List[AnomalyEvent] = []

        # 1. Entropy collapse spikes
        ev = self._check_entropy_collapse()
        if ev is not None:
            events.append(ev)

        # 2. Contradictory engine outputs
        ev = self._check_contradictory_engines()
        if ev is not None:
            events.append(ev)

        # 3. Alignment inversion events
        ev = self._check_alignment_inversion()
        if ev is not None:
            events.append(ev)

        # 4. Shadow mirror divergence
        ev = self._check_shadow_divergence()
        if ev is not None:
            events.append(ev)

        # 5. Zero / NaN / Inf frame
        ev = self._check_corrupted_frame()
        if ev is not None:
            events.append(ev)

        return events

    def get_baseline(self) -> Dict[str, Dict[str, float]]:
        """Return current baseline statistics for key metrics.

        Returns
        -------
        dict
            Keys are metric names, values are dicts with ``"mean"`` and
            ``"std"`` (both floats).  Returns zero-filled entries when
            insufficient data is available.
        """
        result: Dict[str, Dict[str, float]] = {}

        # Helper to compute mean + std from a deque
        def _stats(values: Deque[float]) -> Tuple[float, float]:
            n = len(values)
            if n < 2:
                return (0.0, 0.0)
            mean = sum(values) / n
            var = sum((v - mean) ** 2 for v in values) / (n - 1)
            std = math.sqrt(var) if var > 0 else 0.0
            return (mean, std)

        for name, buf in [
            ("entropy", self._entropy),
            ("stability", self._stability),
            ("alignment", self._alignment),
            ("tpi_confidence", self._tpi),
        ]:
            mean, std = _stats(buf)
            result[name] = {"mean": mean, "std": std}

        # Shadow divergence baseline
        div_values: List[float] = []
        align_list = list(self._alignment)
        shadow_list = list(self._shadow_alignment)
        n_div = min(len(align_list), len(shadow_list))
        for i in range(n_div):
            div_values.append(abs(align_list[i] - shadow_list[i]))
        if len(div_values) >= 2:
            mean_d = sum(div_values) / len(div_values)
            var_d = sum((d - mean_d) ** 2 for d in div_values) / (len(div_values) - 1)
            result["shadow_divergence"] = {
                "mean": mean_d,
                "std": math.sqrt(var_d) if var_d > 0 else 0.0,
            }
        else:
            result["shadow_divergence"] = {"mean": 0.0, "std": 0.0}

        # Engine correlation baseline (use the first pair as proxy)
        prod_values = list(self._pair_products.get("engine_0_7_corr", []))
        if len(prod_values) >= 2:
            mean_c = sum(prod_values) / len(prod_values)
            var_c = sum((p - mean_c) ** 2 for p in prod_values) / (len(prod_values) - 1)
            result["engine_0_7_corr"] = {
                "mean": mean_c,
                "std": math.sqrt(var_c) if var_c > 0 else 0.0,
            }
        else:
            result["engine_0_7_corr"] = {"mean": 0.0, "std": 0.0}

        return result

    # ── Internal: detector 1 — Entropy collapse spikes ──────────────────────

    def _check_entropy_collapse(self) -> Optional[AnomalyEvent]:
        """Detect entropy drops below mean - 4σ.

        A sudden collapse in entropy indicates a loss of variety in the
        engine-output distribution, often a precursor to mode collapse or
        system stall.

        The baseline (mean, std) is computed from the *older* portion of the
        window, excluding the most-recent quarter, so that a sustained drop
        does not contaminate the statistics.

        Returns
        -------
        AnomalyEvent or None
        """
        entropy_vals = list(self._entropy)
        n = len(entropy_vals)
        if n < _MIN_FRAMES:
            return None

        # Use the older portion for a clean baseline, excluding recent quarter
        recent_n = max(5, n // 4)
        older = entropy_vals[:-recent_n]
        if len(older) < 5:
            older = entropy_vals

        mean = sum(older) / len(older)
        var = sum((v - mean) ** 2 for v in older) / (len(older) - 1)
        std = math.sqrt(var) if var > 0 else 1e-8
        threshold = mean - 4.0 * std

        # Evaluate the *most recent* value against the old baseline
        latest = entropy_vals[-1]
        if latest < threshold:
            magnitude = (threshold - latest) / (abs(threshold) + 1e-8)
            score = min(magnitude, 1.0)
            return AnomalyEvent(
                severity="HIGH",
                subsystem="entropy",
                timestamp=self._latest_timestamp or time.time(),
                vector_signature=[latest],
                description=(
                    f"Entropy collapse: {latest:.4f} below threshold "
                    f"{threshold:.4f} (mean={mean:.4f}, σ={std:.4f})"
                ),
                score=score,
            )
        return None

    # ── Internal: detector 2 — Contradictory engine outputs ────────────────

    def _check_contradictory_engines(self) -> Optional[AnomalyEvent]:
        """Detect divergence between normally correlated engine pairs.

        When a pair of engines that normally correlate show extreme opposite
        values, it suggests contradictory internal signals.

        Returns
        -------
        AnomalyEvent or None
        """
        if len(self._engine_vectors) < _MIN_FRAMES:
            return None

        for idx_a, idx_b, label in _CORRELATED_ENGINE_PAIRS:
            if self._pair_count[label] < _MIN_FRAMES:
                continue

            # Compute rolling correlation coefficient
            prod_vals = list(self._pair_products[label])
            if len(prod_vals) < 5:
                continue

            # Get recent pair values
            recent_vectors = list(self._engine_vectors)
            recent_a = [v[idx_a] for v in recent_vectors[-10:]]
            recent_b = [v[idx_b] for v in recent_vectors[-10:]]

            # Check for extreme divergence: large opposite values
            # (one strongly positive, the other strongly negative)
            last_a = recent_a[-1]
            last_b = recent_b[-1]

            # Detect contradiction: both have absolute value above threshold
            # and their signs differ, and the magnitude of difference is large
            abs_a = abs(last_a)
            abs_b = abs(last_b)

            # Compute baseline average absolute values for comparison
            all_a = [v[idx_a] for v in recent_vectors]
            all_b = [v[idx_b] for v in recent_vectors]
            baseline_abs_a = sum(abs(v) for v in all_a) / len(all_a)
            baseline_abs_b = sum(abs(v) for v in all_b) / len(all_b)
            baseline_abs = (baseline_abs_a + baseline_abs_b) / 2.0

            # Contradiction threshold: both values are significant (> 2x baseline)
            # and they have opposite signs
            if (abs_a > 1.5 * baseline_abs + 0.5
                    and abs_b > 1.5 * baseline_abs + 0.5
                    and (last_a * last_b) < 0):
                magnitude = (abs_a + abs_b) / (2.0 * (abs(baseline_abs) + 1e-8))
                score = min(magnitude * 0.5, 1.0)
                return AnomalyEvent(
                    severity="MEDIUM",
                    subsystem="engine_vector",
                    timestamp=self._latest_timestamp or time.time(),
                    vector_signature=[last_a, last_b],
                    description=(
                        f"Contradictory engine outputs: "
                        f"engine[{idx_a}]={last_a:.4f} vs "
                        f"engine[{idx_b}]={last_b:.4f} (normally correlated)"
                    ),
                    score=score,
                )

        return None

    # ── Internal: detector 3 — Alignment inversion events ──────────────────

    def _check_alignment_inversion(self) -> Optional[AnomalyEvent]:
        """Detect alignment sign flips between consecutive frames.

        An alignment inversion (positive → negative or negative → positive)
        in consecutive frames indicates a sudden reversal in the system's
        directional consensus.

        Returns
        -------
        AnomalyEvent or None
        """
        align_vals = list(self._alignment)
        if len(align_vals) < 2:
            return None

        # Look at the last two pairs for inversion
        for i in range(len(align_vals) - 1, 0, -1):
            prev = align_vals[i - 1]
            curr = align_vals[i]

            # Skip near-zero values (noise floor)
            if abs(prev) < 1e-6 or abs(curr) < 1e-6:
                continue

            prev_sign = 1 if prev > 0 else -1
            curr_sign = 1 if curr > 0 else -1

            if prev_sign != curr_sign:
                score = min(abs(curr - prev) / (max(abs(prev), abs(curr), 1e-8)), 1.0)
                return AnomalyEvent(
                    severity="HIGH",
                    subsystem="alignment",
                    timestamp=self._latest_timestamp or time.time(),
                    vector_signature=[prev, curr],
                    description=(
                        f"Alignment inversion: {prev:.4f} → {curr:.4f} "
                        f"(sign flip in consecutive frames)"
                    ),
                    score=score,
                )

        return None

    # ── Internal: detector 4 — Shadow mirror divergence ────────────────────

    def _check_shadow_divergence(self) -> Optional[AnomalyEvent]:
        """Detect when shadow-state metrics deviate from primary metrics.

        The shadow mirror should closely track the primary system state.
        Significant divergence indicates a split between the primary and
        shadow execution paths, which is a CRITICAL concern.

        The baseline divergence (mean, std) is computed from the *older*
        portion of the window, excluding the most-recent quarter, so that
        a sustained divergence does not contaminate the statistics.

        Returns
        -------
        AnomalyEvent or None
        """
        align_vals = list(self._alignment)
        shadow_vals = list(self._shadow_alignment)
        n = min(len(align_vals), len(shadow_vals))
        if n < _MIN_FRAMES:
            return None

        # Use the older portion for a clean baseline, excluding recent quarter
        recent_n = max(5, n // 4)
        older_align = align_vals[:-recent_n]
        older_shadow = shadow_vals[:-recent_n]
        if len(older_align) < 5:
            older_align = align_vals
            older_shadow = shadow_vals

        divergences = [abs(older_align[i] - older_shadow[i])
                       for i in range(len(older_align))]
        mean_div = sum(divergences) / len(divergences)
        var_div = sum((d - mean_div) ** 2 for d in divergences) / (len(divergences) - 1)
        std_div = math.sqrt(var_div) if var_div > 0 else 1e-8

        threshold = mean_div + self.zscore_threshold * std_div

        latest_align = align_vals[-1]
        latest_shadow = shadow_vals[-1]
        latest_div = abs(latest_align - latest_shadow)

        if latest_div > threshold:
            magnitude = (latest_div - threshold) / (threshold + 1e-8)
            score = min(magnitude, 1.0)
            return AnomalyEvent(
                severity="CRITICAL",
                subsystem="composite",
                timestamp=self._latest_timestamp or time.time(),
                vector_signature=[latest_align, latest_shadow],
                description=(
                    f"Shadow mirror divergence: "
                    f"alignment={latest_align:.4f} vs "
                    f"shadow={latest_shadow:.4f} "
                    f"(divergence={latest_div:.4f}, threshold={threshold:.4f})"
                ),
                score=score,
            )
        return None

    # ── Internal: detector 5 — Zero / NaN / Inf frame ──────────────────────

    def _check_corrupted_frame(self) -> Optional[AnomalyEvent]:
        """Detect frames with corrupted engine vectors.

        Checks each element of the latest engine vector for zeros (all),
        NaN, or infinity.  A corrupted frame suggests memory corruption,
        buffer overrun, or a catastrophic numerical error.

        Returns
        -------
        AnomalyEvent or None
        """
        vector = self._latest_engine_vector
        if not vector:
            return None

        # Check for NaN / Inf
        has_nan = any(math.isnan(v) for v in vector)
        has_inf = any(math.isinf(v) for v in vector)

        if has_nan or has_inf:
            problem = "NaN" if has_nan else "Infinity"
            return AnomalyEvent(
                severity="CRITICAL",
                subsystem="engine_vector",
                timestamp=self._latest_timestamp or time.time(),
                vector_signature=vector,
                description=f"Corrupted frame: {problem} detected in engine vector",
                score=1.0,
            )

        # Check for all-zero vector
        if len(vector) > 0 and all(v == 0.0 for v in vector):
            return AnomalyEvent(
                severity="CRITICAL",
                subsystem="engine_vector",
                timestamp=self._latest_timestamp or time.time(),
                vector_signature=vector,
                description="Corrupted frame: all-zero engine vector detected",
                score=1.0,
            )

        return None
