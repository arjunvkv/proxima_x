"""
TPI Engine — Wave 2: Unified TPI Core Repair Layer.

Replaces fragmented TPI heuristics with a structured temporal signal tensor:
- Magnitude-weighted polarity (P0.15)
- Overlap-adjusted confirmation (P0.16)
- Temporal density weighting (P0.18)
- Neutral-collapse detection (P0.20)
- Self-dominance orthogonality (P0.21)
- Participant-weighted thresholds (T2)
- Curvature-to-inversion confidence (T4)
- Adaptive threshold (T6)
"""
import numpy as np
from typing import List, Optional


def tpi_magnitude_polarity(up_moves: List[float], down_moves: List[float]) -> float:
    """P0.15: Magnitude-weighted polarity — replaces count-based polarity.
    Weighs each directional move by its absolute magnitude.
    """
    up_mag = sum(abs(x) for x in up_moves)
    down_mag = sum(abs(x) for x in down_moves)
    total = up_mag + down_mag
    if total < 1e-8:
        return 0.0
    return (up_mag - down_mag) / total


def overlap_adjusted_confirmation(signals: List[float], overlap_window: int = 5) -> float:
    """P0.16: Overlap-adjusted confirmation — penalises clustered confirmation.
    Each signal is divided by (1 + number of prior signals within overlap_window).
    """
    adjusted = []
    for i, s in enumerate(signals):
        overlap_factor = 0.0
        for j in range(max(0, i - overlap_window), i):
            overlap_factor += 1.0
        adjusted.append(s / (1.0 + overlap_factor))
    return sum(adjusted) if adjusted else 0.0


def temporal_density_weighted_tpi(signals: List[float], timestamps: List[float]) -> float:
    """P0.18: Temporal density weighting — downweights high-frequency bursts.
    Weight = 1/dt where dt is time since previous signal.
    """
    if not signals:
        return 0.0
    weighted_sum = 0.0
    weight_total = 0.0
    for i in range(len(signals)):
        dt = max(1e-8, timestamps[i] - timestamps[i - 1] if i > 0 else 1.0)
        weight = dt
        weighted_sum += signals[i] * weight
        weight_total += weight
    return weighted_sum / (weight_total + 1e-8)


def detect_neutral_collapse(signal_series: List[float], threshold: float = 0.0005) -> bool:
    """P0.20: Detect neutral collapse — variance below threshold indicates stagnation."""
    if len(signal_series) < 2:
        return False
    variance = sum(x * x for x in signal_series) / (len(signal_series) + 1e-8)
    return variance < threshold


def tpi_self_dominance(polarity: float, confirmation: float, density: float) -> bool:
    """P0.21: Self-dominance orthogonality test — checks if any single axis dominates >92%."""
    v = np.array([polarity, confirmation, density])
    norm = np.linalg.norm(v) + 1e-8
    unit = v / norm
    dominance = float(max(abs(unit)))
    return dominance > 0.92


def participant_weighted_threshold(base_threshold: float, participant_weights: List[float]) -> float:
    """T2: Participant-weighted thresholds — adjusts threshold based on market participation."""
    if not participant_weights:
        return base_threshold
    weighted = sum(participant_weights) / (len(participant_weights) + 1e-8)
    return base_threshold * (0.5 + 0.5 * weighted)


def curvature_to_inversion_confidence(curvature_series: List[float]) -> float:
    """T4: Curvature-to-inversion confidence — uses second derivative acceleration."""
    if len(curvature_series) < 3:
        return 0.0
    second_derivatives = [
        curvature_series[i] - 2 * curvature_series[i - 1] + curvature_series[i - 2]
        for i in range(2, len(curvature_series))
    ]
    return max(0.0, min(1.0, sum(abs(x) for x in second_derivatives) / (len(second_derivatives) + 1e-8)))


class AdaptiveTPIThreshold:
    """T6: Adaptive TPI thresholds — tracks rolling mean+0.5*std as threshold."""

    def __init__(self, base: float = 0.5, max_history: int = 50):
        self.base = base
        self.max_history = max_history
        self.history: List[float] = []

    def update(self, tpi_value: float) -> float:
        self.history.append(tpi_value)
        if len(self.history) > self.max_history:
            self.history.pop(0)
        mean = sum(self.history) / len(self.history)
        std = (sum((x - mean) ** 2 for x in self.history) / len(self.history)) ** 0.5
        self.base = mean + 0.5 * std
        return self.base

    def current(self) -> float:
        return self.base


class TPIEngine:
    """Unified TPI Engine — orchestrates all Wave 2 corrections.

    Provides a single compute_full_tpi() call that returns a structured result
    with all TPI tensor dimensions, flags, and diagnostics.

    Usage:
        engine = TPIEngine()
        result = engine.compute_full_tpi(
            signals=[...], timestamps=[...],
            up_moves=[...], down_moves=[...],
            curvature_series=[...], participant_weights=[...]
        )
        # result["polarity"], result["confirmation"], result["threshold"], etc.
    """

    def __init__(self, base_threshold: float = 0.5, overlap_window: int = 5):
        self.adaptive_threshold = AdaptiveTPIThreshold(base=base_threshold)
        self.overlap_window = overlap_window
        self.neutral_collapse_flag = False
        self._history: List[float] = []

    def compute_full_tpi(
        self,
        signals: List[float],
        timestamps: List[float],
        up_moves: Optional[List[float]] = None,
        down_moves: Optional[List[float]] = None,
        curvature_series: Optional[List[float]] = None,
        participant_weights: Optional[List[float]] = None,
    ) -> dict:
        polarity = 0.0
        if up_moves is not None and down_moves is not None:
            polarity = tpi_magnitude_polarity(up_moves, down_moves)

        confirmation = overlap_adjusted_confirmation(signals, self.overlap_window)

        density = temporal_density_weighted_tpi(signals, timestamps)

        self.neutral_collapse_flag = detect_neutral_collapse(signals)

        is_dominated = tpi_self_dominance(polarity, confirmation, density)

        threshold = self.adaptive_threshold.update(polarity)

        if participant_weights:
            threshold = participant_weighted_threshold(threshold, participant_weights)

        inversion_confidence = 0.0
        if curvature_series is not None:
            inversion_confidence = curvature_to_inversion_confidence(curvature_series)

        self._history.append(polarity)
        if len(self._history) > 100:
            self._history.pop(0)

        return {
            "polarity": polarity,
            "confirmation": confirmation,
            "density": density,
            "threshold": threshold,
            "inversion_confidence": inversion_confidence,
            "neutral_collapse": self.neutral_collapse_flag,
            "is_dominated": is_dominated,
            "dominance_warning": "TPI_AXIS_DOMINANCE" if is_dominated else None,
        }
