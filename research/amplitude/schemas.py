"""
Phase V — Structural Amplitude Discovery
Immutable data contracts for amplitude state, forward observations, surface
entries, and surface-level statistics.
"""

import hashlib
import sys
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

sys.path.insert(0, ".")
sys.path.insert(0, "proxima_x")


# ---------------------------------------------------------------------------
# Core amplitude state — a single tick-level snapshot of the microstructure.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AmplitudeState:
    """State features extracted from a tick-level window."""

    symbol: str
    ts: float
    oss_bucket: int  # 0–9
    tross_delta: int  # -2 to +2
    sal_score: float  # 0–1
    entropy_slope: float  # rate of change of entropy
    compression_density: float  # 0–1
    tick_velocity: float  # ticks per second
    spread: float  # in bps
    regime_id: str


# ---------------------------------------------------------------------------
# Forward-looking amplitude — what actually happened over a horizon.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ForwardAmplitude:
    """Observed forward price movement over a fixed horizon."""

    horizon_sec: int
    abs_move: float  # absolute price move in bps
    signed_move: float  # signed price move in bps
    max_excursion: float  # maximum favourable excursion in bps
    min_excursion: float  # maximum adverse excursion in bps


# ---------------------------------------------------------------------------
# Observation — pairing a state with its realised forward amplitude.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AmplitudeObservation:
    """A single labelled example: state + future."""

    state: AmplitudeState
    future: ForwardAmplitude


# ---------------------------------------------------------------------------
# Surface entry — aggregated statistics for a (state-hash, horizon) bucket.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AmplitudeSurfaceEntry:
    """Statistics aggregated over many observations sharing the same
    (state-hash, horizon) key."""

    state_hash: str  # composite key produced by StateHasher
    horizon: int
    n: int
    mean_abs: float
    median_abs: float
    std_abs: float
    p90_abs: float
    spread_multiple: float  # mean_abs / spread
    aer: float  # conditional std / unconditional std
    exceed_prob: Dict[float, float]  # k → probability for k in {1.0, 1.5, 2.0, 3.0, 4.0}


# ---------------------------------------------------------------------------
# Surface-level roll-up statistics.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SurfaceStats:
    """High-level view of the amplitude surface."""

    total_observations: int
    n_buckets: int
    n_horizons: int
    symbols: List[str]
    coverage_pct: float


# ---------------------------------------------------------------------------
# State hasher — deterministic composite key for surface bucketing.
# ---------------------------------------------------------------------------


class StateHasher:
    """Deterministic composite hash from *oss_bucket*, *regime_id* and
    coarsely-quantised numeric fields."""

    # Quantisation bins for continuous fields.
    _SAL_BINS: int = 10  # 0.0 – 1.0  → 10 bins
    _ENTROPY_BINS: int = 10  # clamped to ±some reasonable range
    _COMPRESSION_BINS: int = 10  # 0.0 – 1.0 → 10 bins
    _VELOCITY_BINS: int = 10  # log-like bins
    _SPREAD_BINS: int = 10  # bps bins

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def hash(cls, state: AmplitudeState) -> str:
        """Return a deterministic string key for an ``AmplitudeState``.

        The key is built from *oss_bucket*, *regime_id*, and compressed
        (quantised) versions of the remaining feature fields.  This
        guarantees that near-identical states map to the same surface
        bucket while still providing useful discrimination.
        """
        parts = [
            str(state.oss_bucket),
            state.regime_id,
            str(cls._quantise(state.sal_score, bins=cls._SAL_BINS, lo=0.0, hi=1.0)),
            str(cls._quantise_entropy(state.entropy_slope)),
            str(cls._quantise(state.compression_density, bins=cls._COMPRESSION_BINS, lo=0.0, hi=1.0)),
            str(cls._quantise_velocity(state.tick_velocity)),
            str(cls._quantise(state.spread, bins=cls._SPREAD_BINS, lo=0.0, hi=200.0)),
        ]
        raw = "|".join(parts)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    # ------------------------------------------------------------------
    # Internal quantisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _quantise(value: float, bins: int, lo: float, hi: float) -> int:
        """Bin *value* into ``bins`` equal-width buckets over *[lo, hi)*."""
        if hi <= lo:
            return 0
        clamped = max(lo, min(value, hi - 1e-12))
        return int((clamped - lo) / (hi - lo) * bins)

    @classmethod
    def _quantise_entropy(cls, value: float) -> int:
        """Bin entropy slope into a small integer, clamping to ±5."""
        clamped = max(-5.0, min(value, 5.0 - 1e-12))
        normalised = (clamped + 5.0) / 10.0  # map [-5, 5) → [0, 1)
        return int(normalised * cls._ENTROPY_BINS)

    @classmethod
    def _quantise_velocity(cls, value: float) -> int:
        """Bin tick velocity using a log-like scale.

        The mapping compresses the long tail of high velocities into 10
        bins, with bin 0 reserved for zero velocity.
        """
        if value <= 0.0:
            return 0
        # Piecewise log-like bins compressing the long tail.
        if value < 0.5:
            return 1
        if value < 1.0:
            return 2
        if value < 2.0:
            return 3
        if value < 5.0:
            return 4
        if value < 10.0:
            return 5
        if value < 25.0:
            return 6
        if value < 50.0:
            return 7
        if value < 100.0:
            return 8
        return 9
