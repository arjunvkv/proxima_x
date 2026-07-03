"""
Phase V — Amplitude Regime Mapper

Maps tick-level microstructure features (burst score, compression density,
tick velocity, entropy slope, spread) into one of five distinct amplitude
regimes.  The mapper implements a deterministic, priority-ordered rule set
that captures the physics of price-amplitude expansion and contraction.
"""

import sys
from typing import List

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "proxima_x")

from research.amplitude.schemas import AmplitudeState


# ---------------------------------------------------------------------------
# Regime ID constants
# ---------------------------------------------------------------------------
QUIET_COMPRESSION = "QUIET_COMPRESSION"
ELASTIC_BUILDUP = "ELASTIC_BUILDUP"
BURST_RELEASE = "BURST_RELEASE"
CHAOTIC_EXPANSION = "CHAOTIC_EXPANSION"
DEAD_FLOW = "DEAD_FLOW"

_ALL_REGIMES = [
    QUIET_COMPRESSION,
    ELASTIC_BUILDUP,
    BURST_RELEASE,
    CHAOTIC_EXPANSION,
    DEAD_FLOW,
]

_REGIME_DESCRIPTIONS = {
    QUIET_COMPRESSION: (
        "Low-velocity, high-compression regime where price action is tightly "
        "compressed and tick activity is subdued — typical of consolidation "
        "or pre-breakout quiet periods."
    ),
    ELASTIC_BUILDUP: (
        "Moderate-velocity, high-compression regime where energy is being "
        "stored in the microstructure; tick activity picks up but price "
        "remains constrained, suggesting an imminent directional release."
    ),
    BURST_RELEASE: (
        "High-velocity, low-compression regime where accumulated energy "
        "discharges as a sharp burst of ticks and price expansion — the "
        "microstructure equivalent of an elastic snap."
    ),
    CHAOTIC_EXPANSION: (
        "Very-high-velocity, low-compression, wide-spread regime where "
        "the market enters a disordered, high-volume state with poor "
        "price discovery and elevated noise."
    ),
    DEAD_FLOW: (
        "Extremely-low-velocity, low-compression regime where both tick "
        "activity and price movement are minimal — the market is essentially "
        "stagnant or asleep."
    ),
}


# ---------------------------------------------------------------------------
# AmplitudeRegimeMapper
# ---------------------------------------------------------------------------


class AmplitudeRegimeMapper:
    """Maps microstructure state features to a qualitative amplitude regime.

    The mapper applies a deterministic rule hierarchy that prioritises
    burst-release and chaotic-expansion states (rare, high-energy) over
    quieter regimes.  Every state is assigned exactly one of the five
    regime labels.
    """

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def map(self, state: AmplitudeState) -> str:
        """Map an ``AmplitudeState`` to its amplitude regime.

        The ``burst_score`` feature is approximated from ``state.sal_score``
        (both lie in the 0–1 range and measure short-horizon micro-structural
        intensity).  All other features are read directly from the state.
        """
        return self.map_from_features(
            burst_score=state.sal_score,
            compression_density=state.compression_density,
            tick_velocity=state.tick_velocity,
            entropy_slope=state.entropy_slope,
            spread=state.spread,
        )

    def map_from_features(
        self,
        burst_score: float,
        compression_density: float,
        tick_velocity: float,
        entropy_slope: float,
        spread: float,
    ) -> str:
        """Map raw microstructure features to an amplitude regime.

        Parameters
        ----------
        burst_score : float
            Short-horizon burst intensity, typically 0–1.
        compression_density : float
            Fraction of recent price changes below half the median — 0–1.
        tick_velocity : float
            Ticks per second over the observation window.
        entropy_slope : float
            Rate of change of microstructural entropy.
        spread : float
            Current bid-ask spread in basis points.

        Returns
        -------
        str
            One of the five regime ID constants.
        """
        # Coerce to native floats for clean comparisons.
        b = float(burst_score)
        c = float(compression_density)
        v = float(tick_velocity)
        s = float(spread)

        # ---- Priority-ordered rule hierarchy -------------------------

        # 1. BURST_RELEASE — high burst, high velocity, low compression.
        if b > 0.3 and v > 15 and c < 0.3:
            return BURST_RELEASE

        # 2. CHAOTIC_EXPANSION — very high velocity, low compression, wide spread.
        if v > 20 and c < 0.2 and s > 0.3:
            return CHAOTIC_EXPANSION

        # 3. DEAD_FLOW — extremely low velocity, low compression.
        if v < 2 and c < 0.2:
            return DEAD_FLOW

        # 4. QUIET_COMPRESSION — low velocity, high compression.
        if v < 5 and c > 0.5:
            return QUIET_COMPRESSION

        # 5. ELASTIC_BUILDUP — moderate velocity, high compression.
        if 5 <= v <= 20 and c > 0.4:
            return ELASTIC_BUILDUP

        # ---- Fallback: velocity-tier default -------------------------
        if v < 10:
            return QUIET_COMPRESSION
        if v <= 20:
            return ELASTIC_BUILDUP
        return CHAOTIC_EXPANSION

    # ------------------------------------------------------------------
    # Class / static helpers
    # ------------------------------------------------------------------

    @classmethod
    def describe(cls, regime_id: str) -> str:
        """Return a human-readable description of the given *regime_id*."""
        return _REGIME_DESCRIPTIONS.get(
            regime_id,
            f"Unknown regime: {regime_id!r}",
        )

    @staticmethod
    def all_regimes() -> List[str]:
        """Return the list of all five amplitude regime names."""
        return list(_ALL_REGIMES)
