"""Program VI — Exogenous Amplitude Discovery.

Data contracts for exogenous session-state amplitude surfaces.
"""

import sys
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, ".")
sys.path.insert(0, "proxima_x")


# ── Session-state snapshot ────────────────────────────────────────────────


@dataclass
class ExogenousState:
    """Raw state vector sampled at a point in time."""

    symbol: str
    ts: float  # unix epoch seconds

    # Session classification
    session: str  # TOKYO / LONDON / NEWYORK / DEADZONE

    # Fixing window the sample falls in (if any)
    fixing_window: Optional[str]  # TOKYO_FIX / WM_FIX / NY_CUT / None

    # Structural flags
    rollover: bool
    liquidity_void: bool
    news_proxy: bool

    # Microstructure measures
    spread: float
    tick_velocity: float


# ── Labelled observation ──────────────────────────────────────────────────


@dataclass
class ExogenousObservation:
    """A single observation pairing a state with a forward move."""

    state: ExogenousState
    horizon_sec: int
    abs_move: float
    signed_move: float


# ── Binned surface entry (one cell of the amplitude surface) ──────────────


@dataclass
class ExogenousSurfaceEntry:
    """Aggregate statistics for a unique (state-bucket, horizon) cell."""

    session: str
    fixing_window: Optional[str]
    rollover: bool
    liquidity_void: bool
    news_proxy: bool

    horizon: int

    # Count
    n: int

    # Distribution of absolute moves
    mean_abs: float
    median_abs: float
    std_abs: float
    p90_abs: float

    # Ratio of mean_abs to typical spread at observation time
    spread_multiple: float

    # Amplitude excess ratio — see Program VI spec
    aer: float

    # Tail exceedance probabilities for k-sigma thresholds
    exceed_prob: dict  # k -> probability, keys = [2.0, 3.0]


# ── Top-level surface summary ─────────────────────────────────────────────


@dataclass
class ExogenousStats:
    """Roll-up statistics for a computed amplitude surface."""

    total_observations: int
    n_sessions: int
    n_horizons: int
    symbols: list[str]


# ── Composite key helper (interpretable, no hashing) ──────────────────────


def make_key(state: ExogenousState) -> str:
    """Deterministic composite key from the exogenous feature vector.

    Fields (pipe-separated, in order):
        session | fixing_window (literal "None" when None) |
        rollover | liquidity_void | news_proxy
    """
    fixing = "None" if state.fixing_window is None else state.fixing_window
    return "|".join(
        [
            state.session,
            fixing,
            str(state.rollover),
            str(state.liquidity_void),
            str(state.news_proxy),
        ]
    )
