import sys; sys.path.insert(0, "."); sys.path.insert(0, "proxima_x")

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class MacroEvent:
    """A scheduled macro-economic event with consensus forecast data."""
    ts: float
    currency: str
    impact: str
    name: str
    actual: Optional[float]
    forecast: Optional[float]
    previous: Optional[float]


@dataclass(frozen=True)
class EventProximityState:
    """Describes how close the system is to a known macro event."""
    bucket: str
    impact: Optional[str]
    currency_match: bool
    nearest_event_name: Optional[str]
    seconds_to_event: float


@dataclass(frozen=True)
class EventAmplitudeObservation:
    """Records price action amplitude around an event proximity window."""
    event_state: EventProximityState
    symbol: str
    ts: float
    horizon_sec: int
    abs_move: float
    signed_move: float
    spread: float
