"""TPI Layer 7 — Data types."""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class TPIObservation:
    obs_id: str
    symbol: str
    timestamp: datetime

    tpi: float
    direction: str
    confidence: float
    percentile: float

    session: str
    eligible: bool
    aligned_with_signal: Optional[bool]

    bar_open_time: datetime
    entry_price: float

    resolved_h1: bool = False
    resolved_h3: bool = False

    h1_return: Optional[float] = None
    h3_return: Optional[float] = None

    h1_hit: Optional[bool] = None
    h3_hit: Optional[bool] = None
