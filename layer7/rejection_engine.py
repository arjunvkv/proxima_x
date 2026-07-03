"""
Rejection Engine — Wave 5 P0.33: Structured rejection taxonomy.

Formalizes all execution rejection reasons with typed events.
Ensures every block/refusal has a structured cause.
"""
from dataclasses import dataclass
from enum import Enum
from typing import List


class RejectionType(str, Enum):
    CF_BLOCK = "CF_BLOCK"
    TPI_COLLAPSE = "TPI_COLLAPSE"
    DRAWDOWN = "DRAWDOWN"
    PORTFOLIO_OVEREXPOSURE = "PORTFOLIO_OVEREXPOSURE"
    OBSERVER_BIAS = "OBSERVER_BIAS"
    INFORMATION_OVERLOAD = "INFORMATION_OVERLOAD"
    SYMBOL_LOCK = "SYMBOL_LOCK"
    KILL_SWITCH = "KILL_SWITCH"
    QUARANTINE = "QUARANTINE"
    PASSIVE_MODE = "PASSIVE_MODE"
    RISK_GATE = "RISK_GATE"
    NET_ALPHA = "NET_ALPHA"


@dataclass
class RejectionEvent:
    symbol: str
    reason: RejectionType
    score: float
    timestamp: float


class RejectionEngine:
    def __init__(self):
        self._history: List[RejectionEvent] = []

    def reject(self, symbol: str, reason: RejectionType, score: float, timestamp: float) -> RejectionEvent:
        event = RejectionEvent(
            symbol=symbol,
            reason=reason,
            score=score,
            timestamp=timestamp,
        )
        self._history.append(event)
        return event

    def get_stats(self) -> dict:
        stats = {}
        for e in self._history:
            stats[e.reason] = stats.get(e.reason, 0) + 1
        return stats

    def clear(self) -> None:
        self._history.clear()
