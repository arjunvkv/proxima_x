"""Strict boundary contracts between tri-system layers.

SignalOutput — immutable output from Signal System → Decision System
Decision     — arbitration output from Decision System → Execution System
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class SignalOutput:
    """Immutable signal contract from Signal System to Decision System.

    The Signal System produces these; the Decision System reads them.
    No subsystem may modify a SignalOutput after creation.
    """
    symbol: str
    direction: int  # -1 (SELL), 0 (FLAT), +1 (BUY)
    strength: float  # confidence magnitude [0.0, 1.0]
    horizon: int  # lookahead bars
    ecdf_rank: float  # [0.0, 1.0]
    confidence: float  # overall confidence [0.0, 1.0]
    source: str  # "oss", "shadow", "arbitrated"


@dataclass(frozen=True)
class Decision:
    """Arbitration output from Decision System to Execution System.

    The Decision System evaluates SignalOutput objects against portfolio
    and risk constraints; the Execution System reads Decisions only.
    """
    symbol: str
    entry_authorized: bool
    exit_authorized: bool
    rejection_reason: Optional[str] = None
    exit_reason: Optional[str] = None
    exit_fraction: float = 0.0  # [0.0, 1.0]


@dataclass(frozen=True)
class PortfolioState:
    """Read-only portfolio snapshot for Decision System evaluation."""
    current_positions: int = 0
    max_positions: int = 6
    positions_by_symbol: dict = field(default_factory=dict)
    account_balance: float = 0.0
    day_pnl: float = 0.0
    session_pnl: float = 0.0


@dataclass(frozen=True)
class GateResult:
    """Result of a single gating check."""
    gate: str
    passed: bool
    reason: str = ""
    value: Optional[float] = None
