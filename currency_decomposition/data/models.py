from dataclasses import dataclass, field
from typing import Optional
from config.settings import CURRENCY_LIST


@dataclass
class Tick:
    symbol: str
    timestamp: float
    bid: float
    ask: float
    volume: float

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float:
        return self.ask - self.bid


@dataclass
class TickBatch:
    ticks: list[Tick]
    market_timestamp: float
    sequence: int
    received_timestamp: float


@dataclass
class CurrencyState:
    strengths: dict[str, float] = field(default_factory=lambda: {c: 0.0 for c in CURRENCY_LIST})
    prior: dict[str, float] = field(default_factory=lambda: {c: 0.0 for c in CURRENCY_LIST})
    covariance: list[list[float]] = field(default_factory=lambda: [[1.0 if i == j else 0.0 for j in range(len(CURRENCY_LIST))] for i in range(len(CURRENCY_LIST))])
    observability: dict[str, float] = field(default_factory=lambda: {c: 0.0 for c in CURRENCY_LIST})
    quality: float = 0.0
    coverage: float = 0.0
    last_solve_timestamp: float = 0.0
    solve_count: int = 0


@dataclass
class DirectionHypothesis:
    symbol: str
    direction: float
    confidence: float
    base_strength: float
    quote_strength: float
    residual: float
    drs_score: float = 0.0
    timestamp: float = 0.0


@dataclass
class PaperPosition:
    id: str
    symbol: str
    direction: str
    entry_price: float
    current_price: float
    entry_time: float
    lots: float
    stop_loss: float
    take_profit: float
    drs_entry: float
    currency_strengths_entry: dict[str, float] = field(default_factory=dict)
    pnl: float = 0.0
    entry_market_timestamp: float = 0.0


@dataclass
class TradeRecord:
    id: str
    symbol: str
    direction: str
    entry_price: float
    exit_price: Optional[float] = None
    entry_time: float = 0.0
    exit_time: Optional[float] = None
    confidence: float = 0.0
    drs_score: float = 0.0
    currency_snapshot: Optional[dict] = None
    exit_reason: str = ""
    pnl: Optional[float] = None


@dataclass
class ExecutionResult:
    success: bool
    position_id: Optional[str] = None
    reason: str = ""
    price: float = 0.0


@dataclass
class HealthStatus:
    state: str = "OK"
    mt5_ok: bool = True
    tick_quality: float = 1.0
    graph_quality: float = 1.0
    last_snapshot_ok: bool = True
    solve_latency_ms: float = 0.0
    memory_mb: float = 0.0


@dataclass
class StateEnvelope:
    market_timestamp: float
    wall_timestamp: float
    schema_version: int
    checksum: str
    payload: dict
