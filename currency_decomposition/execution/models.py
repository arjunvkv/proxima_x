from dataclasses import dataclass
from typing import Optional


@dataclass
class ExecutionRequest:
    symbol: str
    direction: str
    lots: float
    stop_loss: float
    take_profit: float
    confidence: float
    drs_score: float
    currency_snapshot: dict


@dataclass
class ExecutionReport:
    success: bool
    request: ExecutionRequest
    fill_price: float
    fill_time: float
    position_id: Optional[str] = None
    error: str = ""
    slippage: float = 0.0
    commission: float = 0.0
