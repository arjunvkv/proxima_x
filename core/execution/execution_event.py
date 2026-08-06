"""core/execution/execution_event.py — canonical execution-event record.

The single, shared contract emitted by BOTH PaperBroker (backtest) and
MT5Connector (live) so the same tape can drive both and events diff
1:1. This is the instrumentation layer GPT-7 put at the heart of Track A
execution-lifecycle observability (C3/C4).

Each event describes one execution decision/fill so a downstream live-
shadow comparer can prove: PaperBroker filled at the same bid/ask that
MT5 (or its replay) filled at, for the same signal tick.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ExecutionEvent:
    event_type: str                       # "OPEN" | "CLOSE"
    ticket: Optional[int]
    symbol: str
    side: str                             # "BUY" | "SELL"
    volume: float

    timestamp: float                      # seconds (clock.time)
    bid: float
    ask: float

    requested_price: float                # price the caller asked to fill at
    fill_price: Optional[float]           # actual post-slippage fill

    latency_ms: float = 0.0
    slippage_points: float = 0.0          # fill_price - (ask|bid) at decision
    status: str = "FILLED"

    # optional reconciliation payload (MT5 connection emits these; paper can
    # leave None) so a consumer can diff against history_deals without
    # reverse-engineering net<->gross.
    gross_profit: Optional[float] = None
    commission: Optional[float] = None    # signed, MT5-shaped (neg = cost)
    swap: Optional[float] = None
    net_profit: Optional[float] = None

    # free-form extras preserved through the sink (e.g. requote/retry notes)
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        """Plain serializable form for JSON report emission."""
        return {
            "event_type": self.event_type,
            "ticket": self.ticket,
            "symbol": self.symbol,
            "side": self.side,
            "volume": self.volume,
            "timestamp": self.timestamp,
            "bid": self.bid,
            "ask": self.ask,
            "requested_price": self.requested_price,
            "fill_price": self.fill_price,
            "latency_ms": self.latency_ms,
            "slippage_points": self.slippage_points,
            "status": self.status,
            "gross_profit": self.gross_profit,
            "commission": self.commission,
            "swap": self.swap,
            "net_profit": self.net_profit,
            "extra": dict(self.extra),
        }