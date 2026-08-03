from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from proxima_honest_backtest.engine.types import ExecutionReport, Trade
from proxima_honest_backtest.execution.models import (
    BrokerProfile,
    FillModel,
    LatencyModel,
    SlippageModel,
    SpreadModel,
    _pip_to_price,
    _pnl_to_usd,
)

_PROFILES_DIR = Path(__file__).parent / "broker_profiles"


def _build_profile_from_dict(data: Dict) -> BrokerProfile:
    return BrokerProfile(
        name=data["name"],
        spread=SpreadModel(**data["spread"]),
        slippage=SlippageModel(**data["slippage"]),
        latency=LatencyModel(**data["latency"]),
        fill=FillModel(**data["fill"]),
        commission_per_lot=data["commission_per_lot"],
        commission_type=data.get("commission_type", "per_lot"),
        min_commission=data.get("min_commission", 0.0),
        max_leverage=data.get("max_leverage", 500),
    )


def load_broker_profile(profile_name: str) -> BrokerProfile:
    json_path = _PROFILES_DIR / f"{profile_name.lower()}.json"
    if not json_path.exists():
        available = ", ".join(list_broker_profiles())
        raise FileNotFoundError(
            f"Broker profile '{profile_name}' not found. "
            f"Available profiles: [{available}]"
        )
    with open(json_path, "r") as f:
        data = json.load(f)
    return _build_profile_from_dict(data)


def list_broker_profiles() -> List[str]:
    return sorted(
        p.stem for p in _PROFILES_DIR.glob("*.json")
    )


class ExecutionSimulator:
    def __init__(self, profile_name: str, seed: int = 42) -> None:
        self.profile: BrokerProfile = load_broker_profile(profile_name)
        self._rng: np.random.RandomState = np.random.RandomState(seed)
        self._seed = seed

    @property
    def profile_name(self) -> str:
        return self.profile.name

    def execute_order(
        self,
        side: str,
        quantity: float,
        symbol: str,
        price: float,
        volatility: float,
        hour_utc: int,
        liquidity: float = 0.8,
        timestamp: Optional[datetime] = None,
    ) -> ExecutionReport:
        spread_pips = self.profile.spread.get_spread(symbol, volatility, hour_utc, rng=self._rng)
        slippage_pips = self.profile.slippage.get_slippage(side, volatility, liquidity, rng=self._rng)
        filled, partial_fill, fill_pct = self.profile.fill.should_fill(side, volatility, spread_pips, rng=self._rng)
        latency_ms = self.profile.latency.get_latency(rng=self._rng)

        timestamp = timestamp or datetime.utcnow()

        if not filled:
            trade = Trade(
                timestamp=timestamp,
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=price,
                commission=0.0,
                pnl=0.0,
            )
            return ExecutionReport(
                trade=trade,
                broker_profile=self.profile_name,
                fill_price=price,
                slippage=0.0,
                latency_ms=latency_ms,
                filled=False,
                reject_reason="no_fill",
            )

        effective_quantity = quantity * fill_pct if partial_fill else quantity
        slippage_price = _pip_to_price(slippage_pips, symbol)

        if side.upper() == "BUY":
            fill_price = price + slippage_price
        else:
            fill_price = price - slippage_price

        commission = self.profile.commission_per_lot * (effective_quantity / 100000.0)
        commission += self.profile.min_commission

        trade = Trade(
            timestamp=timestamp,
            symbol=symbol,
            side=side,
            quantity=effective_quantity,
            price=fill_price,
            commission=commission,
            pnl=0.0,
        )

        return ExecutionReport(
            trade=trade,
            broker_profile=self.profile_name,
            fill_price=fill_price,
            slippage=slippage_pips,
            latency_ms=latency_ms,
            filled=True,
            reject_reason="",
        )

    @staticmethod
    def calculate_pnl(
        entry_price: float,
        exit_price: float,
        quantity: float,
        side: str,
        symbol: str,
    ) -> float:
        raw_pnl = (exit_price - entry_price) * quantity if side.upper() == "BUY" else (entry_price - exit_price) * quantity
        return _pnl_to_usd(raw_pnl, symbol, (entry_price + exit_price) / 2)
