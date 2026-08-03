from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np

_RNG = Union[np.random.RandomState, np.random.Generator]


def _pip_value(symbol: str) -> float:
    """Return pip value in price units for a given symbol."""
    if symbol.upper().endswith("JPY"):
        return 0.01
    return 0.0001


def _pip_to_price(pips: float, symbol: str) -> float:
    """Convert pips to price units."""
    return pips * _pip_value(symbol)


def _quote_currency(symbol: str) -> str:
    """Return quote currency for a forex symbol."""
    ccies = ["USD", "EUR", "JPY", "GBP", "AUD", "NZD", "CAD", "CHF"]
    for c in ccies:
        if symbol.endswith(c):
            return c
    return "USD"


def _pnl_to_usd(pnl: float, pair: str, price: float) -> float:
    """Convert PnL from quote currency to USD using approximate rate."""
    quote = _quote_currency(pair)
    if quote == "USD":
        return pnl
    if quote == "JPY":
        return pnl / price if price > 0 else pnl
    if quote == "EUR":
        return pnl * 1.10
    if quote == "GBP":
        return pnl * 1.28
    if quote == "AUD":
        return pnl * 0.65
    if quote == "NZD":
        return pnl * 0.60
    if quote == "CAD":
        return pnl * 0.73
    if quote == "CHF":
        return pnl * 1.12
    return pnl


@dataclass
class SpreadModel:
    base_spread_pips: float
    std_spread_pips: float
    min_spread_pips: float
    max_spread_pips: float

    def get_spread(self, symbol: str, volatility: float, hour_utc: int, rng: Optional[_RNG] = None) -> float:
        rng = rng if rng is not None else np.random
        std_scale = self.std_spread_pips * (1.0 + 0.5 * volatility)
        spread = rng.normal(loc=self.base_spread_pips, scale=std_scale)

        asian_session = hour_utc < 7 or hour_utc >= 20
        if asian_session and spread < self.max_spread_pips:
            spread *= 1.2

        return float(np.clip(spread, self.min_spread_pips, self.max_spread_pips))

    def get_spread_quote_ccy(self, spread_pips: float, symbol: str) -> float:
        return spread_pips * _pip_value(symbol)


@dataclass
class SlippageModel:
    mean_slippage_pips: float
    std_slippage_pips: float
    max_slippage_pips: float

    def get_slippage(self, side: str, volatility: float, liquidity: float, rng: Optional[_RNG] = None) -> float:
        rng = rng if rng is not None else np.random
        _ = side
        liq_penalty = 1.0 + (1.0 - liquidity) * 0.5
        vol_scale = 1.0 + volatility
        adj_mean = self.mean_slippage_pips * vol_scale * liq_penalty
        adj_std = self.std_slippage_pips * (1.0 + 0.5 * volatility) * (1.0 + (1.0 - liquidity) * 0.3)
        slippage = rng.normal(loc=adj_mean, scale=adj_std)
        return float(np.clip(slippage, 0.0, self.max_slippage_pips))


@dataclass
class LatencyModel:
    base_ms: float
    std_ms: float
    jitter_ms: float

    def get_latency(self, rng: Optional[_RNG] = None) -> float:
        rng = rng if rng is not None else np.random
        total_std = self.std_ms + self.jitter_ms
        latency = rng.normal(loc=self.base_ms, scale=total_std)
        return max(0.0, latency)

    def get_latency_with_network_load(self, load_factor: float, rng: Optional[_RNG] = None) -> float:
        return self.get_latency(rng=rng) * (1.0 + load_factor)


@dataclass
class FillModel:
    fill_rate: float
    requote_rate: float
    partial_fill_prob: float

    def should_fill(
        self, side: str, volatility: float, spread_pips: float,
        rng: Optional[_RNG] = None,
    ) -> Tuple[bool, bool, float]:
        rng = rng if rng is not None else np.random
        _ = side
        effective_fill_rate = self.fill_rate * (1.0 - 0.2 * volatility)
        effective_fill_rate = max(0.0, min(1.0, effective_fill_rate))

        if rng.random() > effective_fill_rate:
            return (False, False, 0.0)

        requote_prob = self.requote_rate * (1.0 + spread_pips / 5.0)
        requote_prob = min(1.0, requote_prob)
        if rng.random() < requote_prob:
            return (False, False, 0.0)

        if rng.random() < self.partial_fill_prob:
            fill_pct = float(rng.uniform(0.3, 0.9))
            return (True, True, fill_pct)

        return (True, False, 1.0)


@dataclass
class BrokerProfile:
    name: str
    spread: SpreadModel
    slippage: SlippageModel
    latency: LatencyModel
    fill: FillModel
    commission_per_lot: float
    commission_type: str = "per_lot"
    min_commission: float = 0.0
    max_leverage: int = 500
