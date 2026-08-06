"""data.execution_cost — shared execution-friction model for backtest/replay + live.

Phase 2 (apples-to-apples friction): backtest and live must pay the SAME
gross costs so a strategy that survives replay provably survives live. This
module is the single source of truth for commission/slippage friction. It is
consumed by the replay broker (core/adapters/broker.py) so paper fills are
charged like the live MT5 path.

Canonical tick ``spread`` is always ``ask - bid`` in price units
(see data/canonical_tick.py); spread friction is paid naturally by filling at
ask (BUY) / bid (SELL), so the model here only adds what a mid-fill would
miss: commission + slippage.

Worked example (EURJPY, 1.0 lot, round trip, commission = $3.5/lot/side):
  commission           $3.50 per leg -> $7.00 round trip
  spread               paid by fill side (ask-bid)
  slippage             uniform 0..max bps x notional per entry (optional)
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Optional, Tuple

logger = logging.getLogger("proxima.execution")

# Per-lot per-side commission, FTMO raw/zero-account-like. Override via
# PROXIMA_COMMISSION_PER_LOT env to match the account actually deployed on.
# 0.0 keeps backtests identical to legacy behaviour; set to your FTMO rate to
# make replay pay the same gross cost as live.
DEFAULT_COMMISSION_PER_LOT: float = 0.0

# Realistic one-way slippage cap in basis points (0.03% of notional).
DEFAULT_SLIPPAGE_BPS: float = 3.0


def pip_value_per_lot(symbol: str, price: Optional[float] = None) -> float:
    """USD value of one pip per 1.0 lot for the given symbol (MT5 semantics)."""
    s = symbol.upper()
    if "XAU" in s or "GOLD" in s:
        return 1.0  # $1 per $0.10 gold move per lot
    if "JPY" in s and "XAU" not in s and "GOLD" not in s:
        # JPY quote: 1 pip = 0.01 JPY per unit -> 1000 JPY per lot / rate = USD
        if price and price > 0:
            return 1000.0 / price
        return 8.0  # ~118 EURJPY/USDJPY median baseline
    # Direct USD-quote: 1 pip = $10 per lot
    return 10.0


@dataclass
class ExecutionCost:
    """Shared friction parameters (commission + slippage).

    Attributes:
        commission_per_lot: per-lot, per-side order commission (USD).
        min_commission: flat minimum charged per order when commission != 0 (USD).
        slippage_bps_range: uniform slippage band in basis points per entry.
        enabled: when False, commission/slippage cost is zeroed (friction omitted).
    """
    commission_per_lot: float = DEFAULT_COMMISSION_PER_LOT
    min_commission: float = 0.0
    slippage_bps_range: Tuple[float, float] = (0.0, 0.0)
    enabled: bool = True

    def commission(self, volume: float) -> float:
        """Per-leg (one side) commission for `volume` lots, USD."""
        if not self.enabled or self.commission_per_lot <= 0:
            return 0.0
        raw = self.commission_per_lot * volume
        if self.min_commission > 0 and raw < self.min_commission:
            return self.min_commission
        return raw

    def round_trip_commission(self, volume: float) -> float:
        return round(self.commission(volume) * 2, 8)

    def slippage_bps(self, symbol: str, side: str) -> float:
        """Deterministic per-symbol/leg slippage draw in basis points."""
        if not self.enabled:
            return 0.0
        lo, hi = self.slippage_bps_range
        if hi <= 0:
            return 0.0
        r = random.Random(f"{symbol.upper()}|{side.upper()}|cost")
        return r.uniform(lo, hi)

    def slippage_price(self, symbol: str, side: str, price: float) -> float:
        """Fill-price adjustment in price units (BUY pays up, SELL pays down)."""
        bps = self.slippage_bps(symbol, side)
        if bps <= 0:
            return price
        adj = price * bps / 10000.0
        return price + adj if side.upper() == "BUY" else price - adj


@dataclass
class AccountModel:
    """Per-symbol friction bundle; built from Settings + per-symbol costs."""
    symbol: str
    commission_per_lot: float = DEFAULT_COMMISSION_PER_LOT
    slippage_bps_range: Tuple[float, float] = (0.0, 0.0)
    enabled: bool = True
    _cost: ExecutionCost = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._cost = ExecutionCost(
            commission_per_lot=self.commission_per_lot,
            slippage_bps_range=self.slippage_bps_range,
            enabled=self.enabled,
        )

    @property
    def cost(self) -> ExecutionCost:
        return self._cost
