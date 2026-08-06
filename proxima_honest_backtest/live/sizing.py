"""Sizing layer — the ONLY place where backtest units become MT5 lots.

Contract (apples-to-apples): 1 backtest trade == 1 MT5 position with matching
dollar exposure. The backtest engine trades in *units* (base-currency notional,
e.g. 10000 units = 0.10 lot on a 100000-contract symbol). Live MT5 trades in
*lots*. This module converts between them using the real symbol contract.

        backtest_units --( / trade_contract_size )--> lots
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple


def _contract_size(symbol_info: Any, contract_size: Optional[float]) -> float:
    if contract_size is not None:
        return float(contract_size)
    cs = getattr(symbol_info, "trade_contract_size", None)
    if cs:
        return float(cs)
    raise ValueError("contract_size not available from symbol_info")


def units_to_lots(symbol_info: Any, units: float, contract_size: Optional[float] = None) -> float:
    """Convert backtest notional units to a valid MT5 volume (lots).

    lots = units / contract_size, clamped to [volume_min, volume_max] and
    aligned to volume_step.
    """
    cs = _contract_size(symbol_info, contract_size)
    if cs <= 0:
        raise ValueError("contract_size must be > 0 to convert units->lots")
    raw = units / cs
    vmin = getattr(symbol_info, "volume_min", 0.01) or 0.01
    vmax = getattr(symbol_info, "volume_max", 100.0) or 100.0
    step = getattr(symbol_info, "volume_step", 0.01) or 0.01
    lots = min(max(raw, vmin), vmax)
    lots = round(lots / step) * step
    return float(lots)


def lots_to_units(symbol_info: Any, lots: float, contract_size: Optional[float] = None) -> float:
    return float(lots * _contract_size(symbol_info, contract_size))


def validate_volume(symbol_info: Any, lots: float, symbol: str = "") -> Tuple[bool, str]:
    """MT5 volume contract: min <= volume <= max and step-aligned."""
    vmin = getattr(symbol_info, "volume_min", None)
    vmax = getattr(symbol_info, "volume_max", None)
    step = getattr(symbol_info, "volume_step", None)
    if vmin is None or vmax is None or step is None:
        return True, ""  # cannot validate without symbol info
    if lots < vmin or lots > vmax:
        return False, f"{symbol or ''} volume {lots} outside [{vmin},{vmax}]"
    rem = (lots - vmin) % step
    if rem > 1e-9 and (step - rem) > 1e-9:
        return False, f"{symbol or ''} volume {lots} not aligned to step {step}"
    return True, ""


def pip_value_usd(symbol_info: Any, lots: float = 1.0) -> float:
    """USD value of one pip for a given volume on a real symbol.

    MT5 gives tick_value (USD per 1 tick of 1 lot) and tick_size. One pip is
    (pip / tick_size) ticks, so per-lot pip value = tick_value * pip / tick_size.
    """
    tick_value = getattr(symbol_info, "trade_tick_value", None)
    if tick_value is None:
        tick_value = getattr(symbol_info, "tick_value", None)
    tick_size = getattr(symbol_info, "trade_tick_size", None)
    if tick_size is None:
        tick_size = getattr(symbol_info, "tick_size", None)
    if tick_value is None or tick_size is None or tick_size <= 0:
        cs = getattr(symbol_info, "trade_contract_size", 100000.0) or 100000.0
        return float(cs * 0.0001 * lots)
    tick_value = float(tick_value)
    tick_size = float(tick_size)
    pip = 0.01 if "JPY" in getattr(symbol_info, "name", "").upper() else 0.0001
    return float(tick_value * (pip / tick_size) * lots)


def max_loss_usd(symbol_info: Any, lots: float, sl_pips: float) -> float:
    """Worst-case account loss (USD) if emergency SL fills at sl_pips."""
    return pip_value_usd(symbol_info, lots) * sl_pips


@dataclass
class SizingDecision:
    """Result of sizing one decision — ties backtest units to broker lots."""
    symbol: str
    backtest_units: float
    volume_lots: float
    contract_size: float
    pip_value_usd: float
    max_loss_sl_usd: float
    valid: bool
    reason: str = ""


def size_entry(symbol_info: Any, units: float, sl_pips: float = 50.0,
               symbol: str = "") -> SizingDecision:
    """Full sizing for a single ENTER decision."""
    contract = _contract_size(symbol_info, None)
    lots = units_to_lots(symbol_info, units)
    ok, reason = validate_volume(symbol_info, lots, symbol)
    pv = pip_value_usd(symbol_info, lots)
    return SizingDecision(
        symbol=symbol or getattr(symbol_info, "name", ""),
        backtest_units=units,
        volume_lots=lots,
        contract_size=contract,
        pip_value_usd=pv,
        max_loss_sl_usd=pv * sl_pips,
        valid=ok,
        reason=reason,
    )