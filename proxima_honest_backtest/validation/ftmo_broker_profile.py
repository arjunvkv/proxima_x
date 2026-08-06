"""FTMO broker profile from REAL MT5 symbol/deal data.

Direction of truth for apples-to-apples: the REAL broker is the source of truth.
The backtest simulator must reproduce the same dollar costs. This module pulls
per-symbol contract/tick/volume facts from symbol_info, derives commission-per-
lot from actual closed deals when available, and emits a profile dict the
ExecutionSimulator can consume (real costs + modeled spread/slippage noise).

No hardcoded approximation rates — everything is read from the account.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def symbol_facts(mt5: Any, symbol: str) -> Dict[str, Any]:
    info = mt5.symbol_info(symbol)
    if info is None:
        return {}
    tick = getattr(info, "trade_tick_size", None)
    return {
        "symbol": symbol,
        "digits": getattr(info, "digits", None),
        "point": getattr(info, "point", None),
        "trade_contract_size": getattr(info, "trade_contract_size", None),
        "trade_tick_size": getattr(info, "trade_tick_size", None),
        "trade_tick_value": getattr(info, "trade_tick_value", None),
        "volume_min": getattr(info, "volume_min", None),
        "volume_max": getattr(info, "volume_max", None),
        "volume_step": getattr(info, "volume_step", None),
        "swap_long": getattr(info, "swap_long", None),
        "swap_short": getattr(info, "swap_short", None),
    }


def deal_commission_per_lot(mt5: Any, symbols: List[str],
                            lookback_days: int = 7) -> float:
    """Derive average commission per lot (round-turn) from recent closed deals."""
    from datetime import timedelta
    fr = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    deals = mt5.history_deals_get(fr) or []
    comm_lots: List[float] = []
    for d in deals:
        if d.symbol not in symbols:
            continue
        vol = float(getattr(d, "volume", 0.0) or 0.0)
        comm = float(getattr(d, "commission", 0.0) or 0.0)
        if vol > 0 and comm != 0:
            comm_lots.append(comm / vol)  # commission per lot
    if not comm_lots:
        return 0.0
    return round(sum(comm_lots) / len(comm_lots), 4)


def build_profile(mt5: Any, symbols: List[str],
                  base_profile: Optional[Dict[str, Any]] = None,
                  commission_per_lot: Optional[float] = None) -> Dict[str, Any]:
    """Merge real broker facts into a simulator-consumable profile dict."""
    comm = commission_per_lot if commission_per_lot is not None \
        else deal_commission_per_lot(mt5, symbols)
    facts = {s: symbol_facts(mt5, s) for s in symbols}
    profile = dict(base_profile or {})
    profile["name"] = "FTMO_REAL"
    profile["commission_per_lot"] = comm
    profile["commission_type"] = "per_lot"
    profile["_facts"] = facts
    return profile


def save_profile(profile: Dict[str, Any], path: str) -> None:
    with open(path, "w") as f:
        json.dump(profile, f, indent=2, default=str)