"""
PHASE 1 — FULL RISK TRACE

For every triggered signal, trace the complete risk calculation chain:
  timestamp, symbol, account_balance, risk_percent, risk_budget,
  entry_price, stop_distance_points, stop_distance_pips, pip_value,
  raw_position_size, broker_min_lot, broker_max_lot, broker_step,
  adjusted_volume, calculated_risk_dollars, accepted, rejected, rejection_reason

Persists to risk_trace.jsonl.

Usage:
    python research/risk_reality/full_risk_trace.py

Dependencies:
    - MetaTrader5 (for broker symbol info)
    - Reads from the demo's DuckDB trade ledger
"""

import json
import os
import sys
import math
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from proxima_ops.config.settings import SETTINGS
from proxima_ops.risk.catastrophic_stop import (
    catastrophic_sl, catastrophic_tp, CATASTROPHIC_STOP_PIPS, pip_distance,
)
from proxima_ops.execution.mt5_connector import MT5Connector


# ---------------------------------------------------------------------------
# Broker info cache
# ---------------------------------------------------------------------------
_broker_info: dict = {}


def get_broker_info(symbol: str) -> dict:
    """Cache and return MT5 symbol info."""
    global _broker_info
    if symbol in _broker_info:
        return _broker_info[symbol]
    try:
        import MetaTrader5 as mt5
        if not mt5.initialize():
            return {}
        si = mt5.symbol_info(symbol)
        if si is None:
            return {}
        info = {
            "volume_min": float(si.volume_min),
            "volume_max": float(si.volume_max),
            "volume_step": float(si.volume_step),
            "trade_stops_level": int(si.trade_stops_level),
            "digits": int(si.digits),
            "point": float(si.point),
            "trade_mode": si.trade_mode,
        }
        _broker_info[symbol] = info
        return info
    except Exception:
        return {}


def get_broker_symbol(sym: str) -> str:
    """Resolve broker symbol using MT5 connector logic."""
    try:
        import MetaTrader5 as mt5
        if mt5.symbol_info(sym) is not None:
            return sym
        mappings = {

            "XAUUSD": ["GOLD"],
            "EURJPY": ["EURJPY.", "EURJPYecn", "EURJPYm"],
            "USDJPY": ["USDJPY.", "USDJPYecn", "USDJPYm"],
            "GBPJPY": ["GBPJPY.", "GBPJPYecn", "GBPJPYm"],
        }
        for alt in mappings.get(sym.upper(), []):
            if mt5.symbol_info(alt) is not None:
                return alt
    except Exception:
        pass
    return sym


# ---------------------------------------------------------------------------
# Core trace function
# ---------------------------------------------------------------------------
def trace_signal(
    symbol: str,
    account_balance: float = 25000.0,
    risk_pct: float = 0.0025,
    sizing_mult: float = 1.0,
    entry_price: float = None,
) -> dict:
    """
    Trace the full risk decision for one signal.
    Simulates the exact calculation chain used by run_proxima_demo.py.
    """
    record = {
        "timestamp": datetime.now().isoformat(),
        "symbol": symbol,
        "account_balance": account_balance,
        "risk_percent": risk_pct,
        "adjusted_risk_pct": risk_pct * sizing_mult,
        "sizing_multiplier": sizing_mult,
    }

    # 1. Risk budget
    risk_budget = account_balance * risk_pct * sizing_mult
    record["risk_budget"] = round(risk_budget, 2)

    # 2. Entry price (use current tick or fallback)
    if entry_price is None:
        try:
            mt5_obj = MT5Connector()
            mt5_obj.connect()
            tick = mt5_obj.get_tick(symbol)
            if tick:
                entry_price = tick["ask"]
            mt5_obj.disconnect()
        except Exception:
            pass
    record["entry_price"] = entry_price or 0.0

    # 3. Catastrophic stop (SL)
    sl_price = catastrophic_sl(symbol, entry_price or 0.0, "BUY")
    record["sl_price"] = round(sl_price, 5)

    # 4. Stop distance
    pip_dist = pip_distance(symbol, entry_price or 0.0, sl_price)
    record["stop_distance_pips"] = round(pip_dist, 1)
    if "JPY" in symbol:
        stop_points = pip_dist * 10  # 1 pip = 10 points for JPY
    elif "XAU" in symbol or "XAG" in symbol:
        stop_points = pip_dist * 1   # 1 pip = 1 point for XAU
    else:
        stop_points = pip_dist * 10  # 1 pip = 10 points for FX
    record["stop_distance_points"] = round(stop_points, 1)

    # 5. Pip value per lot
    if "JPY" in symbol:
        pip_value_per_lot = 100.0 / (entry_price or 1.0) if (entry_price or 0) > 0 else 1.0
    else:
        pip_value_per_lot = 10.0  # $10 per pip per lot for standard FX
    if "XAU" in symbol or "XAG" in symbol:
        pip_value_per_lot = 10.0  # $10 per pip per lot for gold (1 lot = 100 oz)
    record["pip_value_per_lot"] = round(pip_value_per_lot, 4)

    # 6. Raw position size (from calculate_volume logic)
    assumed_sl_points = max(SETTINGS.max_spread_points.get(symbol, 50), 50)
    risk_amount = account_balance * risk_pct * sizing_mult
    # calculate_volume() uses pip_value_per_lot directly as "point value"
    # (the "assumed_sl_points" is actually used as pip count, not broker points)
    point_value = pip_value_per_lot
    record["assumed_sl_points"] = assumed_sl_points
    record["point_value_per_lot"] = round(point_value, 6)

    raw_lots = risk_amount / max(assumed_sl_points * point_value, 1.0)
    raw_lots = round(raw_lots, 2)
    record["raw_position_size"] = raw_lots

    # 7. Broker constraints
    broker_sym = get_broker_symbol(symbol)
    bkr = get_broker_info(broker_sym)
    record["broker_symbol"] = broker_sym
    record["broker_volume_min"] = bkr.get("volume_min", 0.01)
    record["broker_volume_max"] = bkr.get("volume_max", 100.0)
    record["broker_volume_step"] = bkr.get("volume_step", 0.01)

    # 8. Adjusted volume (apply step rounding)
    step = bkr.get("volume_step", 0.01)
    adjusted = round(raw_lots / step) * step
    adjusted = max(bkr.get("volume_min", 0.01), min(adjusted, bkr.get("volume_max", 100.0)))
    record["adjusted_volume"] = round(adjusted, 2)

    # 9. Calculated risk in dollars (using risk verifier logic)
    # The verifier uses point_value_per_lot = 1.0 for all symbols
    if "JPY" in symbol:
        pip_to_points = 10  # 1 pip = 10 points
    elif "XAU" in symbol or "XAG" in symbol:
        pip_to_points = 1   # 1 pip = 1 point
    else:
        pip_to_points = 10  # 1 pip = 10 points
    calc_stop_points = pip_dist * pip_to_points
    calc_dollar_risk = adjusted * 1.0 * calc_stop_points  # verifier's 1.0 point value
    record["calculated_dollar_risk"] = round(calc_dollar_risk, 2)

    # 10. Decision
    record["rejection_threshold"] = round(risk_budget * 1.05, 2)
    if calc_dollar_risk > risk_budget * 1.05:
        record["accepted"] = False
        record["rejected"] = True
        record["rejection_reason"] = (
            f"risk_exceeds_budget: ${calc_dollar_risk:.2f} > ${risk_budget:.2f}"
        )
    else:
        record["accepted"] = True
        record["rejected"] = False
        record["rejection_reason"] = ""

    # 11. Catastrophic stop pips from config
    record["catastrophic_stop_pips"] = CATASTROPHIC_STOP_PIPS.get(symbol, 50)

    return record


def trace_all_symbols(account_balance: float = 25000.0, risk_pct: float = 0.0025):
    """Trace all configured symbols with multiple sizing multipliers."""
    records = []
    for symbol in SETTINGS.symbols:
        for sizing_mult in [0.10, 0.25, 0.50, 0.75, 1.0]:
            rec = trace_signal(
                symbol=symbol,
                account_balance=account_balance,
                risk_pct=risk_pct,
                sizing_mult=sizing_mult,
            )
            records.append(rec)
    return records


def main():
    output_dir = os.path.join(os.path.dirname(__file__), "reports")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "risk_trace.jsonl")

    account = 25000.0
    records = trace_all_symbols(account)
    with open(output_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"Risk trace written to {output_path}")
    print(f"Total records: {len(records)}")
    print()

    # Print summary table
    print(f"{'Symbol':<10} {'Sizing':<8} {'Volume':<8} {'Risk$':<10} {'Budget$':<10} {'Accepted':<10} {'Reason'}")
    print("-" * 80)
    for rec in records:
        print(
            f"{rec['symbol']:<10} "
            f"{rec['sizing_multiplier']:<8.2f} "
            f"{rec['adjusted_volume']:<8.2f} "
            f"${rec['calculated_dollar_risk']:<7.2f} "
            f"${rec['risk_budget']:<8.2f} "
            f"{'YES' if rec['accepted'] else 'NO':<10} "
            f"{rec['rejection_reason'][:40]}"
        )


if __name__ == "__main__":
    main()
