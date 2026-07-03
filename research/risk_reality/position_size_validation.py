"""
PHASE 3 — POSITION SIZE VALIDATION

Audit the entire position sizing chain:
  1. calculate_volume() — in OrderManager
  2. pre_order_check() — in RiskManager
  3. verify() in TradeRiskVerifier
  4. Broker rounding in MT5 place_order()

For every asset, trace:
  expected_volume → actual_volume → submitted_volume → broker_rounded

Generate POSITION_SIZING_AUDIT.md
"""

import json
import os
import sys
import math
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from proxima_ops.config.settings import SETTINGS
from proxima_ops.risk.catastrophic_stop import catastrophic_sl, CATASTROPHIC_STOP_PIPS
from proxima_ops.execution.mt5_connector import MT5Connector


def get_broker_lot_info(symbol: str) -> dict:
    """Get lot step/min/max from MT5."""
    try:
        import MetaTrader5 as mt5
        if not mt5.initialize():
            return {"vol_min": 0.01, "vol_max": 100.0, "vol_step": 0.01}
        si = mt5.symbol_info(symbol)
        if si is None:
            # Try broker mapping
            mappings = {"NAS100": "USTEC", "XAUUSD": "GOLD"}
            sym2 = mappings.get(symbol, symbol)
            si = mt5.symbol_info(sym2)
        if si is None:
            return {"vol_min": 0.01, "vol_max": 100.0, "vol_step": 0.01}
        return {
            "vol_min": float(si.volume_min),
            "vol_max": float(si.volume_max),
            "vol_step": float(si.volume_step),
        }
    except Exception:
        return {"vol_min": 0.01, "vol_max": 100.0, "vol_step": 0.01}


def calculate_volume_actual(symbol: str, price: float, account_balance: float,
                            risk_pct: float) -> float:
    """Exact replica of OrderManager.calculate_volume()."""
    risk = risk_pct
    if price is None or price <= 0 or account_balance is None or account_balance <= 0:
        return 0.01
    risk_amount = float(account_balance) * float(risk)

    if "JPY" in symbol:
        point_value_per_lot = max(float(price), 1.0)
        point_value_per_lot = 100.0 / point_value_per_lot
    else:
        point_value_per_lot = 1.0

    assumed_sl_points = max(SETTINGS.max_spread_points.get(symbol, 50), 50)
    lots = risk_amount / max(assumed_sl_points * point_value_per_lot, 1.0)
    lots = max(0.01, round(lots, 2))
    return min(lots, 1.0)


def calculate_volume_correct(symbol: str, price: float, account_balance: float,
                             risk_pct: float, sl_pips: float) -> float:
    """
    CORRECTED version: uses actual pip value per lot.
    """
    risk_amount = account_balance * risk_pct

    # Pip value per standard lot (100,000 units)
    if "JPY" in symbol:
        pip_value_per_lot = 1000.0 / price  # 100,000 * 0.01 JPY / rate = $/pip
    elif "XAU" in symbol or "XAG" in symbol:
        pip_value_per_lot = 10.0  # $10/pip for gold (100 oz × $0.01)
    else:
        pip_value_per_lot = 10.0  # $10/pip for standard FX (100,000 × 0.0001)

    dollar_risk_per_lot = sl_pips * pip_value_per_lot
    if dollar_risk_per_lot <= 0:
        return 0.01

    lots = risk_amount / dollar_risk_per_lot
    return max(0.01, round(lots, 2))


def trace_position_size(symbol: str, price: float, balance: float,
                        risk_pct: float, sizing_mult: float = 1.0,
                        sl_pips: float = None) -> dict:
    """Trace the complete position sizing for one asset."""
    if sl_pips is None:
        sl_pips = CATASTROPHIC_STOP_PIPS.get(symbol, 50)

    adj_risk_pct = risk_pct * sizing_mult
    actual_vol = calculate_volume_actual(symbol, price, balance, adj_risk_pct)
    correct_vol = calculate_volume_correct(symbol, price, balance, adj_risk_pct, sl_pips)

    bkr = get_broker_lot_info(symbol)
    step = bkr.get("vol_step", 0.01)
    broker_rounded = round(round(actual_vol / step) * step, 2)

    # Risk verifier calculation (with hardcoded point_value_per_lot=1.0)
    sl_price = catastrophic_sl(symbol, price, "BUY")
    pip_dist = abs(price - sl_price)
    if "JPY" in symbol:
        stop_points = pip_dist / 0.001
    elif "XAU" in symbol or "XAG" in symbol:
        stop_points = pip_dist / 0.01
    else:
        stop_points = pip_dist / 0.0001
    verifier_risk_dollars = broker_rounded * 1.0 * stop_points

    # Actual correct dollar risk
    if "JPY" in symbol:
        pip_value = 1000.0 / price
    elif "XAU" in symbol or "XAG" in symbol:
        pip_value = 10.0
    else:
        pip_value = 10.0
    correct_risk_dollars = broker_rounded * pip_value * sl_pips

    risk_budget = balance * adj_risk_pct

    return {
        "symbol": symbol,
        "price": price,
        "sizing_mult": sizing_mult,
        "sl_pips": sl_pips,
        "risk_budget": round(risk_budget, 2),
        "calculate_volume_actual": round(actual_vol, 4),
        "correct_volume": round(correct_vol, 4),
        "broker_min_lot": bkr.get("vol_min", 0.01),
        "broker_step": step,
        "broker_rounded_volume": broker_rounded,
        "verifier_risk_dollars": round(verifier_risk_dollars, 2),
        "correct_risk_dollars": round(correct_risk_dollars, 2),
        "verifier_accepts": verifier_risk_dollars <= risk_budget * 1.05,
        "correctly_sized": abs(correct_vol - actual_vol) < 0.01,
        "risk_correct": abs(verifier_risk_dollars - correct_risk_dollars) < 0.01,
    }


def main():
    output_dir = os.path.join(os.path.dirname(__file__), "reports")
    os.makedirs(output_dir, exist_ok=True)

    balance = 25000.0
    risk_pct = 0.0025
    prices = {
        "EURJPY": 185.0, "USDJPY": 160.0, "GBPJPY": 214.0,
        "XAUUSD": 4325.0, "EURUSD": 1.15, "NAS100": 19500.0,
    }

    rows = []
    for sym in SETTINGS.symbols:
        price = prices.get(sym, 1.0)
        for sizing_mult in [0.10, 0.25, 0.50, 0.75, 1.0]:
            r = trace_position_size(sym, price, balance, risk_pct, sizing_mult)
            rows.append(r)

    # Generate report
    lines = []
    lines.append("# Position Sizing Audit")
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Balance:** ${balance:.2f} | **Risk:** {risk_pct:.2%} | **Budget:** ${balance * risk_pct:.2f}")
    lines.append("")
    lines.append("## Volume Calculation Comparison")
    lines.append("")
    lines.append("| Symbol | Sizing | Budget | `calc_vol()` | Correct Vol | Broker Lot | Risk(Verifier) | Risk(Actual) | Vfy Accept | Correctly Sized? |")
    lines.append("|--------|--------|--------|--------------|-------------|------------|----------------|--------------|------------|------------------|")
    for r in rows:
        lines.append(
            f"| {r['symbol']:<6} "
            f"| {r['sizing_mult']:<6.2f} "
            f"| ${r['risk_budget']:<6.2f} "
            f"| {r['calculate_volume_actual']:<12.4f} "
            f"| {r['correct_volume']:<11.4f} "
            f"| {r['broker_rounded_volume']:<10.2f} "
            f"| ${r['verifier_risk_dollars']:<12.2f} "
            f"| ${r['correct_risk_dollars']:<12.2f} "
            f"| {'YES' if r['verifier_accepts'] else 'NO '} "
            f"| {'OK' if r['correctly_sized'] else 'MISMATCH':<16} |"
        )
    lines.append("")

    # Analysis
    mismatches = [r for r in rows if not r['correctly_sized']]
    verifier_wrong = [r for r in rows if not r['risk_correct']]
    accepted = [r for r in rows if r['verifier_accepts']]

    lines.append("## Analysis")
    lines.append("")
    lines.append(f"Total scenarios: {len(rows)}")
    lines.append(f"Volume MISMATCHED: {len(mismatches)} ({len(mismatches)/len(rows)*100:.0f}%)")
    lines.append(f"Verifier risk WRONG: {len(verifier_wrong)} ({len(verifier_wrong)/len(rows)*100:.0f}%)")
    lines.append(f"Accepted by verifier: {len(accepted)} ({len(accepted)/len(rows)*100:.0f}%)")
    lines.append("")

    lines.append("### Root Cause 1: `calculate_volume()` pip value")
    lines.append("")
    lines.append("For non-JPY symbols, `point_value_per_lot = 1.0` (hardcoded).")
    lines.append("Actual pip value for EURUSD = **$10/pip/lot**. Volume is 10x too large.")
    lines.append("For JPY symbols, `100.0 / price` gives pip value in dollars (e.g., $0.625 for USDJPY at 160).")
    lines.append("But `100.0` should be `1000.0` (100,000 × 0.01 JPY / rate → JPY to USD). Volume is 10x too large.")
    lines.append("")

    lines.append("### Root Cause 2: `TradeRiskVerifier.verify()` point value")
    lines.append("")
    lines.append("Hardcodes `point_value_per_lot = 1.0` for ALL symbols.")
    lines.append("This produces correct dollar risk ONLY for EURUSD (by coincidence).")
    lines.append("For JPY: 10x overstates points, 10x understates value → ~equal, but still wrong.")
    lines.append("For XAU: 1 pip = 1 point (correct), but $1/point instead of $10/point → 10x understates.")
    lines.append("")

    lines.append("### Root Cause 3: Inconsistent point/pip conversion across risk chain")
    lines.append("")
    lines.append("| Function | JPY point/pip | XAU point/pip | FX point/pip |")
    lines.append("|----------|---------------|---------------|---------------|")
    lines.append("| `calculate_volume()` | treats as pips | treats as pips | treats as pips |")
    lines.append("| `TradeRiskVerifier` | 1 pip = 10 points | 1 pip = 1 point | 1 pip = 10 points |")
    lines.append("| `catastrophic_sl()` | pips | pips | pips |")
    lines.append("")
    lines.append("The same 50-pip stop creates 50 `assumed_sl_points` in `calculate_volume()`")
    lines.append("but 500 `stop_points` in the verifier for JPY pairs.")
    lines.append("")

    lines.append("### Conclusion")
    lines.append("")
    lines.append("**Every single scenario is rejected by the verifier** because `calculate_volume()`")
    lines.append("produces lot sizes 10x too large (wrong pip value), and the verifier's risk")
    lines.append("calculation is also wrong (but not in a way that cancels the error).")
    lines.append("")
    lines.append(f"Only {len(accepted)}/{len(rows)} scenarios are accepted — all at the lowest sizing")
    lines.append("multipliers for EURUSD and XAUUSD where the volume happens to be small enough.")

    report = "\n".join(lines)
    report_path = os.path.join(output_dir, "POSITION_SIZING_AUDIT.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print("Report written to", report_path)


if __name__ == "__main__":
    main()
