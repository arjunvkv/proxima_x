"""
PHASE 2 — ASSET-SPECIFIC RISK AUDIT

For every asset (EURJPY, USDJPY, GBPJPY, XAUUSD, EURUSD), compute:

  1 lot risk @ current stop
  0.1 lot risk
  0.01 lot risk
  minimum achievable risk

Classification:
  TRADEABLE    — min risk <= $62.50 budget
  MARGINAL     — min risk <= $125 budget (2x)
  UNTRADEABLE  — even 0.01 lot exceeds $125

Output:
  research/risk_reality/reports/ASSET_RISK_AUDIT.md
"""

import json
import os
import sys
import math
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from proxima_ops.config.settings import SETTINGS
from proxima_ops.risk.catastrophic_stop import catastrophic_sl, CATASTROPHIC_STOP_PIPS

try:
    import MetaTrader5 as mt5
    MT5_OK = True
except ImportError:
    MT5_OK = False


SYMBOLS = ["EURJPY", "USDJPY", "GBPJPY", "XAUUSD", "EURUSD"]
RISK_BUDGET = 62.50  # 0.25% of $25,000


def get_pip_value(symbol: str, price: float) -> float:
    """Return pip value per 1 standard lot (100,000 units)."""
    if "JPY" in symbol:
        return 100.0 / price if price > 0 else 1.0  # ~$0.62 per pip for USDJPY
    elif "XAU" in symbol or "XAG" in symbol:
        return 10.0  # $10 per pip per lot
    elif "NAS" in symbol:
        return 1.0   # $1 per point per lot
    else:
        return 10.0  # $10 per pip per lot


def get_point_value(symbol: str, price: float) -> float:
    """Return $ risk per point per lot (used by risk verifier)."""
    return 1.0  # risk verifier hardcodes this to 1.0 for ALL symbols


def stop_in_points(symbol: str, entry_price: float) -> tuple:
    """Return (stop_pips, stop_points) using catastrophic stop."""
    sl_price = catastrophic_sl(symbol, entry_price, "BUY")
    pip_dist = abs(entry_price - sl_price)
    if "JPY" in symbol:
        pips = pip_dist / 0.01
        points = pip_dist / 0.001  # 1 pip = 10 points
    elif "XAU" in symbol or "XAG" in symbol:
        pips = pip_dist / 0.01
        points = pip_dist / 0.01   # 1 pip = 1 point for gold
    elif "NAS" in symbol:
        pips = pip_dist / 0.01
        points = pip_dist / 0.01   # 1 pip = 1 point
    else:
        pips = pip_dist / 0.0001
        points = pip_dist / 0.0001
    return round(pips, 1), round(points, 1)


def dollar_risk_at_lot(symbol: str, entry_price: float, lots: float,
                       stop_points: float) -> float:
    """Dollar risk = lots * 1.0 (point_value) * stop_points (verifier formula)."""
    return lots * 1.0 * stop_points


def audit_asset(symbol: str, entry_price: float) -> dict:
    pips, points = stop_in_points(symbol, entry_price)
    stop_pips = CATASTROPHIC_STOP_PIPS.get(symbol, 50)

    result = {
        "symbol": symbol,
        "entry_price": entry_price,
        "stop_pips_config": stop_pips,
        "stop_pips_actual": pips,
        "stop_points_actual": points,
    }

    for lots in [1.0, 0.1, 0.01]:
        risk_dollars = dollar_risk_at_lot(symbol, entry_price, lots, points)
        result[f"risk_at_{lots:.2f}_lot"] = round(risk_dollars, 2)

    # Minimum achievable risk (0.01 lot)
    min_risk = dollar_risk_at_lot(symbol, entry_price, 0.01, points)
    result["minimum_achievable_risk"] = round(min_risk, 2)

    # Classification
    if min_risk <= RISK_BUDGET:
        result["classification"] = "TRADEABLE"
    elif min_risk <= RISK_BUDGET * 2:
        result["classification"] = "MARGINAL"
    else:
        result["classification"] = "UNTRADEABLE"

    # Can trade?
    result["can_trade_under_62_50"] = min_risk <= RISK_BUDGET

    return result


def get_current_prices() -> dict:
    """Fetch current prices from MT5."""
    prices = {
        "EURJPY": 185.0, "USDJPY": 160.0, "GBPJPY": 214.0,
        "XAUUSD": 4325.0, "EURUSD": 1.15, "NAS100": 19500.0,
    }
    if not MT5_OK:
        return prices
    try:
        if not mt5.initialize():
            return prices
        for sym in SYMBOLS:
            try:
                tick = mt5.symbol_info_tick(sym)
                if tick:
                    prices[sym] = (tick.ask + tick.bid) / 2
            except Exception:
                pass
        mt5.shutdown()
    except Exception:
        pass
    return prices


def main():
    output_dir = os.path.join(os.path.dirname(__file__), "reports")
    os.makedirs(output_dir, exist_ok=True)

    prices = get_current_prices()
    results = []
    for sym in SYMBOLS:
        price = prices.get(sym, 1.0)
        r = audit_asset(sym, price)
        results.append(r)

    # Generate report
    lines = []
    lines.append("# Asset-Specific Risk Audit")
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Risk Budget:** ${RISK_BUDGET:.2f} (0.25% of $25,000)")
    lines.append("")
    lines.append("## Per-Asset Results")
    lines.append("")
    lines.append("| Asset | Price | Stop (cfg) | Stop (pips) | 1 Lot | 0.1 Lot | 0.01 Lot | Min Risk | $62.50 | Classification |")
    lines.append("|-------|-------|------------|-------------|-------|---------|----------|----------|--------|----------------|")
    for r in results:
        lines.append(
            f"| {r['symbol']:<6} "
            f"| ${r['entry_price']:<8.2f} "
            f"| {r['stop_pips_config']:<10} "
            f"| {r['stop_pips_actual']:<11} "
            f"| ${r['risk_at_1.00_lot']:<5.2f} "
            f"| ${r['risk_at_0.10_lot']:<7.2f} "
            f"| ${r['risk_at_0.01_lot']:<8.2f} "
            f"| ${r['minimum_achievable_risk']:<8.2f} "
            f"| {'YES' if r['can_trade_under_62_50'] else 'NO':<6} "
            f"| {r['classification']:<14} |"
        )
    lines.append("")

    lines.append("## Analysis")
    lines.append("")
    lines.append("The risk verifier uses `point_value_per_lot = 1.0` for ALL symbols.")
    lines.append("This means dollar risk = lots × 1.0 × stop_points.")
    lines.append("")
    lines.append("For JPY pairs: 1 pip = 10 points (stop is divided by 0.001, not 0.01)")
    lines.append("For XAU: 1 pip = 1 point (stop is divided by 0.01)")
    lines.append("For FX: 1 pip = 10 points (stop is divided by 0.0001)")
    lines.append("")
    non_tradeable = [r for r in results if not r["can_trade_under_62_50"]]
    if non_tradeable:
        lines.append("### Untradeable Assets Under $62.50 Budget")
        lines.append("")
        for r in non_tradeable:
            lines.append(f"- **{r['symbol']}**: Min risk ${r['minimum_achievable_risk']:.2f} at 0.01 lot with {r['stop_pips_actual']} pip stop")
        lines.append("")
        lines.append("These assets CANNOT be traded under current catastrophic stop distances.")
    else:
        lines.append("### All assets are tradeable under $62.50")

    report = "\n".join(lines)
    report_path = os.path.join(output_dir, "ASSET_RISK_AUDIT.md")
    with open(report_path, "w") as f:
        f.write(report)
    print(report)
    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()
