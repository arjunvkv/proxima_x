"""
Phase 4 / V2.4 Risk Engine Repair — Failed Case Replay

Replays ALL symbols × 5 sizing multipliers through the fixed
OrderManager.calculate_volume() + TradeRiskVerifier.verify() pipeline.

Expected: ALL 25+ scenarios return accepted=True.
"""

import sys
sys.path.insert(0, r"C:\Trading\Agentic_Trading\proxima_x")

from unittest.mock import MagicMock
from proxima_ops.config.settings import SETTINGS
from proxima_ops.execution.order_manager import OrderManager
from proxima_ops.risk.trade_risk_verifier import TradeRiskVerifier
from proxima_ops.risk.catastrophic_stop import catastrophic_sl, CATASTROPHIC_STOP_PIPS

BALANCE = 25000.0
RISK_PCT = 0.0025

SYMBOLS = {
    "EURJPY": 140.0,
    "USDJPY": 140.0,
    "GBPJPY": 165.0,
    "XAUUSD": 2000.0,
    "EURUSD": 1.05,
    "NAS100": 19500.0,
}

MULTIPLIERS = [0.10, 0.25, 0.50, 0.75, 1.00]


def run_replay():
    manager = OrderManager(mt5=MagicMock())
    verifier = TradeRiskVerifier()

    results = []
    total = len(SYMBOLS) * len(MULTIPLIERS)
    passed = 0

    for symbol, price in SYMBOLS.items():
        sl_price = catastrophic_sl(symbol, price, "BUY")
        for m in MULTIPLIERS:
            adj_risk_pct = RISK_PCT * m
            risk_budget = BALANCE * adj_risk_pct

            volume = manager.calculate_volume(
                symbol, price, BALANCE, risk_pct=adj_risk_pct
            )

            result = verifier.verify(
                symbol=symbol,
                volume=volume,
                entry_price=price,
                sl_price=sl_price,
                account_balance=BALANCE,
                risk_budget=risk_budget,
                order_type="BUY",
            )

            actual_risk = result.get("actual_dollar_risk", 0)
            accepted = result.get("accepted", False)
            if accepted:
                passed += 1

            results.append({
                "symbol": symbol,
                "price": price,
                "multiplier": m,
                "adj_risk_pct": adj_risk_pct,
                "risk_budget": risk_budget,
                "volume": volume,
                "cat_pips": CATASTROPHIC_STOP_PIPS.get(symbol, 50),
                "dollar_risk": actual_risk,
                "accepted": accepted,
                "reason": result.get("reason", ""),
            })

    return results, passed, total


def print_table(results):
    header = (
        f"{'Symbol':>8}  {'Mult':>5}  {'Risk$':>8}  {'Volume':>8}  "
        f"{'Dollar$':>9}  {'Budget$':>9}  {'Status':>8}"
    )
    sep = "=" * len(header)
    print()
    print("=" * 80)
    print("  FAILED CASE REPLAY — Phase 4 Risk Engine Repair")
    print("=" * 80)
    print(header)
    print(sep)
    for r in results:
        status = "PASS" if r["accepted"] else "REJECT"
        print(
            f"{r['symbol']:>8}  {r['multiplier']:>5.2f}  "
            f"{r['risk_budget']:>8.2f}  {r['volume']:>8.4f}  "
            f"{r['dollar_risk']:>9.2f}  {r['risk_budget']:>9.2f}  "
            f"{status:>8}"
        )
    print(sep)
    print()


def generate_markdown(results, passed, total):
    lines = []
    lines.append("# Failed Case Replay Report")
    lines.append("")
    lines.append("**Phase 4 — V2.4 Risk Engine Repair**")
    lines.append(f"**Date:** 2026-06-16")
    lines.append(f"**Script:** `research/risk_reality/replay_failed_cases.py`")
    lines.append("")
    lines.append("## Objective")
    lines.append("")
    lines.append("Replay ALL symbols × 5 sizing multipliers through the fixed")
    lines.append("`OrderManager.calculate_volume()` + `TradeRiskVerifier.verify()` pipeline.")
    lines.append("Every scenario is expected to return `accepted=True`.")
    lines.append("")
    lines.append("## Fixes Applied")
    lines.append("")
    lines.append("### 1. `proxima_ops/execution/order_manager.py` — `calculate_volume()`")
    lines.append("")
    lines.append("- `assumed_sl_points` now reads from `CATASTROPHIC_STOP_PIPS` instead of `max_spread_points`")
    lines.append("  so the volume calculation budgets for the actual maximum stop-loss distance")
    lines.append("- Added `NAS` branch: `point_value_per_lot = 0.5`")
    lines.append("- Added `XAU`/`XAG` branch: `point_value_per_lot = 1.0`")
    lines.append("")
    lines.append("### 2. `proxima_ops/risk/trade_risk_verifier.py` — `verify()`")
    lines.append("")
    lines.append("- Replaced hardcoded `point_value_per_lot = 1.0` with symbol-correct values")
    lines.append("- Fixed pip divisors: `0.01` for JPY/XAU/NAS, `0.0001` for others")
    lines.append("- Specific pip values per lot:")
    lines.append("  - JPY pairs: `1000.0 / entry_price`")
    lines.append("  - XAU/XAG: `1.0`")
    lines.append("  - NAS: `0.5`")
    lines.append("  - Others: `10.0`")
    lines.append("")
    lines.append("## Test Parameters")
    lines.append("")
    lines.append(f"| Parameter | Value |")
    lines.append(f"|-----------|-------|")
    lines.append(f"| Balance | ${BALANCE:,.0f} |")
    lines.append(f"| Base risk | {RISK_PCT*100:.2f}% |")
    lines.append(f"| Multipliers | {', '.join(f'{m:.2f}' for m in MULTIPLIERS)} |")
    lines.append(f"| Total scenarios | {total} |")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    col_header = (
        f"| {'Symbol':>8} | {'Mult':>5} | {'Risk$':>8} | {'Volume':>8} | "
        f"{'CatPips':>7} | {'Dollar$':>9} | {'Budget$':>9} | {'Status':>8} |"
    )
    lines.append(col_header)
    col_sep = (
        f"|{'-'*10}:|{'-'*7}:|{'-'*10}:|{'-'*10}:|"
        f"{'-'*9}:|{'-'*11}:|{'-'*11}:|{'-'*10}:|"
    )
    lines.append(col_sep)
    for r in results:
        status = "**PASS**" if r["accepted"] else "**REJECT**"
        lines.append(
            f"| {r['symbol']:>8} | {r['multiplier']:>5.2f} | "
            f"{r['risk_budget']:>8.2f} | {r['volume']:>8.4f} | "
            f"{r['cat_pips']:>7} | {r['dollar_risk']:>9.2f} | "
            f"{r['risk_budget']:>9.2f} | {status:>8} |"
        )
    lines.append("")
    verdict = "ALL SCENARIOS PASS" if passed == total else f"{total - passed} SCENARIOS FAIL"
    lines.append(f"## Verdict: {verdict}")
    lines.append("")
    lines.append(f"**{passed} / {total} scenarios accepted.**")
    lines.append("")
    if passed == total:
        lines.append("The pipeline now correctly budgets risk against the actual catastrophic stop-loss distance")
        lines.append("for every symbol, and the verifier computes dollar risk using the same pip-value")
        lines.append("convention as `calculate_volume()`. All trades pass verification.")
    else:
        lines.append("Remaining failures indicate a gap between the assumed stop distance and the")
        lines.append("catastrophic stop, or a pip-value mismatch. Review settings for failing symbols.")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    results, passed, total = run_replay()
    print_table(results)
    print(f"  Result: {passed} / {total} scenarios passed")
    print()

    md = generate_markdown(results, passed, total)
    report_path = r"C:\Trading\Agentic_Trading\proxima_x\research\risk_reality\reports\FAILED_CASE_REPLAY.md"
    with open(report_path, "w") as f:
        f.write(md)
    print(f"  Report saved -> {report_path}")
