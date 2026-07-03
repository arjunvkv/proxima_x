"""V2.4B Phase 1 — Trace all stop-distance assumptions across the pipeline."""
import sys
sys.path.insert(0, r"C:\Trading\Agentic_Trading\proxima_x")

from unittest.mock import MagicMock
from proxima_ops.config.settings import SETTINGS
from proxima_ops.risk.catastrophic_stop import CATASTROPHIC_STOP_PIPS, catastrophic_sl
from proxima_ops.execution.order_manager import OrderManager
from proxima_ops.risk.trade_risk_verifier import TradeRiskVerifier

BALANCE = 25000.0
RISK_PCT = 0.0025

SYMBOLS = {"EURUSD": 1.08, "EURJPY": 140.0, "USDJPY": 140.0, "GBPJPY": 165.0, "XAUUSD": 2000.0}

print("=" * 90)
print("  V2.4B — Stop Distance Alignment Audit")
print("=" * 90)
print()
print(f"{'Symbol':>8} {'max_spread':>11} {'cat_pips':>9} {'sizing_stop':>12} {'verifier_stop':>14} {'diff':>8} {'verifier_risk':>14} {'budget':>8} {'result':>8}")
print("-" * 90)

for sym, px in SYMBOLS.items():
    ms = SETTINGS.max_spread_points.get(sym, 50)
    cp = CATASTROPHIC_STOP_PIPS.get(sym, 50)

    # compute what calculate_volume() does
    pv = 1000.0 / max(px, 1.0) if "JPY" in sym else 10.0
    sizing_vol = round(BALANCE * RISK_PCT / max(ms * pv, 1.0), 2)
    sizing_vol = max(0.01, sizing_vol)

    # compute what verifier does with catastrophic sl_price
    sl_price = catastrophic_sl(sym, px, "BUY")
    pip_size = 0.01 if "JPY" in sym else 0.0001
    if sym in ("XAUUSD",):
        pip_size = 0.01
    pip_dist = abs(px - sl_price)
    stop_points = int(pip_dist / pip_size)
    vpv = 1000.0 / max(px, 1.0) if "JPY" in sym else 10.0
    verifier_risk = sizing_vol * vpv * stop_points

    diff = stop_points - ms
    budget = BALANCE * RISK_PCT
    result = "OK" if verifier_risk <= budget * 1.05 else "REJECT"

    print(f"{sym:>8} {ms:>11} {cp:>9} {ms:>12} {stop_points:>14} {diff:>8} {verifier_risk:>14.2f} {budget:>8.2f} {result:>8}")

print("-" * 90)
print()
print("=== DETAILED STOP SOURCE TRACE ===")
print()

for sym, px in SYMBOLS.items():
    ms = SETTINGS.max_spread_points.get(sym, 50)
    cp = CATASTROPHIC_STOP_PIPS.get(sym, 50)
    sl_price = catastrophic_sl(sym, px, "BUY")
    pip_size = 0.01 if ("JPY" in sym or "XAU" in sym) else 0.0001
    pip_dist = abs(px - sl_price)
    stop_points = int(pip_dist / pip_size)
    pv = 1000.0 / max(px, 1.0) if "JPY" in sym else 10.0
    vol = round(BALANCE * RISK_PCT / max(ms * pv, 1.0), 2)
    sizing_risk = vol * pv * ms
    verifier_risk = vol * pv * stop_points

    print(f"  {sym}:")
    print(f"    entry_price = {px}")
    print(f"    pip_value_per_lot = {pv:.4f}")
    print(f"    ----")
    print(f"    calculate_volume():")
    print(f"      assumed_sl_points = SETTINGS.max_spread_points[{sym}] = {ms}")
    print(f"      volume = budget / ({ms} * {pv:.4f}) = {BALANCE * RISK_PCT:.2f} / {ms * pv:.2f} = {vol}")
    print(f"      sizing risk at stop = {vol} * {pv:.4f} * {ms} = ${sizing_risk:.2f}")
    print(f"    ----")
    print(f"    catastrophic_sl():")
    print(f"      CATASTROPHIC_STOP_PIPS[{sym}] = {cp}")
    print(f"      pip_size = {pip_size}")
    print(f"      sl_price = {px} - {cp} * {pip_size} = {sl_price}")
    print(f"      pip_dist = |{px} - {sl_price}| = {pip_dist}")
    print(f"    ----")
    print(f"    TradeRiskVerifier:")
    print(f"      stop_points = {pip_dist} / {pip_size} = {stop_points}")
    print(f"      verifier risk = {vol} * {pv:.4f} * {stop_points} = ${verifier_risk:.2f}")
    print(f"      budget = ${BALANCE * RISK_PCT:.2f}")
    print(f"      verdict = {'PASS' if verifier_risk <= BALANCE * RISK_PCT * 1.05 else 'REJECT'}")
    print()

# Also trace the budget mismatch in risk_manager.py
print("=== BUDGET MISMATCH (risk_manager.py) ===")
print()
print("  calculate_volume() uses: risk_pct = SETTINGS.risk_per_trade * sizing_mult")
print("  RiskManager.pre_order_check() hardcodes: risk_budget = account_balance * 0.0025")
print()
print("  This means at sizing_mult = 0.50:")
print(f"    calculate_volume budget = ${BALANCE * RISK_PCT * 0.50:.2f}")
print(f"    verifier budget        = ${BALANCE * RISK_PCT:.2f}")
print(f"    difference             = ${BALANCE * RISK_PCT - BALANCE * RISK_PCT * 0.50:.2f}")
print()
print("  At sizing_mult = 1.00, budgets match (both use 0.0025)")
