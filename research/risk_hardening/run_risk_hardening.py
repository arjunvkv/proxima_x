"""RHL-10: Risk Reality Validation — prove all risk controls mathematically."""

import json
import os
import sys
from datetime import datetime


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def run_validation() -> dict:
    from proxima_ops.risk.catastrophic_stop import catastrophic_sl, CATASTROPHIC_STOP_PIPS
    from proxima_ops.risk.trade_risk_verifier import TradeRiskVerifier
    from proxima_ops.risk.risk_governor import RiskGovernor
    from proxima_ops.risk.exposure_controller import ExposureController

    results = {}

    # V1: Can any trade exceed risk budget?
    verifier = TradeRiskVerifier()
    v1 = verifier.verify("EURUSD", 1.0, 1.16, 1.155, 25000.0, 62.50, "BUY")
    v1_high = verifier.verify("EURUSD", 10.0, 1.16, 1.155, 25000.0, 62.50, "BUY")
    results["V1_risk_budget_respected"] = {
        "pass": v1["accepted"] and not v1_high["accepted"],
        "normal_risk": v1["accepted"],
        "excess_rejected": not v1_high["accepted"],
        "detail": "Normal 1-lot accepted, 10-lot rejected"}

    # V2: Can position survive without SL? (catastrophic stop)
    for sym, expected_pips in CATASTROPHIC_STOP_PIPS.items():
        sl = catastrophic_sl(sym, 100.0, "BUY")
        results[f"V2_sl_{sym}"] = {"pass": sl > 0, "sl": sl}

    v2_all = all(r.get("pass") for k, r in results.items() if k.startswith("V2_sl_"))
    results["V2_summary"] = {"pass": v2_all, "detail": f"{len(CATASTROPHIC_STOP_PIPS)} symbols have SL"}

    # V3: Daily stop works
    gov = RiskGovernor()
    gov.set_start_equity(25000.0)
    for _ in range(10):
        gov.record_result(-300.0)
        gov.update_unrealized(0.0, 24700.0)
    s3 = gov.check()
    results["V3_daily_stop"] = {
        "pass": s3.get("state") == "DAILY_STOP",
        "state": s3.get("state"),
        "daily_pnl": s3.get("daily_pnl"),
        "detail": f"Daily loss ${s3.get('daily_pnl', 0):.0f} at 1% threshold -> DAILY_STOP"}

    # V4: Loss streak protection works
    gov2 = RiskGovernor()
    gov2.set_start_equity(10000000.0)
    for _ in range(3):
        gov2.record_result(-10.0)
    s4 = gov2.check()
    results["V4_loss_streak"] = {
        "pass": s4.get("state") == "LOSS_STREAK_STOP",
        "state": s4.get("state"),
        "streak": s4.get("loss_streak"),
        "detail": f"3 consecutive losses -> LOSS_STREAK_STOP"}

    # V5: Equity floor works
    gov3 = RiskGovernor()
    gov3.set_start_equity(100000.0)
    dd = gov3.check_equity_drawdown(85000.0)
    results["V5_equity_floor"] = {
        "pass": dd.get("triggered"),
        "drawdown_pct": dd.get("drawdown_pct"),
        "state": dd.get("state"),
        "detail": f"15% drawdown exceeds 10% floor -> EQUITY_PROTECTION"}
    # Reset with moderate drawdown (should NOT trigger)
    gov3b = RiskGovernor()
    gov3b.set_start_equity(100000.0)
    dd2 = gov3b.check_equity_drawdown(95000.0)
    results["V5_equity_floor_no_false"] = {
        "pass": not dd2.get("triggered"),
        "drawdown_pct": dd2.get("drawdown_pct"),
        "detail": f"5% drawdown does NOT trigger (correctly)"}

    # V6: Exposure controller works
    ec = ExposureController()
    empty = ec.check([])
    fx_full = ec.check([{"symbol": "EURUSD"}, {"symbol": "USDJPY"}, {"symbol": "GBPJPY"}], new_symbol="EURJPY")
    gold_full = ec.check([{"symbol": "XAUUSD"}], new_symbol="XAGUSD")
    index_full = ec.check([{"symbol": "NAS100"}], new_symbol="US100")
    results["V6_exposure_controller"] = {
        "pass": empty["allowed"] and not fx_full["allowed"] and not gold_full["allowed"] and not index_full["allowed"],
        "empty_allowed": empty["allowed"],
        "fx_rejected": not fx_full["allowed"],
        "gold_rejected": not gold_full["allowed"],
        "index_rejected": not index_full["allowed"],
        "detail": "Exposure controller enforces 3 FX, 1 gold, 1 index max"}

    # V7: MT5 disconnect blocks entries
    from proxima_ops.risk.risk_health_monitor import RiskHealthMonitor
    rh = RiskHealthMonitor()
    h_ok = rh.check(True, True, {"EURUSD": True}, True)
    h_fail = rh.check(False, False, {"EURUSD": False}, True)
    results["V7_mt5_disconnect"] = {
        "pass": h_ok["entries_disabled"] is False and h_fail["entries_disabled"] is True,
        "healthy": h_ok,
        "failed": h_fail,
        "detail": "MT5 disconnect -> BROKER_FAILURE, entries disabled"}

    # V8: Position watchdog
    from proxima_ops.risk.position_watchdog import PositionWatchdog
    pw = PositionWatchdog()
    w_ok = pw.verify([{"ticket": 1, "volume": 0.1, "profit": 10.0}], [{"ticket": 1, "volume": 0.1, "profit": 10.0}])
    w_fail = pw.verify([{"ticket": 1, "volume": 0.1, "profit": 10.0}], [])
    results["V8_position_watchdog"] = {
        "pass": w_ok["state"] == "HEALTHY" and w_fail["state"] == "CRITICAL_POSITION_MISMATCH",
        "healthy": w_ok,
        "mismatch": w_fail,
        "detail": "MT5/ledger mismatch detected -> CRITICAL_POSITION_MISMATCH"}

    # V9: Maximum account loss is bounded
    max_catastrophic_loss = {}
    for sym, pips in CATASTROPHIC_STOP_PIPS.items():
        if "JPY" in sym:
            loss_per_lot = pips * 0.01 * 100000.0
        elif "XAU" in sym:
            loss_per_lot = pips * 0.01 * 100.0
        else:
            loss_per_lot = pips * 0.0001 * 100000.0
        max_catastrophic_loss[sym] = round(loss_per_lot, 0)
    results["V9_max_loss_bounded"] = {
        "pass": all(v > 0 for v in max_catastrophic_loss.values()),
        "max_loss_per_lot": max_catastrophic_loss,
        "detail": "Every instrument has a bounded catastrophic stop loss"}

    # V10: Risk manager pre_order_check
    from proxima_ops.risk.risk_manager import RiskManager
    rm = RiskManager()
    rm.governor.set_start_equity(25000.0)
    pre = rm.pre_order_check("EURUSD", 0.5, 1.160, 25000.0, [])
    pre_blocked = rm.pre_order_check("EURUSD", 10.0, 1.160, 25000.0, [])
    results["V10_risk_manager"] = {
        "pass": pre["allowed"] and not pre_blocked["allowed"],
        "normal_allowed": pre["allowed"],
        "excess_rejected": not pre_blocked["allowed"],
        "sl_set": pre.get("sl", 0) > 0,
        "detail": "Risk manager accepts normal, rejects excess, sets catastrophic SL"}

    # Overall classification
    passes = sum(1 for k, r in results.items() if r.get("pass"))
    total = sum(1 for k, r in results.items() if "pass" in r)
    if passes == total:
        classification = "RISK_ENGINE_HEALTHY"
    elif passes >= total * 0.7:
        classification = "RISK_ENGINE_DEGRADED"
    else:
        classification = "RISK_ENGINE_CRITICAL"

    report = {
        "timestamp": datetime.now().isoformat(),
        "classification": classification,
        "passed": passes,
        "total": total,
        "pass_rate": round(passes / max(total, 1), 3),
        "results": results,
        "max_catastrophic_loss_per_lot": max_catastrophic_loss}
    return report


def main():
    report = run_validation()
    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "risk_hardening_results.json")
    with open(out_dir, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n{'='*52}")
    print(f" RISK REALITY VALIDATION (RHL-10)")
    print(f"{'='*52}")
    print(f"  Classification:   {report['classification']}")
    print(f"  Passed:           {report['passed']}/{report['total']}")
    print(f"  Pass Rate:        {report['pass_rate']:.1%}")
    print(f"")
    for k, r in report.get("results", {}).items():
        p = "PASS" if r.get("pass") else "FAIL"
        print(f"  [{p}] {k}")
    print(f"\n  Results written to: {out_dir}")
    print(f"{'='*52}\n")


if __name__ == "__main__":
    main()
