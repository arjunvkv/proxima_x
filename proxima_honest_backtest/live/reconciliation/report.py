"""Sign-off / validation report generator.

Combines a ReconMonitor verdict plus an optional Level-2 island diff into the
8-section demo_signoff_report structure frozen in IT3/IT7. The honest header
clause is always embedded so nobody overreads an offline island result.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from proxima_honest_backtest.live.reconciliation.monitor import ReconMonitor

HONEST_HEADER = (
    "This validation proves the live execution infrastructure correctly preserves "
    "the validated strategy decision path and reconciles execution state with "
    "broker reality. The offline island validates event flow, state transitions, "
    "reconciliation logic, and execution plumbing using deterministic simulated "
    "execution. It does not prove historical liquidity, order-book behavior, "
    "market impact, partial-fill behavior, or profitability. True execution-price, "
    "latency, spread, and broker behavior validation requires the live FTMO demo phase."
)


def generate_signoff(
    run_id: str,
    events: List[Dict[str, Any]],
    env: str = "offline",
    strategy: str = "tokyo_h0",
    gates: Optional[Dict[str, Any]] = None,
    level2: Optional[Dict[str, Any]] = None,
    out_path: Optional[str] = None,
) -> Dict[str, Any]:
    mon = ReconMonitor(gates or {}).evaluate(events)

    report = {
        "run_id": run_id,
        "honest_scope": HONEST_HEADER,
        "environment": env,
        "strategy": strategy,
        "sections": {
            "decision_integrity": _decision_metrics(events),
            "execution_integrity": _execution_metrics(events),
            "position_reconciliation": _position_metrics(events),
            "clock": _clock_metrics(events),
        },
        "verdict": mon["verdict"],
        "critical_fails": mon["critical_fails"],
        "metrics": mon["metrics"],
    }
    if level2 is not None:
        report["level2_diff"] = level2
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
    return report


def _by(events, et):
    return [e for e in events if e.get("event_type") == et]


def _decision_metrics(events):
    dec = _by(events, "DECISION")
    return {
        "total": len(dec),
        "enters": sum(1 for e in dec if e.get("action") == "ENTER"),
        "exits": sum(1 for e in dec if e.get("action") == "EXIT"),
    }


def _execution_metrics(events):
    fills = _by(events, "BROKER_FILL")
    rej = _by(events, "BROKER_REJECT")
    slips = [e.get("slippage_pips") for e in fills if e.get("slippage_pips") is not None]
    return {
        "orders_sent": len(_by(events, "ORDER_SENT")),
        "fills": len(fills),
        "rejects": len(rej),
        "avg_slippage_pips": round(sum(slips) / len(slips), 4) if slips else 0.0,
    }


def _position_metrics(events):
    syncs = _by(events, "POSITION_SYNC")
    ok = sum(1 for e in syncs if e.get("reconciliation_status") == "PASS")
    return {"syncs": len(syncs), "pass": ok, "fail": len(syncs) - ok}


def _clock_metrics(events):
    ticks = _by(events, "TICK")
    return {"tick_events": len(ticks)}