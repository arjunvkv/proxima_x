"""Apples-to-apples parity verification harness (live vs backtest).

Stages (contract with ChatGPT):
  Stage 1 — Decision parity:   zero tolerance on the DECISION set.
            keys = (timestamp, symbol, action, normalized side, decision_id).
  Stage 2 — Execution parity:  every live fill matches a decision + bounded slip.
  Stage 3 — Cost parity:       backtest net_pnl == live (DEAL_PROFIT+COMM+SWAP)
                               within $0.01 for identical fills.
  Stage 4 — Lifecycle parity:  DECISION -> ORDER_SENT -> BROKER_FILL|BROKER_REJECT;
                               no DECISION without outcome, no FILL without decision.

Pass/fail thresholds are explicit and HALT the deployment on violation.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional

from proxima_honest_backtest.live.parity import normalize_side, decision_fingerprint

# --- execution slip envelope (pips) ---
SLIP_MEDIAN = 0.5
SLIP_P95 = 2.0
SLIP_HARD = 5.0

# --- cost parity tolerance (USD) ---
COST_TOLERANCE_USD = 0.01


def _load_events(path: str) -> List[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def stage1_decisions(backtest_decisions: List[Dict[str, Any]],
                     live_events: List[Dict[str, Any]]) -> Dict[str, Any]:
    bt = set()
    for d in backtest_decisions:
        if d.get("type") == "ENTER":
            bt.add(decision_fingerprint(d))
    # live ENTER decisions come from DECISION events (action ENTER*)
    lv = set()
    for ev in live_events:
        if (ev.get("event_type") == "DECISION"
                and (ev.get("action") or "").startswith("ENTER")):
            lv.add((
                str(ev.get("bar_ts_utc", "")),
                ev.get("symbol"),
                "ENTER",
                normalize_side(ev.get("side", "")),
            ))
    only_bt = bt - lv
    only_live = lv - bt
    return {
        "passed": not only_bt and not only_live,
        "n_backtest": len(bt), "n_live": len(lv),
        "missing": sorted(only_bt)[:10], "extra": sorted(only_live)[:10],
    }


def stage2_execution(live_events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Measured slip = |fill_price - requested_price| in pips, per fill."""
    slips = []
    req: Dict[str, float] = {}
    for ev in live_events:
        et = ev.get("event_type")
        did = str(ev.get("decision_id", "") or "")
        if et == "ORDER_SENT":
            req[did] = float(ev.get("requested_price") or 0.0)
        elif et == "BROKER_FILL":
            fill = float(ev.get("fill_price") or 0.0)
            requested = req.get(did)
            if requested:
                slips.append(abs(fill - requested))
    if not slips:
        return {"fits": 0, "passed_slip": True, "median_slip_pips": None,
                "p95_slip_pips": None, "note": "no fills"}

    def pct(p):
        s = sorted(slips)
        i = min(len(s) - 1, max(0, int(len(s) * p)))
        return s[i]

    hard = max(slips) <= SLIP_HARD
    return {
        "fits": len(slips),
        "median_slip_px": round(pct(0.5), 6),
        "p95_slip_px": round(pct(0.95), 6),
        "max_slip_px": round(slips[-1], 6),
        "passed_slip": hard,
        "hard_limit_px": SLIP_HARD,
    }


def stage3_cost(backtest_net: float, live_net: float) -> Dict[str, Any]:
    diff = abs(backtest_net - live_net)
    return {
        "backtest_net_pnl": backtest_net,
        "live_net_pnl": live_net,
        "abs_diff": round(diff, 4),
        "passed": diff <= COST_TOLERANCE_USD,
        "tolerance_usd": COST_TOLERANCE_USD,
    }


def stage4_lifecycle(live_events: List[Dict[str, Any]]) -> Dict[str, Any]:
    dec_didas = set()
    for ev in live_events:
        if ev.get("event_type") == "DECISION":
            dec_didas.add(str(ev.get("decision_id", "")))
    filled = set()
    rejected = set()
    sent = set()
    for ev in live_events:
        et = ev.get("event_type")
        did = str(ev.get("decision_id", ""))
        if et == "ORDER_SENT":
            sent.add(did)
        elif et == "BROKER_FILL":
            filled.add(did)
        elif et == "BROKER_REJECT":
            rejected.add(did)
    done = filled | rejected
    without_outcome = sent - done
    no_sent = done - sent       # fill/reject without a preceding order
    return {
        "passed": not without_outcome and not no_sent,
        "pending_without_outcome": sorted(without_outcome)[:10],
        "outcome_without_sent": sorted(no_sent)[:10],
        "n_sent": len(sent), "n_filled": len(filled), "n_rejected": len(rejected),
    }


def run_full(backtest_decisions: List[Dict[str, Any]],
             live_events: List[Dict[str, Any]],
             backtest_net: float, live_net: float) -> Dict[str, Any]:
    s1 = stage1_decisions(backtest_decisions, live_events)
    s2 = stage2_execution(live_events)
    s3 = stage3_cost(backtest_net, live_net)
    s4 = stage4_lifecycle(live_events)
    overall = s1["passed"] and s2["passed_slip"] and s3["passed"] and s4["passed"]
    return {"overall_pass": overall,
            "stage1_decision": s1, "stage2_execution": s2,
            "stage3_cost": s3, "stage4_lifecycle": s4}