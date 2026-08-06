"""Replay parity — prove LiveRunner decisions == backtest decisions.

Level 1 (decision parity): timestamp + symbol + action + normalized side.
Entry PRICE is explicitly excluded (live fill latency/spread belongs to Level 2
execution parity).

Ship rule: a strategy is live-ready only if it implements the same interface
path that passed replay parity.
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List


def normalize_side(side: str) -> str:
    up = str(side).upper()
    if up in ("LONG", "BUY", "L"):
        return "L"
    if up in ("SHORT", "SELL", "S"):
        return "S"
    return up


def decision_fingerprint(decision: Dict[str, Any]) -> tuple:
    return (
        str(decision.get("ts")),
        decision.get("symbol"),
        decision.get("type"),
        normalize_side(decision.get("side")),
    )


def decision_id(strategy: str, decision: Dict[str, Any]) -> str:
    """Correlation key across backtest replay / paper / live — NO price in id."""
    payload = "|".join([
        strategy,
        str(decision.get("ts")),
        str(decision.get("symbol")),
        str(decision.get("type")),
        normalize_side(decision.get("side")),
    ])
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def fingerprint_enters(decisions: List[Dict[str, Any]]) -> List[tuple]:
    return sorted(
        decision_fingerprint(d)
        for d in decisions if d.get("type") == "ENTER"
    )


def compare_level1(backtest_decisions: List[Dict[str, Any]],
                   live_decisions: List[Dict[str, Any]]) -> Dict[str, Any]:
    b = fingerprint_enters(backtest_decisions)
    l = fingerprint_enters(live_decisions)
    only_backtest = [x for x in b if x not in l]
    only_live = [x for x in l if x not in b]
    return {
        "passed": not only_backtest and not only_live,
        "n_backtest": len(b),
        "n_live": len(l),
        "only_backtest": only_backtest[:10],
        "only_live": only_live[:10],
    }


def extract_backtest_enters(trades: List[Any]) -> List[Dict[str, Any]]:
    """Convert Trade objects to Level-1 decision dicts (ENTER only).

    ENTER trades in the backtest carry no pnl/commission (they are fills);
    EXIT trades carry pnl/commission. Used only to build backtest fingerprints.
    """
    out = []
    for t in trades:
        is_enter = (t.pnl == 0.0 and t.commission == 0.0)
        out.append({
            "ts": str(t.timestamp),
            "symbol": t.symbol,
            "side": t.side,
            "type": "ENTER" if is_enter else "EXIT",
        })
    return out


def extract_decision_enters(decisions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Level-1 ENTER fingerprints from strategy DECISIONS (pre-fill).

    Live/backtest both run the identical strategy through the DecisionKernel,
    so decision parity must compare the strategy's emitted signals — NOT the
    (simulator-dependent) filled trades. Fill acceptance is Level-2 territory.
    """
    return [
        {"ts": str(d.get("ts")), "symbol": d.get("symbol"),
         "side": d.get("side"), "type": "ENTER"}
        for d in decisions if d.get("type") == "ENTER"
    ]