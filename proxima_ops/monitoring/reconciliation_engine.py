import logging
import time
from typing import Optional

from proxima_x.proxima_ops.execution.execution_ledger import ExecutionLedger, TradeEvent

SEGL_STATES = {"OBSERVE": 0, "ARMED": 1, "EXECUTING": 2, "COOLDOWN": 3, "LOCKED": 4}

logger = logging.getLogger("proxima_ops.reconciliation")


class ReconciliationEngine:
    def __init__(self, ledger: ExecutionLedger):
        self.ledger = ledger
        self.history = []

    def _get_ticket(self, pos):
        return pos.get("ticket") if isinstance(pos, dict) else pos.ticket

    def _get_symbol(self, pos):
        return pos.get("symbol") if isinstance(pos, dict) else pos.symbol

    def _get_volume(self, pos):
        return pos.get("volume") if isinstance(pos, dict) else pos.volume

    def reconcile(self, mt5_positions: list, governor_state: str,
                  lifecycle_state: dict = None) -> dict:
        ledger_open = self.ledger.get_open_trades()
        mt5_by_ticket = {self._get_ticket(p): p for p in mt5_positions}
        ledger_by_ticket = {e.mt5_ticket: e for e in ledger_open if e.mt5_ticket}

        mt5_tickets = set(mt5_by_ticket.keys())
        ledger_tickets = set(ledger_by_ticket.keys())

        orphan_mt5 = []
        for t in sorted(mt5_tickets - ledger_tickets):
            pos = mt5_by_ticket[t]
            orphan_mt5.append({"ticket": t, "symbol": self._get_symbol(pos),
                               "volume": self._get_volume(pos)})

        orphan_ledger = []
        for t in sorted(ledger_tickets - mt5_tickets):
            ev = ledger_by_ticket[t]
            orphan_ledger.append({"ticket": t, "symbol": ev.symbol,
                                  "volume": ev.volume})

        matched_tickets = mt5_tickets & ledger_tickets
        volume_mismatches = []
        for t in sorted(matched_tickets):
            pos = mt5_by_ticket[t]
            ev = ledger_by_ticket[t]
            mt5_vol = self._get_volume(pos)
            ledger_vol = ev.volume
            if abs(mt5_vol - ledger_vol) > 0.001:
                volume_mismatches.append({"ticket": t, "mt5_volume": mt5_vol,
                                          "ledger_volume": ledger_vol})

        state_key = governor_state.upper()
        state_match = state_key in SEGL_STATES
        governor_count = SEGL_STATES.get(state_key, -1)

        lifecycle_issues = []
        if lifecycle_state:
            lc_result = self.verify_lifecycle(lifecycle_state)
            lifecycle_issues = lc_result.get("issues", [])

        match = (len(orphan_mt5) == 0 and len(orphan_ledger) == 0
                 and len(volume_mismatches) == 0 and state_match
                 and len(lifecycle_issues) == 0)

        report = {
            "match": match,
            "timestamp": time.time(),
            "mt5_count": len(mt5_positions),
            "ledger_count": len(ledger_open),
            "governor_count": governor_count,
            "orphan_mt5": orphan_mt5,
            "orphan_ledger": orphan_ledger,
            "volume_mismatches": volume_mismatches,
            "state_match": state_match,
            "governor_state": governor_state,
            "lifecycle_issues": lifecycle_issues,
            "drift_score": self.compute_drift_score(),
        }

        self.history.append(report)
        if len(self.history) > 1000:
            self.history = self.history[-1000:]

        return report

    def detect_orphans(self, mt5_positions: list) -> list:
        ledger_open = self.ledger.get_open_trades()
        ledger_tickets = {e.mt5_ticket for e in ledger_open if e.mt5_ticket}

        orphans = []
        for p in mt5_positions:
            t = self._get_ticket(p)
            if t and t not in ledger_tickets:
                orphans.append({"ticket": t, "symbol": self._get_symbol(p),
                                "volume": self._get_volume(p)})
        return orphans

    def compute_drift_score(self) -> float:
        if len(self.history) < 2:
            return 0.0

        recent = self.history[-20:]

        position_drifts = []
        for r in recent:
            diff = abs(r["mt5_count"] - r["ledger_count"])
            max_c = max(r["mt5_count"], r["ledger_count"], 1)
            position_drifts.append(diff / max_c)
        avg_position_drift = sum(position_drifts) / len(position_drifts)

        state_failures = sum(1 for r in recent if not r["state_match"])
        state_drift = state_failures / len(recent)

        timing_gaps = []
        for i in range(1, len(recent)):
            gap = abs(recent[i]["timestamp"] - recent[i - 1]["timestamp"])
            timing_gaps.append(min(gap / 60.0, 1.0))
        timing_drift = sum(timing_gaps) / len(timing_gaps) if timing_gaps else 0.0

        drift = (avg_position_drift * 0.4 + state_drift * 0.35 + timing_drift * 0.25)
        return min(max(drift, 0.0), 1.0)

    def verify_lifecycle(self, lifecycle_state: dict) -> dict:
        issues = []
        events = lifecycle_state.get("events", [])
        if not events:
            events = self.ledger.get_all()

        seen_ids = set()
        prev_ts = 0.0
        open_tickets = {}

        for ev in events:
            if isinstance(ev, dict):
                eid = ev.get("event_id", "")
                ts = ev.get("timestamp", 0)
                etype = ev.get("event_type", "")
                ticket = ev.get("mt5_ticket", 0)
                eid = ev.get("event_id", "")
            else:
                eid = ev.event_id
                ts = ev.timestamp
                etype = ev.event_type
                ticket = ev.mt5_ticket

            if eid in seen_ids:
                issues.append(f"Duplicate event: {eid}")
            seen_ids.add(eid)

            if ts < prev_ts:
                issues.append(f"Out of order: {eid} timestamp {ts} < {prev_ts}")
            prev_ts = max(prev_ts, ts)

            if etype == "trade_opened" and ticket:
                if ticket in open_tickets:
                    issues.append(f"Double open for ticket {ticket} at {eid}")
                open_tickets[ticket] = eid

            if etype == "trade_closed" and ticket:
                if ticket not in open_tickets:
                    issues.append(f"Close without open: ticket {ticket} at {eid}")
                else:
                    del open_tickets[ticket]

        all_events = self.ledger.get_all()
        ledger_event_ids = {e.event_id for e in all_events if e.event_id}
        given_ids = set(seen_ids)
        missing_in_given = ledger_event_ids - given_ids
        if missing_in_given:
            issues.append(f"Missing events from lifecycle_state: {sorted(missing_in_given)[:5]}")

        return {"pass": len(issues) == 0, "issues": issues,
                "events_checked": len(events), "open_remaining": len(open_tickets)}

    def full_audit(self, mt5_positions: list, governor_state: str,
                   lifecycle_state: dict = None, intent_result: dict = None) -> dict:
        report = self.reconcile(mt5_positions, governor_state, lifecycle_state)
        orphans = self.detect_orphans(mt5_positions)

        ledger_integrity = self.ledger.integrity_check()

        lc_audit = {}
        if lifecycle_state:
            lc_audit = self.verify_lifecycle(lifecycle_state)
        else:
            lc_audit = self.verify_lifecycle({"events": self.ledger.get_all()})

        all_events = self.ledger.get_all()
        open_trades = self.ledger.get_open_trades()
        ledger_stats = self.ledger.get_stats()

        return {
            "match": report["match"],
            "timestamp": report["timestamp"],
            "mt5_count": report["mt5_count"],
            "ledger_count": report["ledger_count"],
            "governor_count": report["governor_count"],
            "orphan_mt5": report["orphan_mt5"],
            "orphan_ledger": report["orphan_ledger"],
            "detected_orphans": orphans,
            "extra_mt5": len(report["orphan_mt5"]),
            "extra_ledger": len(report["orphan_ledger"]),
            "state_match": report["state_match"],
            "governor_state": governor_state,
            "ledger_integrity": ledger_integrity,
            "lifecycle": lc_audit,
            "volume_mismatches": report["volume_mismatches"],
            "drift_score": report["drift_score"],
            "total_events": len(all_events),
            "open_positions": len(open_trades),
            "ledger_stats": ledger_stats,
            "intent_consistency": self._audit_intent(intent_result) if intent_result else None,
        }

    def _audit_intent(self, intent_result: dict) -> dict:
        issues = []
        events = self.ledger.get_all()
        trades = [e for e in events if e.event_type in ("trade_opened", "trade_closed")]

        expected_trades = intent_result.get("expected_trades", 0)
        actual_trades = len(trades)
        if expected_trades and actual_trades != expected_trades:
            issues.append(f"Trade count mismatch: expected {expected_trades}, got {actual_trades}")

        intent_compliant = intent_result.get("compliant", True)
        actual_compliant = all(getattr(e, "intent_compliant", True) for e in trades)
        if intent_compliant != actual_compliant:
            issues.append(f"Intent compliance mismatch: expected {intent_compliant}, actual all={actual_compliant}")

        return {"pass": len(issues) == 0, "issues": issues,
                "expected_trades": expected_trades, "actual_trades": actual_trades}


if __name__ == "__main__":
    import json, os, tempfile

    tmp = tempfile.mkdtemp()
    ledger_path = os.path.join(tmp, "test_reconciliation.jsonl")
    ledger = ExecutionLedger(ledger_path)

    engine = ReconciliationEngine(ledger)

    ev1 = TradeEvent(event_type="trade_opened", symbol="EURJPY", direction="BUY",
                     volume=0.1, mt5_ticket=1001)
    ev2 = TradeEvent(event_type="trade_opened", symbol="USDJPY", direction="SELL",
                     volume=0.2, mt5_ticket=1002)
    ledger.append(ev1)
    ledger.append(ev2)

    mt5_positions = [
        {"ticket": 1001, "symbol": "EURJPY", "volume": 0.1, "type": "BUY"},
        {"ticket": 1002, "symbol": "USDJPY", "volume": 0.2, "type": "SELL"},
    ]

    report = engine.reconcile(mt5_positions, "GREEN")
    assert report["match"] is True, f"Expected match=True, got {report}"
    assert report["mt5_count"] == 2
    assert report["ledger_count"] == 2
    assert len(report["orphan_mt5"]) == 0
    assert len(report["orphan_ledger"]) == 0
    assert report["state_match"] is True
    print(f"Reconcile OK: match={report['match']} drift={report['drift_score']:.4f}")

    orphan_test = engine.detect_orphans(mt5_positions)
    assert len(orphan_test) == 0
    print("Detect orphans OK: no orphans")

    orphans = engine.detect_orphans([
        {"ticket": 9999, "symbol": "XAUUSD", "volume": 0.5},
    ] + mt5_positions)
    assert len(orphans) == 1
    assert orphans[0]["ticket"] == 9999
    print(f"Detect orphans OK: found {len(orphans)} orphan(s)")

    drift = engine.compute_drift_score()
    assert 0.0 <= drift <= 1.0
    print(f"Drift score OK: {drift:.4f}")

    lc = engine.verify_lifecycle({"events": ledger.get_all()})
    assert lc["pass"] is True
    print(f"Lifecycle OK: pass={lc['pass']} checked={lc['events_checked']}")

    audit_result = engine.full_audit(mt5_positions, "GREEN")
    assert audit_result["match"] is True
    assert audit_result["lifecycle"]["pass"] is True
    assert audit_result["ledger_integrity"]["pass"] is True
    assert audit_result["drift_score"] >= 0.0
    print(f"Full audit OK: match={audit_result['match']} "
          f"events={audit_result['total_events']} "
          f"open={audit_result['open_positions']}")

    bad_mt5 = [
        {"ticket": 1001, "symbol": "EURJPY", "volume": 0.1},
        {"ticket": 9999, "symbol": "XAUUSD", "volume": 0.5},
    ]
    bad_report = engine.reconcile(bad_mt5, "RED")
    assert bad_report["match"] is False
    assert len(bad_report["orphan_mt5"]) == 1
    assert len(bad_report["orphan_ledger"]) == 1
    print(f"Bad reconcile OK: match={bad_report['match']} "
          f"orphan_mt5={len(bad_report['orphan_mt5'])} "
          f"orphan_ledger={len(bad_report['orphan_ledger'])}")

    lc_fail = engine.verify_lifecycle({
        "events": [
            TradeEvent(event_id="EVT_000001", timestamp=100, event_type="trade_closed", mt5_ticket=2000),
            TradeEvent(event_id="EVT_000001", timestamp=100, event_type="trade_closed", mt5_ticket=2000),
        ]
    })
    assert lc_fail["pass"] is False
    assert any("Duplicate" in i for i in lc_fail["issues"])
    print(f"Lifecycle failure OK: issues={lc_fail['issues']}")

    import shutil
    shutil.rmtree(tmp)
    print("\nAll reconciliation engine tests passed.")
