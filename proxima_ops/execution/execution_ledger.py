from dataclasses import dataclass, field, asdict
from typing import Optional, Any
import json, time, os
from collections import defaultdict

@dataclass
class TradeEvent:
    event_id: str = ""
    timestamp: float = field(default_factory=time.time)
    event_type: str = ""  # "signal_evaluated", "execution_authorized", "trade_opened", "trade_closed", "execution_denied", "system_error", "reconciliation"
    signal_id: str = ""  # "edge_04"
    symbol: str = ""
    direction: str = ""
    volume: float = 0.0
    entry_price: float = 0.0
    exit_price: float = 0.0
    mt5_ticket: int = 0
    pnl: float = 0.0
    segl_state: str = ""
    mof_state: str = ""
    mof_score: float = 0.0
    rf_drift: float = 0.0
    portfolio_conflict: float = 0.0
    frequency_budget_remaining: int = 0
    authorization_path: list = field(default_factory=list)
    denial_reason: str = ""
    intent_compliant: bool = True
    lifecycle_match: bool = True
    reconciliation_status: str = ""
    system_state_snapshot: dict = field(default_factory=dict)

class ExecutionLedger:
    def __init__(self, ledger_path: str = "state/execution_ledger.jsonl"):
        self.ledger_path = ledger_path
        self._ensure_ledger()
        self._events: list[TradeEvent] = []
        self._event_counter = 0
        self._load()

    def _ensure_ledger(self):
        os.makedirs(os.path.dirname(self.ledger_path), exist_ok=True)
        if not os.path.exists(self.ledger_path):
            with open(self.ledger_path, "w") as f:
                pass

    def _load(self):
        if os.path.exists(self.ledger_path) and os.path.getsize(self.ledger_path) > 0:
            with open(self.ledger_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        self._events.append(TradeEvent(**data))
                        self._event_counter = max(self._event_counter, int(data.get("event_id", "0").split("_")[-1]))

    def append(self, event: TradeEvent):
        event.event_id = f"EVT_{self._event_counter + 1:06d}"
        self._event_counter += 1
        self._events.append(event)
        with open(self.ledger_path, "a") as f:
            f.write(json.dumps(asdict(event), default=str) + "\n")
        return event.event_id

    def get_all(self) -> list[TradeEvent]:
        return list(self._events)

    def get_trades(self) -> list[TradeEvent]:
        return [e for e in self._events if e.event_type in ("trade_opened", "trade_closed")]

    def get_open_trades(self) -> list[TradeEvent]:
        opened = {}
        for e in sorted(self._events, key=lambda x: x.timestamp):
            if e.event_type == "trade_opened" and e.mt5_ticket:
                opened[e.mt5_ticket] = e
            elif e.event_type == "trade_closed" and e.mt5_ticket:
                opened.pop(e.mt5_ticket, None)
        return list(opened.values())

    def get_events_by_type(self, event_type: str) -> list[TradeEvent]:
        return [e for e in self._events if e.event_type == event_type]

    def get_last_event(self) -> Optional[TradeEvent]:
        return self._events[-1] if self._events else None

    def get_last_trade(self) -> Optional[TradeEvent]:
        trades = self.get_trades()
        return trades[-1] if trades else None

    def get_stats(self) -> dict:
        closed_trades = [e for e in self._events if e.event_type == "trade_closed" and e.pnl != 0]
        wins = [t for t in closed_trades if t.pnl > 0]
        losses = [t for t in closed_trades if t.pnl < 0]
        total_pnl = sum(t.pnl for t in closed_trades)
        return {
            "total_events": len(self._events),
            "total_trades": len(closed_trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(closed_trades) * 100 if closed_trades else 0,
            "total_pnl": total_pnl,
            "avg_win": sum(t.pnl for t in wins) / len(wins) if wins else 0,
            "avg_loss": sum(t.pnl for t in losses) / len(losses) if losses else 0,
            "profit_factor": abs(sum(t.pnl for t in wins) / sum(t.pnl for t in losses)) if losses and sum(t.pnl for t in losses) != 0 else float('inf') if wins else 0,
        }

    def reconcile_with_mt5(self, mt5_positions: list) -> dict:
        ledger_open = self.get_open_trades()
        mt5_tickets = {p.get("ticket") if isinstance(p, dict) else p.ticket for p in mt5_positions}
        ledger_tickets = {e.mt5_ticket for e in ledger_open if e.mt5_ticket}

        orphan_mt5 = mt5_tickets - ledger_tickets
        orphan_ledger = ledger_tickets - mt5_tickets

        return {
            "match": len(orphan_mt5) == 0 and len(orphan_ledger) == 0,
            "orphan_mt5": list(orphan_mt5),
            "orphan_ledger": list(orphan_ledger),
            "mt5_count": len(mt5_positions),
            "ledger_count": len(ledger_open),
        }

    def integrity_check(self) -> dict:
        issues = []
        for i, e in enumerate(self._events):
            if e.event_type == "trade_closed" and e.mt5_ticket:
                opened = [oe for oe in self._events[:i] if oe.event_type == "trade_opened" and oe.mt5_ticket == e.mt5_ticket]
                if not opened:
                    issues.append(f"Close without open: ticket {e.mt5_ticket}")
        return {
            "pass": len(issues) == 0,
            "issues": issues,
            "total_events": len(self._events),
        }


if __name__ == "__main__":
    ledger = ExecutionLedger("state/test_ledger.jsonl")
    event = TradeEvent(
        event_type="signal_evaluated",
        signal_id="edge_04",
        symbol="EURJPY",
        direction="BUY",
        volume=0.1,
        segl_state="GREEN",
        mof_state="NORMAL",
        mof_score=78.5,
        rf_drift=0.02,
        portfolio_conflict=0.0,
        frequency_budget_remaining=5,
        authorization_path=["moi_check", "segl_check", "lifecycle_check"],
        intent_compliant=True,
        lifecycle_match=True,
    )
    eid = ledger.append(event)
    print(f"Appended event: {eid}")
    print(f"Total events: {len(ledger.get_all())}")
    print(f"LEDGER OK")
