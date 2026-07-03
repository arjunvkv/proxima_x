"""Broker-driven state reconciliation.

Derives internal system state (ledger, staircase, PnL) from MT5 deal history
as the authoritative source of truth. Designed to restore consistency after
the executor misses a trade closure event (e.g., broker-hit TP/SL while
the orchestrator loop was not running).

Only reconciles trades that our ledger already knows about (by mt5_ticket).
Does NOT import MT5 history for trades our system didn't originate.
"""

import json
import logging
import os
import time
from collections import defaultdict
from typing import Optional

from proxima_x.proxima_ops.execution.execution_ledger import ExecutionLedger, TradeEvent

logger = logging.getLogger("proxima_ops.reconciliation.broker")

_MT5_TYPE_TO_DIR = {0: "BUY", 1: "SELL"}


class BrokerReconciliation:
    def __init__(self, mt5_connector, ledger: ExecutionLedger,
                 staircase_path: str = "state/volume_staircase_state.json"):
        self.mt5 = mt5_connector
        self.ledger = ledger
        self.staircase_path = staircase_path
        self._last_logged_state: dict[int, str] = {}

    def _load_staircase(self) -> dict:
        if os.path.exists(self.staircase_path):
            with open(self.staircase_path) as f:
                return json.load(f)
        return {"completed_trades": 0, "current_phase": 1}

    def _save_staircase(self, state: dict):
        os.makedirs(os.path.dirname(self.staircase_path) or ".", exist_ok=True)
        with open(self.staircase_path, "w") as f:
            json.dump(state, f, indent=2)

    def reconcile(self, hours_back: int = 48,
                  positions: Optional[list] = None) -> dict:
        """Reconcile ledger open trades against MT5 deal history.

        For each open trade in our ledger (with mt5_ticket), checks if
        MT5 has a corresponding exit deal. If yes, appends a close event
        and updates the staircase.

        Args:
            hours_back: How many hours of MT5 history to check.
            positions: Current MT5 open positions (fetched if not provided).
        """
        report = {
            "timestamp": time.time(),
            "mt5_deals_fetched": 0,
            "ledger_events_before": len(self.ledger.get_all()),
            "ledger_open_before": len(self.ledger.get_open_trades()),
            "closes_appended": 0,
            "confirmed_closed_no_deal": 0,
            "staircase_before": self._load_staircase(),
            "staircase_after": None,
            "errors": [],
        }

        # 1. Find which mt5_tickets in our ledger are still open
        ledger_open = self.ledger.get_open_trades()
        open_tickets = {e.mt5_ticket for e in ledger_open if e.mt5_ticket}

        if positions is None:
            positions = self.mt5.get_positions()
        mt5_open_tickets = {p["ticket"] for p in positions}
        mt5_closed_tickets = open_tickets - mt5_open_tickets

        if not open_tickets:
            report["staircase_after"] = report["staircase_before"]
            report["note"] = "No open tickets in ledger to reconcile"
            return report

        # 2. Fetch all MT5 deals (no position filter) and match by order field
        all_mt5_deals = self.mt5.get_deal_history(hours_back=hours_back)
        report["mt5_deals_fetched"] = len(all_mt5_deals)

        # Build map: order_ticket -> deals for that order
        deals_by_order = defaultdict(list)
        for d in all_mt5_deals:
            oid = d.get("order", d.get("position_id", 0))
            if oid:
                deals_by_order[oid].append(d)


        closes_appended = 0
        confirmed_closed = set()
        for ticket in open_tickets:
            deals = deals_by_order.get(ticket, [])
            exit_deals = [d for d in deals if d.get("entry") == 1]

            if exit_deals:
                entry_deal = next((d for d in deals if d.get("entry") == 0), None)
                if not entry_deal:
                    continue

                exit_deal = exit_deals[0]
                sym = entry_deal.get("symbol", "")
                vol = entry_deal.get("volume", 0.01)
                entry_price = entry_deal.get("price", 0.0)
                exit_price = exit_deal.get("price", 0.0)
                pnl = exit_deal.get("profit", 0.0) + exit_deal.get("swap", 0.0) + exit_deal.get("commission", 0.0)
                mt5_type = entry_deal.get("type", 0)
                direction = _MT5_TYPE_TO_DIR.get(mt5_type, "BUY")
                exit_time = exit_deal.get("time", time.time())

                close_ev = TradeEvent(
                    event_type="trade_closed",
                    signal_id="unknown",
                    symbol=sym,
                    direction=direction,
                    volume=vol,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    mt5_ticket=ticket,
                    pnl=pnl,
                    timestamp=exit_time,
                    segl_state="",
                    mof_state="",
                    mof_score=0.0,
                    rf_drift=0.0,
                    portfolio_conflict=0.0,
                    frequency_budget_remaining=0,
                    authorization_path=[],
                    intent_compliant=True,
                    lifecycle_match=True,
                    reconciliation_status="broker_reconciled",
                )
                self.ledger.append(close_ev)
                closes_appended += 1
                confirmed_closed.add(ticket)
                if self._last_logged_state.get(ticket) != "APPENDED_CLOSE":
                    self._last_logged_state[ticket] = "APPENDED_CLOSE"
                    logger.info(
                        f"[BROKER_RECON] Appended close for ticket={ticket} "
                        f"{sym} {direction} pnl={pnl:.2f}"
                    )
            elif ticket in mt5_closed_tickets:
                confirmed_closed.add(ticket)
                if self._last_logged_state.get(ticket) != "CLOSED_CONFIRMED":
                    self._last_logged_state[ticket] = "CLOSED_CONFIRMED"
                    logger.info(
                        f"[BROKER_RECON] ticket={ticket} CLOSED_CONFIRMED "
                        f"(not in MT5 positions, no deal history yet)"
                    )

        report["closes_appended"] = closes_appended
        report["confirmed_closed_no_deal"] = len(confirmed_closed) - closes_appended

        # 3. Recalculate completed_trades from ledger + confirmed closed trades
        all_events = self.ledger.get_all()
        closed_from_ledger = [e for e in all_events
                              if e.event_type == "trade_closed" and e.pnl != 0]

        # Confirmed closed tickets that aren't yet in ledger count too
        ledger_closed_tickets = {e.mt5_ticket for e in closed_from_ledger}
        additional_completed = len(confirmed_closed - ledger_closed_tickets)
        completed_trades = len(closed_from_ledger) + additional_completed

        staircase = self._load_staircase()
        old_completed = staircase["completed_trades"]
        staircase["completed_trades"] = completed_trades

        if completed_trades < 10:
            staircase["current_phase"] = 1
        elif completed_trades < 25:
            staircase["current_phase"] = 2
        elif completed_trades < 50:
            staircase["current_phase"] = 3
        else:
            staircase["current_phase"] = 4

        self._save_staircase(staircase)
        report["staircase_after"] = staircase

        if completed_trades != old_completed:
            logger.info(
                f"[BROKER_RECON] Staircase: {old_completed} -> {completed_trades} "
                f"trades, phase={staircase['current_phase']}"
            )

        report["ledger_events_after"] = len(self.ledger.get_all())
        report["ledger_open_after"] = len(self.ledger.get_open_trades())
        report["total_completed_trades"] = completed_trades
        report["total_pnl_from_ledger"] = sum(e.pnl for e in closed_from_ledger)
        report["success"] = True

        return report
