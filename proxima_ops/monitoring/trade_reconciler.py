import logging
from typing import Optional

logger = logging.getLogger("proxima_ops.reconciler")


class TradeReconciler:
    def __init__(self, ledger, position_manager):
        self._ledger = ledger
        self._positions = position_manager

    def reconcile(self) -> dict:
        mt5_positions = self._positions.positions
        mt5_tickets = {p["ticket"]: p for p in mt5_positions}
        ledger_positions = self._ledger.get_open()
        ledger_tickets = {p["mt5_ticket"]: p for p in ledger_positions}

        mt5_only = set(mt5_tickets.keys()) - set(ledger_tickets.keys())
        ledger_only = set(ledger_tickets.keys()) - set(mt5_tickets.keys())
        matched = set(mt5_tickets.keys()) & set(ledger_tickets.keys())

        mismatches = []
        for t in mt5_only:
            mismatches.append({"ticket": t, "issue": "MT5_ONLY",
                               "symbol": mt5_tickets[t]["symbol"]})
        for t in ledger_only:
            mismatches.append({"ticket": t, "issue": "LEDGER_ONLY",
                               "symbol": ledger_tickets[t]["symbol"]})

        return {
            "mt5_open": len(mt5_positions),
            "ledger_open": len(ledger_positions),
            "matched": len(matched),
            "mismatches": mismatches,
            "healthy": len(mismatches) == 0}

    def resolve(self) -> dict:
        mt5_positions = self._positions.positions
        mt5_tickets = {p["ticket"]: p for p in mt5_positions}
        ledger_open = self._ledger.get_open()
        closed_count = 0
        for entry in ledger_open:
            t = entry.get("mt5_ticket", 0)
            if t > 0 and t not in mt5_tickets:
                self._ledger.close_by_ticket(t, exit_reason="BROKER_MISSING_RECONCILE")
                closed_count += 1
        result = self.reconcile()
        result["ghosts_closed"] = closed_count
        if closed_count > 0:
            logger.warning(f"Reconciliation closed {closed_count} ghost positions")
        return result
