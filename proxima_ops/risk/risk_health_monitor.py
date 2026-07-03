"""RHL-7: MT5 Health Monitor — connection + symbol + order health."""

import logging
from typing import Optional

logger = logging.getLogger("proxima_ops.risk.health")


class RiskHealthMonitor:
    def __init__(self):
        self._failures: list[dict] = []
        self._entries_disabled: bool = False
        self._state = "HEALTHY"

    def check(self, mt5_connected: bool, account_connected: bool,
              symbols_available: dict, order_submission_ok: bool = True) -> dict:
        issues = []
        if not mt5_connected:
            issues.append("terminal_disconnected")
        if not account_connected:
            issues.append("account_disconnected")
        failed_symbols = [s for s, ok in symbols_available.items() if not ok]
        if failed_symbols:
            issues.append(f"symbols_unavailable:{','.join(failed_symbols[:3])}")
        if not order_submission_ok:
            issues.append("order_submission_failed")

        if issues:
            self._entries_disabled = True
            self._state = "BROKER_FAILURE"
            self._failures.append({"issues": issues})
            logger.warning(f"Risk health: BROKER_FAILURE — {issues}")
        else:
            if self._state == "BROKER_FAILURE":
                self._entries_disabled = False
                self._state = "HEALTHY"
                logger.info("Risk health: restored to HEALTHY")

        return {"state": self._state, "entries_disabled": self._entries_disabled, "issues": issues}
