import logging
from datetime import datetime

logger = logging.getLogger("proxima_ops.tracker")


class OrderTracker:
    def __init__(self):
        self._attempts: list[dict] = []

    def record_attempt(self, signal_id: str, symbol: str, action: str,
                       volume: float, price: float, es_rank: float = 0.0,
                       at_rank: float = 0.0):
        rec = {
            "timestamp": datetime.now().strftime('%H:%M:%S'),
            "signal_id": signal_id, "symbol": symbol, "action": action,
            "volume": volume, "price": price,
            "es_rank": es_rank, "at_rank": at_rank,
            "submission": "PENDING", "retcode": None, "ticket": None}
        self._attempts.append(rec)
        return rec

    def record_result(self, signal_id: str, success: bool,
                      retcode: int = None, ticket: int = None,
                      comment: str = None):
        for rec in reversed(self._attempts):
            if rec["signal_id"] == signal_id:
                rec["submission"] = "SUCCESS" if success else "FAILED"
                rec["retcode"] = retcode
                rec["ticket"] = ticket
                rec["comment"] = comment
                break

    def get_recent(self, n: int = 20) -> list[dict]:
        return self._attempts[-n:]

    def last_for_symbol(self, symbol: str) -> dict:
        for rec in reversed(self._attempts):
            if rec["symbol"] == symbol:
                return rec
        return {}
