import logging
from typing import Optional

logger = logging.getLogger("proxima_ops.drl.decay")


class SignalDecay:
    def __init__(self):
        self._records: list[dict] = []

    def record(self, signal_id: str, symbol: str,
               es_rank: float, at_rank: float,
               returns: dict, source: str = "EXECUTED"):
        rec = {
            "signal_id": signal_id,
            "symbol": symbol,
            "es_rank": es_rank,
            "at_rank": at_rank,
            "source": source}
        for k, v in returns.items():
            rec[k] = v
        self._records.append(rec)
        return rec

    def summary(self) -> dict:
        executed = [r for r in self._records if r.get("source") == "EXECUTED"]
        blocked = [r for r in self._records if r.get("source") == "BLOCKED"]
        result = {"executed_count": len(executed), "blocked_count": len(blocked)}
        for horizon in ["return_h1", "return_h5", "return_h20", "return_h50", "return_h100"]:
            e_vals = [r[horizon] for r in executed if r.get(horizon) is not None]
            b_vals = [r[horizon] for r in blocked if r.get(horizon) is not None]
            result[f"{horizon}_executed_mean"] = round(sum(e_vals) / len(e_vals), 6) if e_vals else None
            result[f"{horizon}_blocked_mean"] = round(sum(b_vals) / len(b_vals), 6) if b_vals else None
            if e_vals and b_vals:
                result[f"{horizon}_decay_gap"] = round(
                    (sum(e_vals) / len(e_vals)) - (sum(b_vals) / len(b_vals)), 6)
        return result
