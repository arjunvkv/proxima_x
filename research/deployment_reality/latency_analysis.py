import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger("proxima_ops.drl.latency")


class LatencyAnalysis:
    def __init__(self):
        self._records: list[dict] = []

    def record(self, signal_id: str, symbol: str,
               ts_generated: str, ts_triggered: Optional[str],
               ts_submitted: Optional[str], ts_accepted: Optional[str],
               ts_opened: Optional[str]) -> dict:
        def _ms(d1, d2):
            if not d1 or not d2:
                return None
            f1 = datetime.fromisoformat(d1)
            f2 = datetime.fromisoformat(d2)
            return int((f2 - f1).total_seconds() * 1000)

        rec = {
            "signal_id": signal_id,
            "symbol": symbol,
            "signal_to_submit_ms": _ms(ts_generated, ts_submitted),
            "submit_to_accept_ms": _ms(ts_submitted, ts_accepted),
            "accept_to_open_ms": _ms(ts_accepted, ts_opened),
            "total_latency_ms": _ms(ts_generated, ts_opened)}
        self._records.append(rec)

        outcome = None
        if rec["total_latency_ms"] is not None:
            outcome = "HIGH" if rec["total_latency_ms"] > 500 else "LOW"
        rec["latency_class"] = outcome
        return rec

    def summary(self) -> dict:
        if not self._records:
            return {"count": 0}
        totals = [r for r in self._records if r["total_latency_ms"] is not None]
        if not totals:
            return {"count": len(self._records), "measurable": 0}
        latencies = [r["total_latency_ms"] for r in totals]
        low_lat = [r for r in totals if r.get("latency_class") == "LOW"]
        high_lat = [r for r in totals if r.get("latency_class") == "HIGH"]
        return {
            "count": len(self._records),
            "measurable": len(totals),
            "mean_latency_ms": round(sum(latencies) / len(latencies), 1),
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
            "low_latency_count": len(low_lat),
            "high_latency_count": len(high_lat),
            "mean_signal_to_submit_ms": round(
                sum(r["signal_to_submit_ms"] for r in totals if r["signal_to_submit_ms"] is not None) /
                max(sum(1 for r in totals if r["signal_to_submit_ms"] is not None), 1), 1),
            "mean_submit_to_accept_ms": round(
                sum(r["submit_to_accept_ms"] for r in totals if r["submit_to_accept_ms"] is not None) /
                max(sum(1 for r in totals if r["submit_to_accept_ms"] is not None), 1), 1)}
