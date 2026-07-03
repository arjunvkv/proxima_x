import os
import json
import logging
from datetime import datetime
from typing import Optional
from proxima_ops.monitoring.deployment_context import DeploymentContext

logger = logging.getLogger("proxima_ops.freq_reality.blocked")


class BlockedSignalTracker:
    def __init__(self, data_dir: str = None, deployment_context: DeploymentContext = None):
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(__file__), "data")
        os.makedirs(data_dir, exist_ok=True)
        self._path = os.path.join(data_dir, "blocked_signals.jsonl")
        self._records: list[dict] = []
        self._ctx = deployment_context
        self._load()

    def _load(self):
        if os.path.exists(self._path):
            with open(self._path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self._records.append(json.loads(line))

    def _save(self, rec: dict):
        with open(self._path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def record(self, symbol: str, es_rank: float, at_rank: float,
               threshold: float, block_reason: str,
               price: float = 0.0,
               future_returns: Optional[dict] = None,
               signal_id: str = None) -> dict:
        rec = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "es_rank": round(es_rank, 4),
            "at_rank": round(at_rank, 4),
            "threshold": threshold,
            "block_reason": block_reason,
            "price": round(price, 5) if price else 0.0}
        if self._ctx:
            rec["deployment_id"] = self._ctx.deployment_id
            rec["session_id"] = self._ctx.session_id
        if signal_id:
            rec["signal_id"] = signal_id
        if future_returns:
            for k, v in future_returns.items():
                rec[k] = v
        self._records.append(rec)
        self._save(rec)
        return rec

    def get_all(self) -> list[dict]:
        return list(self._records)

    def count(self) -> int:
        return len(self._records)

    def filter(self, reason: str = None, symbol: str = None,
               es_min: float = 0.0, at_min: float = 0.0) -> list[dict]:
        result = self._records
        if reason:
            result = [r for r in result if r.get("block_reason") == reason]
        if symbol:
            result = [r for r in result if r.get("symbol") == symbol]
        if es_min > 0:
            result = [r for r in result if r.get("es_rank", 0) >= es_min]
        if at_min > 0:
            result = [r for r in result if r.get("at_rank", 0) >= at_min]
        return result

    def summary(self) -> dict:
        total = len(self._records)
        reasons = {}
        symbols = {}
        for r in self._records:
            reason = r.get("block_reason", "UNKNOWN")
            reasons[reason] = reasons.get(reason, 0) + 1
            sym = r.get("symbol", "UNKNOWN")
            symbols[sym] = symbols.get(sym, 0) + 1
        return {"total": total, "reasons": reasons, "symbols": symbols}
