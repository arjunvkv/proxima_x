import os
import json
import logging
from datetime import datetime
from typing import Optional
from proxima_ops.monitoring.deployment_context import DeploymentContext

logger = logging.getLogger("proxima_ops.freq_reality.executed")


class ExecutedSignalTracker:
    def __init__(self, data_dir: str = None, deployment_context: DeploymentContext = None):
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(__file__), "data")
        os.makedirs(data_dir, exist_ok=True)
        self._path = os.path.join(data_dir, "executed_signals.jsonl")
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
               threshold: float, entry_price: float,
               future_returns: Optional[dict] = None,
               actual_pnl: float = None, ticket: int = None,
               signal_id: str = None) -> dict:
        rec = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "es_rank": round(es_rank, 4),
            "at_rank": round(at_rank, 4),
            "threshold": threshold,
            "entry_price": round(entry_price, 5) if entry_price else 0.0}
        if self._ctx:
            rec["deployment_id"] = self._ctx.deployment_id
            rec["session_id"] = self._ctx.session_id
        if signal_id:
            rec["signal_id"] = signal_id
        if future_returns:
            for k, v in future_returns.items():
                rec[k] = v
        if actual_pnl is not None:
            rec["actual_pnl"] = round(actual_pnl, 2)
        if ticket:
            rec["ticket"] = ticket
        self._records.append(rec)
        self._save(rec)
        return rec

    def get_all(self) -> list[dict]:
        return list(self._records)

    def count(self) -> int:
        return len(self._records)
