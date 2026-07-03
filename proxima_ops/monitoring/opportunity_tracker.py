import time
from typing import List, Dict

class OpportunityTracker:
    def __init__(self):
        self._history: List[dict] = []
        self._latest_evals: Dict[str, dict] = {}

    def record_evaluation(self, symbol: str, es_value: float, es_rank: float,
                          at_rank: float, threshold: float, triggered: bool,
                          blocked: bool, block_reason: str):
        record = {
            "timestamp": int(time.time()),
            "symbol": symbol,
            "es_value": es_value,
            "es_rank": es_rank,
            "at_rank": at_rank,
            "threshold": threshold,
            "triggered": triggered,
            "blocked": blocked,
            "block_reason": block_reason
        }
        self._history.append(record)
        self._latest_evals[symbol] = record
        
        # Limit history length to prevent memory leaks
        if len(self._history) > 5000:
            self._history.pop(0)

    def get_latest_evals(self) -> List[dict]:
        return list(self._latest_evals.values())
        
    def get_history(self) -> List[dict]:
        return self._history
