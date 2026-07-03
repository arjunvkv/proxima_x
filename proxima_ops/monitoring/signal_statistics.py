import numpy as np
from typing import Dict, List

class SignalStatistics:
    def __init__(self):
        self._es_history: Dict[str, List[float]] = {}
        self._at_history: Dict[str, List[float]] = {}

    def record_evaluation(self, symbol: str, es_rank: float, at_rank: float):
        if symbol not in self._es_history:
            self._es_history[symbol] = []
            self._at_history[symbol] = []
        self._es_history[symbol].append(es_rank)
        self._at_history[symbol].append(at_rank)
        
        # Keep rolling window of last 1000 evaluations
        if len(self._es_history[symbol]) > 1000:
            self._es_history[symbol].pop(0)
            self._at_history[symbol].pop(0)

    def get_metrics(self, symbol: str) -> dict:
        es_list = self._es_history.get(symbol, [])
        at_list = self._at_history.get(symbol, [])
        if not es_list:
            return {"mean_es_rank": 0.0, "mean_at_rank": 0.0, "std_es_rank": 0.0}
        return {
            "mean_es_rank": float(np.mean(es_list)),
            "mean_at_rank": float(np.mean(at_list)),
            "std_es_rank": float(np.std(es_list))
        }
