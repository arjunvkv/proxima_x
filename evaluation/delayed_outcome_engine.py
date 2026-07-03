from typing import Dict, List, Any


class DelayedOutcomeEngine:
    """
    DOA Layer — Delayed Outcome Alignment

    Evaluates whether generated signals were directionally correct
    after a fixed horizon.
    """

    def __init__(self, horizon_ticks: int = 20):
        self.horizon = horizon_ticks
        self._history: List[Dict[str, Any]] = []

    @property
    def ready(self) -> bool:
        return len(self._history) >= self.horizon

    def record_snapshot(self, eval_data: Dict[str, Dict[str, Any]]):
        snapshot = {}
        for sym, data in eval_data.items():
            snapshot[sym] = {
                "price": data.get("price"),
                "signal": data.get("signal", 0),
                "ecdf": data.get("ecdf_rank", 0.5),
            }
        self._history.append(snapshot)
        if len(self._history) > self.horizon * 5:
            self._history.pop(0)

    def evaluate(self,
                 current_prices: Dict[str, float]) -> Dict[str, float]:
        if len(self._history) < self.horizon:
            return {}

        past = self._history[-self.horizon]
        results = {}

        for sym, past_data in past.items():
            if sym not in current_prices:
                continue
            entry_price = past_data.get("price")
            signal = past_data.get("signal", 0)
            exit_price = current_prices[sym]
            if entry_price is None:
                continue
            move = exit_price - entry_price
            if signal == 0:
                score = 0.0
            elif signal > 0:
                score = 1.0 if move > 0 else -1.0
            else:
                score = 1.0 if move < 0 else -1.0
            results[sym] = score

        return results
