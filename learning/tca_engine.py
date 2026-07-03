from typing import Dict, List, Any


class TemporalCreditAssignment:
    """
    TCA — Temporal Credit Assignment Layer

    Assigns outcome credit back across historical timesteps.
    """

    def __init__(self,
                 decay: float = 0.85,
                 max_history: int = 50):
        self.decay = decay
        self.max_history = max_history
        self.history: List[Dict[str, Any]] = []

    def record(self, eval_data: Dict[str, Dict[str, Any]]):
        snapshot = {}

        for sym, data in eval_data.items():
            snapshot[sym] = {
                "ecdf": data.get("ecdf_rank", 0.5),
                "entropy": data.get("entropy", 0.5),
                "signal": data.get("signal", 0),
                "price": data.get("price"),
            }

        self.history.append(snapshot)

        if len(self.history) > self.max_history:
            self.history.pop(0)

    def assign_credit(self,
                      current_prices: Dict[str, float],
                      outcome_map: Dict[str, float]) -> Dict[str, Dict[str, float]]:

        if not self.history:
            return {}

        credits = {}

        for i, snapshot in enumerate(reversed(self.history)):
            weight = self.decay ** i

            for sym, data in snapshot.items():
                if sym not in outcome_map:
                    continue

                outcome = outcome_map[sym]

                if sym not in credits:
                    credits[sym] = {"ecdf": 0.0, "entropy": 0.0, "signal": 0.0}

                credits[sym]["ecdf"] += data["ecdf"] * outcome * weight
                credits[sym]["entropy"] += (1.0 - data["entropy"]) * outcome * weight
                credits[sym]["signal"] += data["signal"] * outcome * weight

        return credits
