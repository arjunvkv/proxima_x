from typing import Dict, List


class H20CapEngine:
    """
    V3/V4 Capital Allocation Layer

    Inputs:
        selected symbols (TOP-K)
        eval_data per symbol:
            - ecdf_rank
            - entropy
            - spread (optional)

    Output:
        allocation dict: {symbol: capital_weight}
    """

    def __init__(self,
                 max_cap_per_symbol: float = 0.60,
                 min_cap_per_symbol: float = 0.10):
        self.max_cap = max_cap_per_symbol
        self.min_cap = min_cap_per_symbol

    def _volatility_proxy(self, sym_data: Dict) -> float:
        entropy = sym_data.get("entropy", 0.5)
        spread = sym_data.get("spread", 0.0)
        entropy = float(entropy) if entropy is not None else 0.5
        spread = float(spread) if spread is not None else 0.0
        return entropy + (spread * 0.1)

    def allocate(self,
                 selected: List[str],
                 eval_data: Dict[str, Dict]) -> Dict[str, float]:

        if not selected:
            return {}

        raw_scores = {}
        for sym in selected:
            data = eval_data.get(sym, {})

            ecdf = data.get("ecdf_rank", 0.5)
            entropy = data.get("entropy", 0.5)
            ecdf = float(ecdf) if ecdf is not None else 0.5
            entropy = float(entropy) if entropy is not None else 0.5

            vol = self._volatility_proxy(data)
            score = ecdf * (1.0 - entropy) * (1.0 / (1.0 + vol))

            raw_scores[sym] = max(score, 1e-6)

        total = sum(raw_scores.values())
        weights = {k: v / total for k, v in raw_scores.items()}

        capped = {}
        overflow = 0.0
        for sym, w in weights.items():
            if w > self.max_cap:
                overflow += (w - self.max_cap)
                capped[sym] = self.max_cap
            else:
                capped[sym] = w

        if overflow > 0:
            eligible = [s for s in capped if capped[s] < self.max_cap]
            if eligible:
                add_per_sym = overflow / len(eligible)
                for s in eligible:
                    capped[s] = min(self.max_cap, capped[s] + add_per_sym)

        for sym in capped:
            if capped[sym] < self.min_cap:
                capped[sym] = self.min_cap

        final_total = sum(capped.values())
        return {k: v / final_total for k, v in capped.items()}
