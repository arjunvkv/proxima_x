from typing import Dict, Any


class CALEngine:
    """
    Causal Attribution Layer (CAL)

    Decomposes DOA outcome into feature-level contributions.
    """

    def __init__(self):
        pass

    def attribute(self,
                  eval_data: Dict[str, Dict[str, Any]],
                  doa_results: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        attribution = {}

        for sym, data in eval_data.items():
            if sym not in doa_results:
                continue

            outcome = doa_results[sym]

            ecdf = float(data.get("ecdf_rank", 0.5))
            entropy = float(data.get("entropy", 0.5))
            spread_val = data.get("spread")
            spread = float(spread_val if spread_val is not None else 0.0)
            signal = float(data.get("signal", 0.0))

            ecdf_contrib = ecdf * outcome
            entropy_contrib = (1.0 - entropy) * outcome
            spread_contrib = -spread * abs(outcome)
            signal_contrib = signal * outcome

            attribution[sym] = {
                "ecdf_contrib": ecdf_contrib,
                "entropy_contrib": entropy_contrib,
                "spread_contrib": spread_contrib,
                "signal_contrib": signal_contrib,
                "total": ecdf_contrib + entropy_contrib + spread_contrib + signal_contrib,
            }

        return attribution
