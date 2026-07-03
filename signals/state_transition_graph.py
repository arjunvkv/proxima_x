import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("proxima_demo")


class StateTransitionGraph:
    def __init__(self):
        self._transitions: Dict[str, Dict[tuple, Dict[tuple, int]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int))
        )
        self._states: Dict[str, List[tuple]] = defaultdict(list)
        self._unique_states: Dict[str, set] = defaultdict(set)
        self._total: Dict[str, int] = defaultdict(int)

    def update(self, symbol: str, state: tuple):
        self._states[symbol].append(state)
        self._unique_states[symbol].add(state)
        if len(self._states[symbol]) >= 2:
            prev = self._states[symbol][-2]
            curr = self._states[symbol][-1]
            self._transitions[symbol][prev][curr] += 1
            self._total[symbol] += 1
            fwd = self._transitions[symbol][prev][curr]
            logger.info(f"[STATE_TRANSITION] {symbol} "
                        f"\"{prev}\" -> \"{curr}\" "
                        f"count={fwd} total={self._total[symbol]}")

    def transition_prob(self, from_state: tuple,
                        to_state: tuple, symbol: str) -> float:
        fwd = self._transitions[symbol].get(from_state, {})
        total_from = sum(fwd.values())
        if total_from == 0:
            return 0.0
        return fwd.get(to_state, 0) / total_from

    def entropy(self, symbol: str) -> float:
        import math
        all_fwd = self._transitions[symbol]
        total_entropy = 0.0
        total_transitions = 0
        for from_state, to_dict in all_fwd.items():
            total_from = sum(to_dict.values())
            if total_from <= 1:
                continue
            ent = 0.0
            for count in to_dict.values():
                p = count / total_from
                ent -= p * math.log2(p) if p > 0 else 0
            total_entropy += ent * total_from
            total_transitions += total_from
        if total_transitions == 0:
            return 0.0
        avg_ent = total_entropy / total_transitions
        logger.info(f"[TRANSITION_ENTROPY] {symbol} entropy={avg_ent:.3f}")
        return avg_ent

    def rare_transitions(self, symbol: str,
                         threshold: float = 0.10) -> List[Tuple[tuple, tuple, float]]:
        rare = []
        for from_state, to_dict in self._transitions[symbol].items():
            total_from = sum(to_dict.values())
            for to_state, count in to_dict.items():
                prob = count / total_from
                if prob < threshold:
                    rare.append((from_state, to_state, prob))
        if rare:
            logger.info(f"[RARE_TRANSITION] {symbol} {len(rare)} rare transitions")
        return rare

    def stats(self, symbol: Optional[str] = None) -> dict:
        if symbol:
            unique_states = len(self._unique_states[symbol])
            total_trans = self._total[symbol]
            all_triples = [(f, t, c)
                          for f, td in self._transitions[symbol].items()
                          for t, c in td.items()]
            recurring = sum(1 for _, _, c in all_triples if c >= 2)
            rare = len(self.rare_transitions(symbol))
            ent = self.entropy(symbol)
        else:
            unique_states = sum(len(s) for s in self._unique_states.values())
            total_trans = sum(self._total.values())
            all_triples = [(f, t, c)
                          for sym in self._transitions
                          for f, td in self._transitions[sym].items()
                          for t, c in td.items()]
            recurring = sum(1 for _, _, c in all_triples if c >= 2)
            rare = sum(len(self.rare_transitions(sym)) for sym in self._unique_states)
            ent = self.entropy(list(self._unique_states.keys())[0]) if self._unique_states else 0.0
        return {
            "symbols": len(self._states),
            "unique_states": unique_states,
            "total_transitions": total_trans,
            "recurring_transitions": recurring,
            "rare_transitions": rare,
            "entropy": round(ent, 3),
        }
