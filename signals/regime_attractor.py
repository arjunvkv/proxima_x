import logging
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple, Set
import math

logger = logging.getLogger("proxima_demo")


class RegimeAttractorEngine:
    def __init__(self, max_cycle_search: int = 8):
        self._max_cycle_search = max_cycle_search
        self._history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self._state_visits: Dict[str, Dict[tuple, int]] = defaultdict(lambda: defaultdict(int))
        self._escape_times: Dict[str, Dict[tuple, List[int]]] = defaultdict(lambda: defaultdict(list))
        self._return_times: Dict[str, Dict[tuple, List[int]]] = defaultdict(lambda: defaultdict(list))
        self._cycles: Dict[str, Dict[tuple, int]] = defaultdict(lambda: defaultdict(int))
        self._last_state: Dict[str, Optional[tuple]] = {}
        self._consecutive: Dict[str, int] = defaultdict(int)
        self._last_seen: Dict[str, Dict[tuple, int]] = defaultdict(lambda: defaultdict(int))

    def update(self, symbol: str, state: tuple):
        hist = self._history[symbol]
        self._state_visits[symbol][state] += 1
        prev = self._last_state.get(symbol)
        if prev is not None:
            if prev == state:
                self._consecutive[symbol] += 1
            else:
                self._escape_times[symbol][prev].append(self._consecutive[symbol])
                self._consecutive[symbol] = 1
            if prev in self._last_seen[symbol] and prev != state:
                gap = len(hist) - self._last_seen[symbol][prev]
                if gap > 0:
                    self._return_times[symbol][prev].append(gap)
        else:
            self._consecutive[symbol] = 1
        self._last_seen[symbol][state] = len(hist)
        hist.append(state)
        self._last_state[symbol] = state
        self._detect_cycles(symbol)
        logger.debug(f"[REGIME_ATTRACTOR] {symbol} state={state}")

    def _detect_cycles(self, symbol: str):
        hist = list(self._history[symbol])
        n = len(hist)
        if n < 2:
            return
        for length in range(2, min(n, self._max_cycle_search) + 1):
            if hist[-length] == hist[-1] and length > 1:
                cycle = tuple(hist[-length:])
                self._cycles[symbol][cycle] += 1
                logger.info(f"[ATTRACTOR_CYCLE] {symbol} cycle={'->'.join(str(s) for s in cycle)} "
                            f"count={self._cycles[symbol][cycle]}")

    def attractor_strength(self, symbol: str, state: tuple) -> float:
        visits = self._state_visits.get(symbol, {}).get(state, 0)
        if visits == 0:
            return 0.0
        escape = self._escape_times.get(symbol, {}).get(state, [])
        ret = self._return_times.get(symbol, {}).get(state, [])
        mean_escape = sum(escape) / max(len(escape), 1)
        mean_return = sum(ret) / max(len(ret), 1)
        if mean_return == 0:
            return 0.0
        return_rate = 1.0 / mean_return
        raw = visits * return_rate / max(mean_escape, 1)
        norm = min(raw / 10.0, 1.0)
        return round(norm, 4)

    def escape_time(self, symbol: str, state: tuple) -> float:
        times = self._escape_times.get(symbol, {}).get(state, [])
        if not times:
            cur = self._consecutive.get(symbol, 0)
            last = self._last_state.get(symbol)
            if last == state and cur > 0:
                return float(cur)
            return 0.0
        return round(sum(times) / len(times), 2)

    def return_time(self, symbol: str, state: tuple) -> float:
        times = self._return_times.get(symbol, {}).get(state, [])
        if not times:
            return 0.0
        return round(sum(times) / len(times), 2)

    def cycle_probability(self, symbol: str, cycle: tuple) -> float:
        total = sum(self._cycles.get(symbol, {}).values())
        if total == 0:
            return 0.0
        cnt = self._cycles.get(symbol, {}).get(cycle, 0)
        return round(cnt / total, 4)

    def all_cycles(self, symbol: str) -> Dict[tuple, int]:
        return dict(self._cycles.get(symbol, {}))

    def metastable(self, symbol: str) -> bool:
        dom = self.dominant_attractor(symbol)
        if dom is None:
            return False
        strength = self.attractor_strength(symbol, dom)
        escape = self.escape_time(symbol, dom)
        ret = self.return_time(symbol, dom)
        return strength > 0.3 and escape >= 2.0 and (ret == 0 or ret <= 3.0)

    def dominant_attractor(self, symbol: str) -> Optional[tuple]:
        best = None
        best_str = -1.0
        for st in self._state_visits.get(symbol, {}):
            s = self.attractor_strength(symbol, st)
            if s > best_str:
                best_str = s
                best = st
        return best

    def stats(self) -> dict:
        all_symbols = set(self._history.keys())
        total_states = sum(len(v) for v in self._state_visits.values())
        total_cycles = sum(sum(c.values()) for c in self._cycles.values())
        strengths = []
        dom_attractors = {}
        meta_symbols = []
        for sym in all_symbols:
            dom = self.dominant_attractor(sym)
            if dom is not None:
                s = self.attractor_strength(sym, dom)
                strengths.append(s)
                dom_attractors[sym] = dom
            if self.metastable(sym):
                meta_symbols.append(sym)
        mean_s = round(sum(strengths) / max(len(strengths), 1), 4)
        return {
            "symbols": len(all_symbols),
            "states": total_states,
            "cycles": total_cycles,
            "dominant_attractors": len(dom_attractors),
            "metastable_symbols": len(meta_symbols),
            "mean_strength": mean_s,
        }
