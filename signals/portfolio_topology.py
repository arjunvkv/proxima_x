import logging
from typing import Dict, List, Optional

logger = logging.getLogger("proxima_demo")

PORTFOLIO_STATES = ["DISTRIBUTED", "FOCUSED", "STRESSED", "CASCADING"]


class PortfolioTopologyEngine:
    def __init__(self):
        self._allocations: Dict[str, float] = {}
        self._trust: Dict[str, float] = {}
        self._pressure: Dict[str, float] = {}
        self._ruptures: Dict[str, bool] = {}
        self._history: List[dict] = []
        self._global_reduce = False

    def update(self, symbol: str, alloc: float,
               trust: float, pressure: float, rupture: bool):
        self._allocations[symbol] = alloc
        self._trust[symbol] = trust
        self._pressure[symbol] = pressure
        self._ruptures[symbol] = rupture
        c = self.concentration()
        s = self.stress_cluster()
        cr = self.cascade_risk()
        state = self.portfolio_state()
        self._history.append({
            "symbols": len(self._allocations),
            "concentration": c,
            "stress_cluster": s,
            "cascade_risk": cr,
            "state": state,
        })
        logger.info(f"[PORTFOLIO_TOPOLOGY] {symbol} alloc={alloc:.3f} "
                    f"trust={trust:.3f} pressure={pressure:.3f} "
                    f"rupture={rupture} state={state}")

    def concentration(self) -> float:
        allocs = list(self._allocations.values())
        total = sum(allocs)
        if total == 0:
            return 0.0
        h = sum((a / total) ** 2 for a in allocs)
        logger.info(f"[PORTFOLIO_CONCENTRATION] H={h:.3f}")
        return h

    def stress_cluster(self) -> float:
        if not self._pressure:
            return 0.0
        stressed = sum(
            1 for s in self._pressure
            if self._pressure[s] > 0.70 or self._ruptures.get(s, False)
        )
        return stressed / len(self._pressure)

    def cascade_risk(self) -> float:
        cr = self.concentration() * self.stress_cluster()
        logger.info(f"[CASCADE_RISK] CR={cr:.3f}")
        return cr

    def portfolio_state(self) -> str:
        cr = self.cascade_risk()
        if cr < 0.20:
            return "DISTRIBUTED"
        elif cr < 0.40:
            return "FOCUSED"
        elif cr < 0.60:
            return "STRESSED"
        return "CASCADING"

    def global_compression(self) -> bool:
        if self.portfolio_state() == "CASCADING":
            self._global_reduce = True
            logger.info("[PORTFOLIO_GLOBAL] CASCADING: all new allocations halved")
            return True
        return False

    def history(self) -> List[dict]:
        return list(self._history)

    def stats(self) -> dict:
        states = set(h["state"] for h in self._history)
        spike_events = sum(1 for h in self._history if h["concentration"] > 0.40)
        cascade_events = sum(1 for h in self._history if h["cascade_risk"] > 0.30)
        return {
            "updates": len(self._history),
            "states_seen": len(states),
            "states": sorted(states),
            "concentration_spikes": spike_events,
            "cascade_events": cascade_events,
            "global_compress_triggered": self._global_reduce,
        }
