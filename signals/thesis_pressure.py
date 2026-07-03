import logging
from collections import defaultdict
from typing import Dict, List, Optional

logger = logging.getLogger("proxima_demo")

PRESSURE_BANDS = {
    "stable": (0.0, 0.25),
    "stressed": (0.25, 0.50),
    "unstable": (0.50, 0.75),
    "rupture": (0.75, 1.0),
}


class ThesisPressureEngine:
    def __init__(self):
        self._pressure_history: Dict[str, List[float]] = defaultdict(list)
        self._symbol_pressure: Dict[str, float] = {}

    def compute(self, symbol: str, fracture_score: float,
                cohort_instability: float, alignment: float) -> float:
        pressure = (
            fracture_score * 0.45
            + cohort_instability * 0.35
            + (1.0 - alignment) * 0.20
        )
        pressure = max(0.0, min(1.0, pressure))
        self._pressure_history[symbol].append(pressure)
        self._symbol_pressure[symbol] = pressure
        band = self._band(pressure)
        logger.info(f"[THESIS_PRESSURE] {symbol} P={pressure:.3f} "
                    f"band={band} obs={len(self._pressure_history[symbol])}")
        return pressure

    def delta(self, symbol: str) -> float:
        history = self._pressure_history.get(symbol, [])
        if len(history) < 2:
            return 0.0
        delta_val = history[-1] - history[-2]
        logger.info(f"[PRESSURE_DELTA] {symbol} delta={delta_val:+.3f}")
        return delta_val

    def rupture_risk(self, symbol: str) -> bool:
        p = self._symbol_pressure.get(symbol, 0.0)
        d = self.delta(symbol)
        at_risk = p > 0.70 and d > 0.10
        if at_risk:
            logger.info(f"[RUPTURE_RISK] {symbol} ACTIVE P={p:.3f} delta={d:.3f}")
        return at_risk

    def _band(self, pressure: float) -> str:
        for name, (lo, hi) in PRESSURE_BANDS.items():
            if lo <= pressure < hi:
                return name
        return "rupture" if pressure >= 0.75 else "stable"

    def observations(self, symbol: Optional[str] = None) -> int:
        if symbol:
            return len(self._pressure_history.get(symbol, []))
        return sum(len(v) for v in self._pressure_history.values())

    def bands_seen(self, symbol: Optional[str] = None) -> List[str]:
        if symbol:
            history = self._pressure_history.get(symbol, [])
        else:
            history = [p for v in self._pressure_history.values() for p in v]
        return list(set(self._band(p) for p in history))

    def rising_events(self, symbol: Optional[str] = None) -> int:
        if symbol:
            history = self._pressure_history.get(symbol, [])
        else:
            history = [p for v in self._pressure_history.values() for p in v]
        count = 0
        for i in range(1, len(history)):
            if history[i] > history[i - 1]:
                count += 1
        return count

    def rupture_events(self, symbol: Optional[str] = None) -> int:
        if symbol:
            history = self._pressure_history.get(symbol, [])
        else:
            history = [p for v in self._pressure_history.values() for p in v]
        return sum(1 for p in history if p >= 0.75)

    def stats(self) -> dict:
        all_pressures = [p for v in self._pressure_history.values() for p in v]
        return {
            "symbols": len(self._pressure_history),
            "observations": len(all_pressures),
            "bands_seen": len(self.bands_seen()),
            "bands": sorted(self.bands_seen()),
            "rising_events": self.rising_events(),
            "rupture_events": self.rupture_events(),
        }
