from typing import Dict, Any


class ExecutionMapper:
    """
    V3/V4 Execution Synthesis Layer

    Converts allocation weights into executable trade instructions.
    """

    def __init__(self,
                 base_lot: float = 1.0,
                 max_lot: float = 5.0,
                 min_lot: float = 0.01,
                 risk_per_unit: float = 1000.0):
        self.base_lot = base_lot
        self.max_lot = max_lot
        self.min_lot = min_lot
        self.risk_per_unit = risk_per_unit

    def _lot_size(self, weight: float) -> float:
        raw = self.base_lot * weight * self.risk_per_unit / 1000.0
        if raw > self.max_lot:
            raw = self.max_lot
        if raw < self.min_lot:
            raw = self.min_lot
        return round(raw, 2)

    def _direction(self, sym_data: Dict[str, Any]) -> str:
        sig = sym_data.get("prod_signal", sym_data.get("signal", 0))
        if sig == 1:
            return "BUY"
        elif sig == -1:
            return "SELL"
        return "FLAT"

    def _entry_price_bias(self, sym_data: Dict[str, Any]) -> float:
        spread = float(sym_data.get("spread") or 0.0)
        return spread * 0.5

    def map(self,
            allocations: Dict[str, float],
            eval_data: Dict[str, Dict]) -> Dict[str, Dict]:

        execution_plan = {}
        for sym, weight in allocations.items():
            data = eval_data.get(sym, {})
            execution_plan[sym] = {
                "symbol": sym,
                "direction": self._direction(data),
                "lot": self._lot_size(weight),
                "weight": weight,
                "entry_bias": self._entry_price_bias(data),
                "ecdf": data.get("ecdf_rank", 0.5),
                "entropy": data.get("entropy", 0.5),
                "spread": data.get("spread", 0.0),
            }
        return execution_plan
