from typing import Dict, Any


class CausalWeightFusion:
    """
    CWF — Causal Weight Fusion Layer

    Merges CAL (instant) + TCA (temporal) into unified learning signal.
    """

    def __init__(self,
                 cal_weight: float = 0.5,
                 tca_weight: float = 0.5):
        self.cal_weight = cal_weight
        self.tca_weight = tca_weight

    def fuse(self,
             cal_report: Dict[str, Dict[str, float]],
             tca_report: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:

        fused = {}
        all_symbols = set(cal_report.keys()) | set(tca_report.keys())

        for sym in all_symbols:
            cal = cal_report.get(sym, {})
            tca = tca_report.get(sym, {})

            fused[sym] = {
                "ecdf": self._combine(cal.get("ecdf_contrib", 0.0), tca.get("ecdf", 0.0)),
                "entropy": self._combine(cal.get("entropy_contrib", 0.0), tca.get("entropy", 0.0)),
                "spread": self._combine(cal.get("spread_contrib", 0.0), 0.0),
                "signal": self._combine(cal.get("signal_contrib", 0.0), tca.get("signal", 0.0)),
            }

        return fused

    def fuse_with_weights(self,
                          cal_report: Dict[str, Dict[str, float]],
                          tca_report: Dict[str, Dict[str, float]],
                          cal_w: float,
                          tca_w: float) -> Dict[str, Dict[str, float]]:
        self.cal_weight = cal_w
        self.tca_weight = tca_w
        return self.fuse(cal_report, tca_report)

    def _combine(self, cal_val: float, tca_val: float) -> float:
        return (self.cal_weight * cal_val) + (self.tca_weight * tca_val)
