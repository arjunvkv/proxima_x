from typing import Dict, Any


class ConsensusDriftMonitor:
    """
    CDM — Consensus Drift Monitor

    Detects divergence between CAL and TCA signals.
    """

    def __init__(self,
                 drift_threshold: float = 0.75):
        self.drift_threshold = drift_threshold

    def compute_drift(self,
                       cal_report: Dict[str, Dict[str, float]],
                       tca_report: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        drift_scores = {}

        symbols = set(cal_report.keys()) | set(tca_report.keys())

        for sym in symbols:
            cal = cal_report.get(sym, {})
            tca = tca_report.get(sym, {})

            ecdf_drift = abs(cal.get("ecdf_contrib", 0.0) - tca.get("ecdf", 0.0))
            entropy_drift = abs(cal.get("entropy_contrib", 0.0) - tca.get("entropy", 0.0))
            signal_drift = abs(cal.get("signal_contrib", 0.0) - tca.get("signal", 0.0))

            drift = (ecdf_drift + entropy_drift + signal_drift) / 3.0
            drift_scores[sym] = drift

        return drift_scores

    def is_stable(self, drift_scores: Dict[str, float]) -> bool:
        for v in drift_scores.values():
            if v > self.drift_threshold:
                return False
        return True
