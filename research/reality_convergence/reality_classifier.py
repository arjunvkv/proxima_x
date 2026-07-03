class RealityClassifier:
    def __init__(self, alpha_transfer=None, divergence=None, friction=None, health=None, anomaly=None):
        self._ate = alpha_transfer
        self._div = divergence
        self._friction = friction
        self._health = health
        self._anomaly = anomaly

    def classify(self) -> dict:
        n_trades = 0
        if self._ate and hasattr(self._ate, '_real'):
            n_trades = self._ate._real.n_trades
        elif self._health and hasattr(self._health, '_ate') and hasattr(self._health._ate, '_real'):
            n_trades = self._health._ate._real.n_trades

        if n_trades < 10:
            return {
                "classification": "COLLECTING_EVIDENCE",
                "confidence": "COLLECTING_DATA",
                "ate": "COLLECTING_DATA",
                "divergence_score": "COLLECTING_DATA",
                "friction_index": "COLLECTING_DATA",
                "health_status": "COLLECTING_DATA",
                "anomaly_severity": "COLLECTING_DATA"
            }

        ate = self._ate.ate() if self._ate else 0.0
        div_score = self._div.divergence_score() if self._div else 0.5
        fi = self._friction.friction_index() if self._friction else 0.5
        health = self._health.compute() if self._health else {"classification": "UNKNOWN"}
        anomaly = self._anomaly.check() if self._anomaly else {"severity": "NORMAL"}

        if n_trades < 25:
            if isinstance(ate, (int, float)) and ate >= 0.60 and div_score < 0.35 and fi < 0.35:
                cls = "RESEARCH_CONVERGING"
            elif isinstance(ate, (int, float)) and ate >= 0.40 and anomaly.get("severity") != "CRITICAL":
                cls = "OPERATIONALLY_DEGRADED"
            elif isinstance(ate, (int, float)) and ate >= 0.20:
                cls = "COLLECTING_EVIDENCE"
            else:
                cls = "COLLECTING_EVIDENCE"
        else:
            if isinstance(ate, (int, float)) and ate >= 0.75 and div_score < 0.20 and fi < 0.20 and health.get("classification") in ("HEALTHY",):
                cls = "RESEARCH_CONVERGING"
            elif isinstance(ate, (int, float)) and ate >= 0.60 and div_score < 0.35 and fi < 0.35 and health.get("classification") in ("HEALTHY", "WARNING"):
                cls = "RESEARCH_CONVERGING"
            elif isinstance(ate, (int, float)) and ate >= 0.40 and anomaly.get("severity") != "CRITICAL":
                cls = "OPERATIONALLY_DEGRADED"
            elif isinstance(ate, (int, float)) and ate >= 0.20:
                cls = "COLLECTING_EVIDENCE"
            else:
                cls = "COLLECTING_EVIDENCE"

        confidence = round(max(0.0, 1.0 - (div_score + fi) / 2.0), 3)
        return {
            "classification": cls,
            "confidence": confidence,
            "ate": ate,
            "divergence_score": div_score,
            "friction_index": fi,
            "health_status": health.get("classification", "UNKNOWN"),
            "anomaly_severity": anomaly.get("severity", "NORMAL")}
