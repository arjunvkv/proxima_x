class DeploymentHealth:
    def __init__(self, deployment_score=None, alpha_transfer=None, friction=None, convergence=None, divergence=None):
        self._score = deployment_score
        self._ate = alpha_transfer
        self._friction = friction
        self._conv = convergence
        self._div = divergence

    def compute(self) -> dict:
        n_trades = 0
        if self._ate and hasattr(self._ate, '_real'):
            n_trades = self._ate._real.n_trades
        elif self._conv and hasattr(self._conv, '_real'):
            n_trades = self._conv._real.n_trades

        if n_trades < 10:
            return {"health_index": "COLLECTING_DATA", "classification": "COLLECTING_DATA"}

        h = 50.0
        if self._ate:
            ate_val = self._ate.ate()
            if isinstance(ate_val, (int, float)):
                h += ate_val * 20
        if self._friction:
            h -= self._friction.friction_index() * 15
        if self._conv:
            match_val = self._conv.match_pct()
            if isinstance(match_val, (int, float)):
                h += (match_val / 100.0) * 10
        if self._div:
            h -= self._div.divergence_score() * 15
        if self._score:
            s = self._score.summary() if hasattr(self._score, 'summary') else {}
            h += (s.get("current_score", 0) - 0.5) * 10
        h = max(0.0, min(100.0, h))
        if h >= 80:
            cls = "HEALTHY"
        elif h >= 60:
            cls = "WARNING"
        elif h >= 40:
            cls = "DEGRADED"
        else:
            cls = "CRITICAL"
        return {"health_index": round(h, 1), "classification": cls}

    def summary(self) -> dict:
        return self.compute()
