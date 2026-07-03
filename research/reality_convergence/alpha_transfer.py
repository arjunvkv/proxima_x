class AlphaTransfer:
    def __init__(self, expectation_engine, reality_engine):
        self._exp = expectation_engine
        self._real = reality_engine

    def ate(self):
        if self._real.n_trades < 10:
            return "COLLECTING_DATA"
        exp_sharpe = self._exp.get("expected_sharpe", 1.38)
        if exp_sharpe is None or exp_sharpe <= 0:
            return 0.0
        obs_sharpe = self._real.observed_sharpe()
        if obs_sharpe is None:
            return 0.0
        raw = float(obs_sharpe) / float(exp_sharpe)
        return round(max(0.0, min(raw, 1.0)), 3)

    def classification(self) -> str:
        if self._real.n_trades < 10:
            return "COLLECTING_EVIDENCE"
        v = self.ate()
        if v is None or v == "COLLECTING_DATA":
            return "COLLECTING_EVIDENCE"
        if v > 0.90:
            return "EXCELLENT"
        elif v >= 0.75:
            return "STRONG"
        elif v >= 0.60:
            return "HEALTHY"
        elif v >= 0.40:
            return "DEGRADED"
        else:
            return "FAILURE"

    def pp_transfer(self):
        if self._real.n_trades < 10:
            return "COLLECTING_DATA"
        exp_pp = self._exp.get("expected_pp", 0.59)
        if exp_pp is None or exp_pp <= 0:
            return 0.0
        obs_pp = self._real.observed_pp()
        if obs_pp is None:
            return 0.0
        return round(max(0.0, min(float(obs_pp) / float(exp_pp), 1.0)), 3)

    def summary(self) -> dict:
        return {
            "ate": self.ate(),
            "ate_classification": self.classification(),
            "pp_transfer": self.pp_transfer(),
            "expected_sharpe": self._exp.get("expected_sharpe"),
            "observed_sharpe": self._real.observed_sharpe(),
            "expected_pp": self._exp.get("expected_pp"),
            "observed_pp": self._real.observed_pp()}
