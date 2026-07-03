MIN_TRADES_FOR_FREQ = 10
MIN_DAYS_FOR_ANNUALIZATION = 2


class ConvergenceTracker:
    def __init__(self, expectation_engine, reality_engine):
        self._exp = expectation_engine
        self._real = reality_engine
        self._history: list[dict] = []

    def check(self) -> dict:
        n_trades = self._real.n_trades
        obs = self._real.observed_frequency()
        days = obs.get("days_elapsed", 0)
        total = obs.get("total_signals", 0)
        exp_freq = self._exp.get("expected_frequency", 30)

        if n_trades < MIN_TRADES_FOR_FREQ or days < MIN_DAYS_FOR_ANNUALIZATION:
            return {
                "expected_monthly": exp_freq,
                "observed_monthly": "COLLECTING_DATA",
                "total_signals": total,
                "days_elapsed": days,
                "frequency_error": "COLLECTING_DATA",
                "frequency_error_pct": 0.0,
                "frequency_stable": "COLLECTING_DATA",
                "frequency_drift": "COLLECTING_DATA"
            }
        obs_monthly = obs.get("signals_per_day", 0) * 30
        error = obs_monthly - exp_freq if exp_freq > 0 else 0
        error_pct = round(error / exp_freq, 4) if exp_freq > 0 else 0
        stable = abs(error_pct) < 0.20
        result = {
            "expected_monthly": exp_freq,
            "observed_monthly": round(obs_monthly, 1),
            "total_signals": total,
            "days_elapsed": days,
            "frequency_error": round(error, 1),
            "frequency_error_pct": error_pct,
            "frequency_stable": stable,
            "frequency_drift": self._drift()}
        self._history.append(result)
        return result

    def _drift(self) -> str:
        if len(self._history) < 3:
            return "INSUFFICIENT_DATA"
        recent = self._history[-3:]
        errors = [r.get("frequency_error_pct", 0.0) for r in recent if isinstance(r.get("frequency_error_pct"), (int, float))]
        if len(errors) < 3:
            return "INSUFFICIENT_DATA"
        if all(abs(e) < 0.10 for e in errors):
            return "STABLE"
        if all(e > errors[0] for e in errors[1:]) or all(e < errors[0] for e in errors[1:]):
            return "DRIFTING"
        return "OSCILLATING"

    def match_pct(self):
        n_trades = self._real.n_trades
        obs = self._real.observed_frequency()
        days = obs.get("days_elapsed", 0)
        if n_trades < MIN_TRADES_FOR_FREQ or days < MIN_DAYS_FOR_ANNUALIZATION:
            return "COLLECTING_DATA"
        c = self.check()
        err = c.get("frequency_error_pct", 0)
        if not isinstance(err, (int, float)):
            return "COLLECTING_DATA"
        return round((1 - abs(err)) * 100, 1)
