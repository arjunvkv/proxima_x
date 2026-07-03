class DivergenceDetector:
    def __init__(self, expectations, reality):
        self._expectations = expectations
        self._reality = reality

    def compare(self, expected_key: str, observed_value: float, tolerance: float = 0.15) -> dict:
        expected = self._expectations.get(expected_key, 0.0)
        if observed_value is None or not isinstance(observed_value, (int, float)):
            return {
                "expected": expected,
                "observed": observed_value,
                "deviation_pct": 0.0,
                "within_tolerance": True,
                "alert": False}
        deviation = observed_value - expected
        deviation_pct = round(deviation / expected, 4) if expected != 0 else 0.0
        within_tolerance = abs(deviation_pct) <= tolerance
        return {
            "expected": expected,
            "observed": observed_value,
            "deviation_pct": deviation_pct,
            "within_tolerance": within_tolerance,
            "alert": not within_tolerance}

    def full_scan(self) -> dict:
        results = {}
        es_stats = self._reality.observed_es_rank_stats()
        at_stats = self._reality.observed_at_rank_stats()
        freq = self._reality.observed_frequency()

        obs_pp = self._reality.observed_pp()
        obs_sharpe = self._reality.observed_sharpe()
        obs_win_rate = self._reality.observed_win_rate()

        results["profit_percentage"] = self.compare("expected_pp", obs_pp)
        results["sharpe_ratio"] = self.compare("expected_sharpe", obs_sharpe)
        results["frequency"] = self.compare("expected_frequency", freq.get("signals_per_day", 0) * 30)
        
        mean_ret = obs_pp / max(freq.get("total_signals", 1), 1) if obs_pp is not None else None
        results["mean_return"] = self.compare("expected_mean_return", mean_ret)
        results["win_rate"] = self.compare("expected_win_rate", obs_win_rate)
        results["es_rank_mean"] = self.compare("expected_es_rank_mean", es_stats.get("mean", 0.0))
        results["at_rank_mean"] = self.compare("expected_at_rank_mean", at_stats.get("mean", 0.0))
        results["regime_accuracy"] = self.compare("expected_regime_accuracy", self._reality.observed_score().get("score", 0.0))

        alerts_list = [v for v in results.values() if v.get("alert")]
        dv = [abs(v["deviation_pct"]) for v in results.values() if isinstance(v, dict) and isinstance(v.get("deviation_pct"), (int, float))]
        results["divergence_score"] = round(sum(dv) / max(len(dv), 1), 4)
        results["alert_count"] = len(alerts_list)
        return results

    def alerts(self) -> list[dict]:
        scan = self.full_scan()
        return [{"metric": k, **v} for k, v in scan.items() if isinstance(v, dict) and v.get("alert")]

    def divergence_score(self) -> float:
        scan = self.full_scan()
        return float(scan.get("divergence_score", 0.0))
