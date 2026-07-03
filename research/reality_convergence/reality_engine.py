class RealityEngine:
    def __init__(self, perf_monitor=None, deployment_score=None, signal_funnel=None,
                 freq_reality=None, executed_reality=None, drl_module=None, mt5_monitor=None):
        self._perf = perf_monitor
        self._score = deployment_score
        self._funnel = signal_funnel
        self._freq = freq_reality
        self._exec = executed_reality
        self._drl = drl_module
        self._mt5 = mt5_monitor
        self._start_time = None

    @property
    def n_trades(self) -> int:
        return self._perf_summary().get("n_trades", 0)

    def _perf_summary(self) -> dict:
        if self._perf is not None and hasattr(self._perf, 'summary'):
            return self._perf.summary()
        return {}

    def observed_pp(self):
        val = self._perf_summary().get("pp")
        if val is None or self.n_trades < 10:
            return None
        return val

    def observed_sharpe(self):
        val = self._perf_summary().get("sharpe")
        if val is None or self.n_trades < 10:
            return None
        return val

    def observed_frequency(self) -> dict:
        blocked = 0
        executed = 0
        if self._freq is not None and hasattr(self._freq, 'count'):
            blocked = self._freq.count()
        if self._exec is not None and hasattr(self._exec, 'count'):
            executed = self._exec.count()
        total = blocked + executed
        days = self._days_elapsed()
        return {"total_signals": total, "blocked_signals": blocked,
                "executed_signals": executed, "days_elapsed": days,
                "signals_per_day": round(total / days, 2) if days > 0 else 0.0}

    def observed_win_rate(self) -> float:
        s = self._perf_summary()
        n = s.get("n_trades", 0)
        return s.get("pp", 0.0) if n > 0 else 0.0

    def _collect_ranks(self, field: str) -> list:
        data = []
        if self._freq is not None and hasattr(self._freq, 'get_all'):
            data.extend(r.get(field, 0) for r in self._freq.get_all() if r.get(field) is not None)
        if self._exec is not None and hasattr(self._exec, 'get_all'):
            data.extend(r.get(field, 0) for r in self._exec.get_all() if r.get(field) is not None)
        return data

    def observed_es_rank_stats(self) -> dict:
        data = self._collect_ranks("es_rank")
        if not data:
            return {"mean": 0.0, "std": 0.0, "count": 0}
        mu = sum(data) / len(data)
        var = sum((x - mu) ** 2 for x in data) / len(data)
        return {"mean": round(mu, 4), "std": round(var ** 0.5, 4), "count": len(data)}

    def observed_at_rank_stats(self) -> dict:
        data = self._collect_ranks("at_rank")
        if not data:
            return {"mean": 0.0, "std": 0.0, "count": 0}
        mu = sum(data) / len(data)
        var = sum((x - mu) ** 2 for x in data) / len(data)
        return {"mean": round(mu, 4), "std": round(var ** 0.5, 4), "count": len(data)}

    def observed_score(self) -> dict:
        if self._score is not None and hasattr(self._score, 'summary'):
            return self._score.summary()
        return {"current_score": 0.0, "classification": "NONE"}

    def observed_execution_quality(self) -> str:
        if self._drl is not None and hasattr(self._drl, 'summary'):
            return self._drl.summary().get("classification", "unknown")
        return "unknown"

    def observed_latency(self) -> float:
        if self._drl is not None and hasattr(self._drl, 'summary'):
            return self._drl.summary().get("mean_latency_ms", 0.0)
        return 0.0

    def set_start_time(self, t: float):
        self._start_time = t

    def _days_elapsed(self) -> int:
        if self._start_time:
            import time
            return max(1, int((time.time() - self._start_time) / 86400))
        return 1

    def all_observed(self) -> dict:
        return {
            "pp": self.observed_pp(),
            "sharpe": self.observed_sharpe(),
            "frequency": self.observed_frequency(),
            "win_rate": self.observed_win_rate(),
            "es_rank_stats": self.observed_es_rank_stats(),
            "at_rank_stats": self.observed_at_rank_stats(),
            "score": self.observed_score(),
            "execution_quality": self.observed_execution_quality(),
            "latency": self.observed_latency()}
