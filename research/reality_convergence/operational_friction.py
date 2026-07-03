class OperationalFriction:
    def __init__(self, drl_module, freq_reality, reality_engine):
        self._drl = drl_module
        self._freq = freq_reality
        self._real = reality_engine

    def spread_cost(self) -> float:
        s = 0
        if self._drl:
            s = self._drl.get("mean_slippage_pts", 0) if isinstance(self._drl, dict) else 0
        return min(s / 10.0, 1.0)

    def latency_cost(self) -> float:
        lat = self._real.observed_latency()
        return min(lat / 1000.0, 1.0)

    def blocked_signal_cost(self) -> float:
        blocked = 0
        if self._freq:
            s = self._freq.summary() if hasattr(self._freq, 'summary') else {}
            blocked = s.get("total", 0) if isinstance(s, dict) else 0
        executed = self._real.observed_frequency().get("total_signals", 0)
        total = blocked + executed
        return blocked / total if total > 0 else 0.0

    def missed_opportunity_cost(self) -> float:
        freq = self._real.observed_frequency()
        total = freq.get("total_signals", 0)
        exp = self._real._days_elapsed() * 1.0
        return max(0.0, 1.0 - (total / max(exp, 1)))

    def friction_index(self) -> float:
        costs = [self.spread_cost(), self.latency_cost(),
                 self.blocked_signal_cost(), self.missed_opportunity_cost()]
        return round(sum(costs) / len(costs), 3)

    def summary(self) -> dict:
        return {
            "spread_cost": round(self.spread_cost(), 3),
            "latency_cost": round(self.latency_cost(), 3),
            "blocked_signal_cost": round(self.blocked_signal_cost(), 3),
            "missed_opportunity_cost": round(self.missed_opportunity_cost(), 3),
            "friction_index": self.friction_index()}
