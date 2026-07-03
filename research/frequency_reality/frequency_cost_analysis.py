import numpy as np
from typing import Optional


class FrequencyCostAnalysis:
    def __init__(self, blocked_tracker, executed_tracker):
        self._blocked = blocked_tracker
        self._executed = executed_tracker

    def _returns_for(self, records: list[dict], horizon_tag: str = "h20") -> list[float]:
        key = f"return_{horizon_tag}"
        return [r[key] for r in records if r.get(key) is not None]

    def _pp_for(self, records: list[dict], horizon_tag: str = "h20") -> float:
        rets = self._returns_for(records, horizon_tag)
        if not rets:
            return 0.0
        return sum(1 for r in rets if r > 0) / len(rets)

    def _mean_return(self, records: list[dict], horizon_tag: str = "h20") -> float:
        rets = self._returns_for(records, horizon_tag)
        return float(np.mean(rets)) if rets else 0.0

    def _sharpe(self, records: list[dict], horizon_tag: str = "h20") -> float:
        rets = self._returns_for(records, horizon_tag)
        if len(rets) < 2:
            return 0.0
        mean_r = float(np.mean(rets))
        std_r = float(np.std(rets, ddof=1))
        return mean_r / std_r if std_r > 0 else 0.0

    def _max_dd(self, records: list[dict], horizon_tag: str = "h20") -> float:
        prices = self._returns_for(records, horizon_tag)
        if not prices:
            return 0.0
        cum = np.cumsum(prices)
        peak = np.maximum.accumulate(cum)
        dd = cum - peak
        return float(abs(np.min(dd)))

    # RQ3
    def opportunity_cost(self, horizon_tag: str = "h20") -> dict:
        blocked = self._blocked.get_all()
        executed = self._executed.get_all()
        blocked_rets = self._returns_for(blocked, horizon_tag)
        executed_rets = self._returns_for(executed, horizon_tag)
        return {
            "horizon": horizon_tag,
            "blocked": {
                "count": len(blocked_rets),
                "mean_return": self._mean_return(blocked, horizon_tag),
                "pp": self._pp_for(blocked, horizon_tag),
                "sharpe": self._sharpe(blocked, horizon_tag),
                "max_dd": self._max_dd(blocked, horizon_tag)},
            "executed": {
                "count": len(executed_rets),
                "mean_return": self._mean_return(executed, horizon_tag),
                "pp": self._pp_for(executed, horizon_tag),
                "sharpe": self._sharpe(executed, horizon_tag),
                "max_dd": self._max_dd(executed, horizon_tag)}}

    # RQ4
    def extreme_analysis(self, es_min: float = 0.95, at_min: float = 0.95) -> dict:
        blocked_extreme = self._blocked.filter(es_min=es_min, at_min=at_min)
        executed_extreme = self._executed.get_all()
        executed_extreme = [r for r in executed_extreme
                           if r.get("es_rank", 0) >= es_min and r.get("at_rank", 0) >= at_min]
        result = {}
        for h in ["h20", "h50", "h100"]:
            result[h] = {
                "blocked_count": len(self._returns_for(blocked_extreme, h)),
                "blocked_mean_return": self._mean_return(blocked_extreme, h),
                "blocked_pp": self._pp_for(blocked_extreme, h),
                "executed_count": len(self._returns_for(executed_extreme, h)),
                "executed_mean_return": self._mean_return(executed_extreme, h),
                "executed_pp": self._pp_for(executed_extreme, h)}
        return result

    # RQ6 — only counts resolved signals
    def alpha_destruction_ratio(self, horizon_tag: str = "h20") -> float:
        blocked = [r for r in self._blocked.get_all()
                   if r.get("block_reason") in ("FREQUENCY_FILTER", "INVALID_SPREAD")
                   and r.get("future_resolved")]
        all_signals = [r for r in self._blocked.get_all() + self._executed.get_all()
                       if r.get("future_resolved")]
        blocked_alpha = sum(max(0, r.get(f"return_{horizon_tag}", 0)) for r in blocked)
        total_alpha = sum(max(0, r.get(f"return_{horizon_tag}", 0)) for r in all_signals)
        if total_alpha == 0:
            return 0.0
        return min(blocked_alpha / total_alpha, 1.0)

    # RQ7
    def asset_level_impact(self, horizon_tag: str = "h20") -> dict:
        symbols = set()
        for r in self._blocked.get_all() + self._executed.get_all():
            symbols.add(r.get("symbol"))
        result = {}
        for sym in sorted(symbols):
            blocked_sym = [r for r in self._blocked.get_all() if r.get("symbol") == sym and r.get("block_reason") in ("FREQUENCY_FILTER", "INVALID_SPREAD")]
            executed_sym = [r for r in self._executed.get_all() if r.get("symbol") == sym]
            result[sym] = {
                "blocked_count": len(blocked_sym),
                "blocked_mean_return": self._mean_return(blocked_sym, horizon_tag),
                "blocked_pp": self._pp_for(blocked_sym, horizon_tag),
                "executed_count": len(executed_sym),
                "executed_mean_return": self._mean_return(executed_sym, horizon_tag),
                "executed_pp": self._pp_for(executed_sym, horizon_tag),
                "blocked_better": self._mean_return(blocked_sym, horizon_tag) > self._mean_return(executed_sym, horizon_tag)}
        return result

    # RQ8
    def regime_impact(self, horizon_tag: str = "h20") -> dict:
        regimes = set()
        for r in self._blocked.get_all() + self._executed.get_all():
            reg = r.get("regime", "UNKNOWN")
            regimes.add(reg)
        result = {}
        for reg in sorted(regimes):
            blocked_reg = [r for r in self._blocked.get_all()
                           if r.get("regime") == reg and r.get("block_reason") in ("FREQUENCY_FILTER", "INVALID_SPREAD")]
            executed_reg = [r for r in self._executed.get_all() if r.get("regime") == reg]
            result[reg] = {
                "blocked_count": len(blocked_reg),
                "blocked_mean_return": self._mean_return(blocked_reg, horizon_tag),
                "executed_count": len(executed_reg),
                "executed_mean_return": self._mean_return(executed_reg, horizon_tag)}
        return result

    # RQ5
    def controller_simulation(self, blocked_records: list[dict]) -> dict:
        controller_on = self._executed.get_all() if self._executed.count() > 0 else []
        controller_off = blocked_records + controller_on
        on_pp = self._pp_for(controller_on)
        on_sharpe = self._sharpe(controller_on)
        off_pp = self._pp_for(controller_off)
        off_sharpe = self._sharpe(controller_off)
        return {
            "controller_on": {"count": len(controller_on), "pp": on_pp, "sharpe": on_sharpe},
            "controller_off": {"count": len(controller_off), "pp": off_pp, "sharpe": off_sharpe},
            "improvement": {"pp_delta": off_pp - on_pp, "sharpe_delta": off_sharpe - on_sharpe}}

    # RQ9 — only counts resolved (future_resolved=True) signals
    def leakage_rate(self) -> dict:
        blocked = [r for r in self._blocked.get_all()
                   if r.get("block_reason") in ("FREQUENCY_FILTER", "INVALID_SPREAD")
                   and r.get("future_resolved")]
        total = len(blocked)
        profitable = sum(1 for r in blocked if r.get("return_h20", 0) > 0)
        return {
            "blocked_total": total,
            "blocked_profitable": profitable,
            "leakage_rate": round(profitable / total * 100, 1) if total > 0 else 0.0}
