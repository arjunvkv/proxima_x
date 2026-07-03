import logging
from typing import Optional

logger = logging.getLogger("proxima_ops.drl.blocked_vs_exec")


class BlockedVsExecuted:
    def __init__(self, freq_blocked_tracker, freq_executed_tracker):
        self._blocked = freq_blocked_tracker
        self._executed = freq_executed_tracker

    def compare(self, horizon: str = "return_h20") -> dict:
        blocked = [r for r in self._blocked.get_all() if r.get(horizon) is not None]
        executed = [r for r in self._executed.get_all() if r.get(horizon) is not None]
        if not blocked and not executed:
            return {"blocked_count": 0, "executed_count": 0}
        b_rets = [r[horizon] for r in blocked]
        e_rets = [r[horizon] for r in executed]
        b_pp = sum(1 for r in b_rets if r > 0) / len(b_rets) if b_rets else 0
        e_pp = sum(1 for r in e_rets if r > 0) / len(e_rets) if e_rets else 0
        b_mean = sum(b_rets) / len(b_rets) if b_rets else 0
        e_mean = sum(e_rets) / len(e_rets) if e_rets else 0
        b_sharpe = (b_mean / (sum((r - b_mean) ** 2 for r in b_rets) / len(b_rets)) ** 0.5
                    ) if len(b_rets) > 1 and sum((r - b_mean) ** 2 for r in b_rets) > 0 else 0
        e_sharpe = (e_mean / (sum((r - e_mean) ** 2 for r in e_rets) / len(e_rets)) ** 0.5
                    ) if len(e_rets) > 1 and sum((r - e_mean) ** 2 for r in e_rets) > 0 else 0
        return {
            "horizon": horizon,
            "blocked": {"count": len(b_rets), "pp": round(b_pp, 4), "mean_return": round(b_mean, 6), "sharpe": round(b_sharpe, 4)},
            "executed": {"count": len(e_rets), "pp": round(e_pp, 4), "mean_return": round(e_mean, 6), "sharpe": round(e_sharpe, 4)},
            "blocked_better": b_mean > e_mean}
