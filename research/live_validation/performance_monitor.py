import numpy as np
from collections import deque

MIN_TRADES_FOR_STATS = 10


class PerformanceMonitor:
    def __init__(self):
        self._returns_7d: deque = deque(maxlen=7)
        self._returns_30d: deque = deque(maxlen=30)
        self._returns_90d: deque = deque(maxlen=90)
        self._daily_returns: dict[int, list[float]] = {}

    def record_trade(self, timestamp: int, pnl_pct: float):
        day = timestamp // 24
        if day not in self._daily_returns:
            self._daily_returns[day] = []
        self._daily_returns[day].append(pnl_pct)

    def _compute_rolling(self, window_days: int) -> dict:
        days = sorted(self._daily_returns.keys())
        if len(days) < 2:
            return {"sharpe": None, "pp": None, "return": 0.0, "dd": None, "n": 0}
        recent_days = days[-min(window_days, len(days)):]
        rets = []
        for d in recent_days:
            rets.extend(self._daily_returns[d])
        arr = np.array(rets)
        if len(arr) < MIN_TRADES_FOR_STATS:
            return {"sharpe": None, "pp": None, "return": float(np.sum(arr)) if len(arr) > 0 else 0.0, "dd": None, "n": len(arr)}
        std = np.std(arr)
        if std < 1e-8:
            sharpe = None
        else:
            sharpe = float(np.mean(arr) / std * np.sqrt(252))
        pp = float(np.mean(arr > 0))
        total_ret = float(np.sum(arr))
        cum = np.cumprod(1 + arr)
        running_max = np.maximum.accumulate(cum)
        dd = float(abs(np.min((cum - running_max) / running_max)))
        return {"sharpe": round(sharpe, 3) if sharpe is not None else None,
                "pp": round(pp, 3) if pp is not None else None,
                "return": round(total_ret, 4),
                "dd": round(dd, 4) if dd is not None else None,
                "n": len(arr)}

    @property
    def last_7d(self) -> dict:
        return self._compute_rolling(7)

    @property
    def last_30d(self) -> dict:
        return self._compute_rolling(30)

    @property
    def last_90d(self) -> dict:
        return self._compute_rolling(90)

    @property
    def classification(self) -> str:
        d7 = self.last_7d
        d30 = self.last_30d
        if d7["n"] < MIN_TRADES_FOR_STATS:
            return "INSUFFICIENT_DATA"
        if d7["sharpe"] is None or d30["sharpe"] is None or d7["dd"] is None:
            return "INSUFFICIENT_DATA"
        if d7["sharpe"] > 0.5 and d30["sharpe"] > 0.3 and d7["dd"] < 0.05:
            return "HEALTHY"
        if d7["sharpe"] > -0.5 and d30["sharpe"] > -0.3 and d7["dd"] < 0.10:
            return "WEAKENING"
        return "DEGRADED"

    def summary(self) -> dict:
        return {
            "last_7d": self.last_7d,
            "last_30d": self.last_30d,
            "last_90d": self.last_90d,
            "classification": self.classification}
