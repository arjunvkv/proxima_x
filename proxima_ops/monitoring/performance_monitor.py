import numpy as np
from collections import deque
from datetime import datetime
from proxima_ops.config.settings import SETTINGS


MIN_TRADES_FOR_STATS = 10
MIN_TRADES_FOR_CLASSIFICATION = 25
MIN_TRADES_FOR_CONFIDENCE = 50
MIN_TRADES_FOR_DD = 5


class OpsPerformanceMonitor:
    def __init__(self):
        self._daily_pnls: dict[str, list[float]] = {}
        self._pnls: list[float] = []
        self._points: list[float] = []
        self._dates: list[str] = []
        self._hold_durations: list[int] = []
        self._rolling_30d: deque = deque(maxlen=30)
        self._rolling_7d: deque = deque(maxlen=7)

    def record_trade(self, pnl_money: float, pnl_points: float, date_str: str, hold_bars: int = 0):
        self._pnls.append(pnl_money)
        self._points.append(pnl_points)
        self._dates.append(date_str)
        self._hold_durations.append(hold_bars)
        if date_str not in self._daily_pnls:
            self._daily_pnls[date_str] = []
        self._daily_pnls[date_str].append(pnl_money)

    @property
    def sharpe(self):
        arr = np.array(self._pnls)
        if len(arr) < MIN_TRADES_FOR_STATS:
            return None
        std = np.std(arr)
        if std < 1e-8:
            return None
        return float(np.mean(arr) / std * np.sqrt(252))

    @property
    def pp(self):
        arr = np.array(self._pnls)
        if len(arr) < MIN_TRADES_FOR_STATS:
            return None
        return float(np.mean(arr > 0))

    @property
    def total_return(self) -> float:
        return float(np.sum(self._pnls))

    @property
    def total_points(self) -> float:
        return float(np.sum(self._points))

    @property
    def max_dd(self):
        arr = np.array([float(x) for x in self._pnls])
        if len(arr) < MIN_TRADES_FOR_DD:
            return None
        cum = np.cumsum(arr)
        running_max = np.maximum.accumulate(cum)
        denominator = np.maximum(np.abs(running_max), 1e-12)
        dd = (cum - running_max) / denominator
        return float(min(abs(np.min(dd)), 1.0))

    @property
    def n_trades(self) -> int:
        return len(self._pnls)

    @property
    def today_pnl(self) -> float:
        today = datetime.now().strftime("%Y-%m-%d")
        return float(np.sum(self._daily_pnls.get(today, [0.0])))

    @property
    def avg_hold_bars(self) -> float:
        if not self._hold_durations:
            return 0.0
        return float(np.mean(self._hold_durations))

    def summary(self) -> dict:
        return {
            "sharpe": round(self.sharpe, 3) if self.sharpe is not None else None,
            "pp": round(self.pp, 3) if self.pp is not None else None,
            "total_return": round(self.total_return, 2),
            "total_points": round(self.total_points, 1),
            "max_dd": round(self.max_dd, 4) if self.max_dd is not None else None,
            "n_trades": self.n_trades,
            "today_pnl": round(self.today_pnl, 2),
            "avg_hold_bars": round(self.avg_hold_bars, 1)}
