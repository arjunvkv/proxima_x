import numpy as np
from scipy.stats import pearsonr


class ResidualMonitor:
    """Compares ES vs Residual predictive power on realized trades.

    Tracks the correlation between each signal value at entry and the
    trade outcome. Answers: 'Is residual still beating ES?'
    """

    def __init__(self):
        self._es_values: list[float] = []
        self._residual_values: list[float] = []
        self._pnls: list[float] = []
        self._daily_es_pp: list[float] = []
        self._daily_res_pp: list[float] = []
        self._daily_labels: list[int] = []

    def record_trade(self, timestamp: int, es_value: float,
                     residual_value: float, pnl_pct: float):
        day = timestamp // 24
        self._es_values.append(es_value)
        self._residual_values.append(residual_value)
        self._pnls.append(pnl_pct)
        if day != (self._daily_labels[-1] if self._daily_labels else -1):
            self._daily_labels.append(day)
            self._daily_es_pp.append(0.0)
            self._daily_res_pp.append(0.0)

    @property
    def es_predictive_power(self) -> float:
        if len(self._es_values) < 10:
            return 0.0
        corr, _ = pearsonr(self._es_values, self._pnls)
        return float(np.nan_to_num(corr, nan=0.0))

    @property
    def residual_predictive_power(self) -> float:
        if len(self._residual_values) < 10:
            return 0.0
        corr, _ = pearsonr(self._residual_values, self._pnls)
        return float(np.nan_to_num(corr, nan=0.0))

    @property
    def ratio(self) -> float:
        """Ratio of Residual/ES predictive power. > 1 means Residual has stronger signal."""
        es_p = abs(self.es_predictive_power)
        res_p = abs(self.residual_predictive_power)
        if es_p < 1e-10:
            return 0.0 if res_p < 1e-10 else 999.0
        return res_p / es_p

    @property
    def residual_beats_es(self) -> bool:
        """Residual beats ES if it has stronger positive (or less negative) correlation with P&L."""
        return self.residual_predictive_power >= self.es_predictive_power

    @property
    def es_sharpe(self) -> float:
        """Hypothetical Sharpe from ES-based position sizing."""
        if len(self._pnls) < 5:
            return 0.0
        weights = np.array([max(v, 0.0) for v in self._es_values])
        weighted_rets = np.array(self._pnls) * (weights / max(np.mean(weights), 1e-10))
        if np.std(weighted_rets) < 1e-10:
            return 0.0
        return float(np.mean(weighted_rets) / np.std(weighted_rets) * np.sqrt(252))

    @property
    def residual_sharpe(self) -> float:
        """Hypothetical Sharpe from Residual-based position sizing."""
        if len(self._pnls) < 5:
            return 0.0
        res_arr = np.array(self._residual_values)
        r_min, r_max = float(np.min(res_arr)), float(np.max(res_arr))
        weights = np.array([(v - r_min) / max(r_max - r_min, 1e-10) for v in self._residual_values])
        weighted_rets = np.array(self._pnls) * (weights / max(np.mean(weights), 1e-10))
        if np.std(weighted_rets) < 1e-10:
            return 0.0
        return float(np.mean(weighted_rets) / np.std(weighted_rets) * np.sqrt(252))

    def summary(self) -> dict:
        n = len(self._pnls)
        return {
            "n_trades": n,
            "es_predictive_power": round(self.es_predictive_power, 3),
            "residual_predictive_power": round(self.residual_predictive_power, 3),
            "ratio": round(self.ratio, 3),
            "residual_beats_es": self.residual_beats_es,
            "es_hypothetical_sharpe": round(self.es_sharpe, 3),
            "residual_hypothetical_sharpe": round(self.residual_sharpe, 3),
            "mean_es_value": round(float(np.mean(self._es_values)), 3) if n > 0 else 0.0,
            "mean_residual_value": round(float(np.mean(self._residual_values)), 3) if n > 0 else 0.0}
