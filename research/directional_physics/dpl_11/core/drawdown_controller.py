import numpy as np


class DrawdownController:
    def __init__(self, max_drawdown=0.20, recovery_factor=0.5):
        self.max_drawdown = max_drawdown
        self.recovery_factor = recovery_factor
        self.peak = None
        self.dd_state = 0.0

    def reset(self):
        self.peak = None
        self.dd_state = 0.0

    def current_drawdown(self, equity):
        if self.peak is None:
            self.peak = equity
        self.peak = max(self.peak, equity)
        dd = (equity - self.peak) / (self.peak + 1e-8)
        return dd

    def exposure_multiplier(self, equity):
        dd = self.current_drawdown(equity)
        if dd > -0.01:
            self.dd_state = max(0.0, self.dd_state - 0.01)
        else:
            self.dd_state = min(1.0, self.dd_state + abs(dd) * 2.0)

        if dd < -self.max_drawdown:
            return 0.0
        if dd < -self.max_drawdown * 0.7:
            return 0.25
        if dd < -self.max_drawdown * 0.4:
            return 0.5
        if dd < -0.02:
            return 0.75
        return 1.0

    def equity_curve_smoother(self, daily_pnl, window=10):
        smoothed = np.zeros(len(daily_pnl))
        for i in range(len(daily_pnl)):
            smoothed[i] = float(np.sum(daily_pnl[max(0, i - window + 1):i + 1]))
        return smoothed
