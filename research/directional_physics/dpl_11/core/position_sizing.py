import numpy as np

def rolling_vol(returns, window=20):
    out = np.ones(len(returns))
    for i in range(window, len(returns)):
        out[i] = float(np.std(returns[i - window:i]))
    return np.clip(out, 1e-4, None)


def rolling_sharpe(returns, window=40):
    out = np.zeros(len(returns))
    for i in range(window, len(returns)):
        r = returns[i - window:i]
        mu = np.mean(r)
        sd = np.std(r) + 1e-8
        out[i] = mu / sd * np.sqrt(252)
    return out


def rolling_da(signals, returns, window=60):
    out = np.ones(len(signals)) * 0.5
    for i in range(window, len(signals)):
        correct = np.mean((signals[i - window:i] > 0) == (returns[i - window:i] > 0))
        out[i] = correct
    return out


class PositionSizer:
    def __init__(self, base_risk=0.02, vol_target=0.15):
        self.base_risk = base_risk
        self.vol_target = vol_target

    def volatility_scaler(self, returns_series, window=20):
        vol = rolling_vol(returns_series, window)
        scale = self.vol_target / (vol * np.sqrt(252 / window) + 1e-4)
        return np.clip(scale, 0.1, 3.0)

    def drawdown_scaler(self, equity_curve, window=60):
        dd = np.zeros(len(equity_curve))
        running_max = np.maximum.accumulate(equity_curve)
        dd = (equity_curve - running_max) / (running_max + 1e-8)
        dd_scale = np.exp(5.0 * dd)
        dd_scale = np.clip(dd_scale, 0.1, 1.0)
        return dd_scale

    def da_confidence_scaler(self, signals, returns, window=60):
        da = rolling_da(signals, returns, window)
        da_scale = (da - 0.5) * 4.0
        da_scale = np.clip(da_scale, 0.0, 2.0)
        return da_scale

    def calibrate(self, signals, returns, equity, window=60):
        vol_s = self.volatility_scaler(returns, window)
        dd_s = self.drawdown_scaler(equity, window)
        da_s = self.da_confidence_scaler(signals, returns, window)
        return vol_s, dd_s, da_s
