import numpy as np


class ExecutionGate:
    def __init__(self, vol_percentile=40, min_da=0.55):
        self.vol_percentile = vol_percentile
        self.min_da = min_da
        self.vol_threshold = None

    def fit(self, returns_series):
        vol = np.zeros(len(returns_series))
        for i in range(20, len(returns_series)):
            vol[i] = float(np.std(returns_series[i - 20:i]))
        vol_pos = vol[vol > 0]
        if len(vol_pos) > 0:
            self.vol_threshold = float(np.percentile(vol_pos, self.vol_percentile))
        else:
            self.vol_threshold = 0.01

    def gate(self, z_t, signal, volatility_regime, rolling_da):
        if rolling_da < self.min_da:
            return 0.0
        if volatility_regime > self.vol_threshold:
            return 0.0
        score = abs(signal) / (1.0 + volatility_regime * 10.0)
        return float(np.clip(score, 0.0, 1.0))

    def gate_batch(self, z_seq, signals, returns, vol_regime, rolling_da_series):
        gates = np.zeros(len(signals))
        for i in range(len(signals)):
            if rolling_da_series[i] >= self.min_da and vol_regime[i] <= self.vol_threshold:
                gates[i] = abs(signals[i]) / (1.0 + vol_regime[i] * 10.0)
                gates[i] = float(np.clip(gates[i], 0.0, 1.0))
        return gates


class SessionFilter:
    def is_tradeable(self, timestamp):
        return True


class SpreadFilter:
    def is_tradeable(self, z_t, signal, spread_estimate):
        return abs(signal) > 2.0 * spread_estimate if spread_estimate > 0 else True
