import numpy as np

class RiskWeightedBasket:
    def __init__(self):
        self.asset_weights = {}
        self.correlation_matrix = None
        self.assets = []

    def calibrate_weights(self, asset_returns_dict):
        self.assets = list(asset_returns_dict.keys())
        n = len(self.assets)
        W = np.ones(n)
        for i, sym in enumerate(self.assets):
            r = np.array(asset_returns_dict[sym])
            sharpe = float(np.mean(r) / (np.std(r) + 1e-8))
            W[i] = max(0.1, sharpe + 0.5)
        W = W / np.sum(W)
        self.asset_weights = {self.assets[i]: W[i] for i in range(n)}
        return W

    def compute_correlation(self, asset_returns_dict):
        n = len(self.assets)
        min_len = min(len(v) for v in asset_returns_dict.values())
        corr = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                ri = np.array(asset_returns_dict[self.assets[i]][:min_len])
                rj = np.array(asset_returns_dict[self.assets[j]][:min_len])
                c = float(np.corrcoef(ri, rj)[0, 1])
                corr[i, j] = c
                corr[j, i] = c
        self.correlation_matrix = corr
        return corr

    def portfolio_vol(self, weights):
        if self.correlation_matrix is None:
            return 1.0
        return float(np.sqrt(weights @ self.correlation_matrix @ weights))

    def diversify_weights(self, asset_returns_dict, reg=0.1):
        W = self.calibrate_weights(asset_returns_dict)
        C = self.compute_correlation(asset_returns_dict)
        inv_sqrt = np.linalg.inv(np.linalg.cholesky(C + reg * np.eye(len(self.assets))))
        W_div = inv_sqrt @ W
        W_div = np.maximum(W_div, 0.0)
        W_div = W_div / (np.sum(W_div) + 1e-8)
        self.asset_weights = {self.assets[i]: W_div[i] for i in range(len(self.assets))}
        return W_div

    def weight(self, symbol):
        return self.asset_weights.get(symbol, 0.0)

    def portfolio_return(self, asset_returns):
        pnl = 0.0
        for sym, r in asset_returns.items():
            pnl += self.weight(sym) * r
        return pnl
