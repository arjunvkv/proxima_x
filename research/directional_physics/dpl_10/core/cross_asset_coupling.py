import numpy as np

class CrossAssetCoupling:
    def __init__(self):
        self.coupling_matrix = None
        self.anchor_field = None
        self.target_symbols = []

    def fit(self, anchor_z, target_zs_dict, lookback=20):
        n_targets = len(target_zs_dict)
        dim = anchor_z.shape[1]
        self.target_symbols = list(target_zs_dict.keys())
        n = min(len(anchor_z), *[len(v) for v in target_zs_dict.values()])
        C = np.zeros((n_targets, dim, dim))
        for t_idx, (sym, tz) in enumerate(target_zs_dict.items()):
            for i in range(lookback, n):
                dz_target = tz[i] - tz[i - 1]
                dz_anchor = anchor_z[i] - anchor_z[i - 1]
                dz_hist = anchor_z[i - lookback:i] - anchor_z[i - lookback:i].mean(axis=0)
                dz_design = dz_hist.T / (np.linalg.norm(dz_hist, 'fro') + 1e-8)
                if dz_design.shape[0] != dim or dz_design.shape[1] != lookback:
                    continue
                try:
                    beta = np.linalg.lstsq(dz_design.T, dz_target, rcond=None)[0]
                    C[t_idx] += np.outer(beta, beta) / (np.linalg.norm(beta) + 1e-8)
                except np.linalg.LinAlgError:
                    pass
            C[t_idx] /= n - lookback
        self.coupling_matrix = C
        return C

    def predict_cross(self, anchor_velocity, target_idx=0):
        if self.coupling_matrix is None:
            return None
        C_sym = self.coupling_matrix[target_idx]
        v_n = anchor_velocity / (np.linalg.norm(anchor_velocity) + 1e-8)
        return C_sym @ v_n

    def cross_correlation(self, z_a, z_b, max_lag=10):
        n = min(len(z_a), len(z_b))
        norms_a = np.linalg.norm(z_a[:n], axis=1)
        norms_b = np.linalg.norm(z_b[:n], axis=1)
        cc = np.correlate(norms_a - np.mean(norms_a), norms_b - np.mean(norms_b), mode="full")
        cc = cc / (np.std(norms_a) * np.std(norms_b) * n + 1e-8)
        mid = len(cc) // 2
        lags = np.arange(-max_lag, max_lag + 1)
        return lags, cc[mid - max_lag:mid + max_lag + 1]

    def optimal_lag(self, z_a, z_b, max_lag=10):
        lags, cc = self.cross_correlation(z_a, z_b, max_lag)
        best_idx = int(np.argmax(np.abs(cc)))
        return int(lags[best_idx]), float(cc[best_idx])
