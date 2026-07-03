import numpy as np

def cosine(a, b):
    na = np.linalg.norm(a) + 1e-8
    nb = np.linalg.norm(b) + 1e-8
    return float(np.dot(a, b) / (na * nb))


class PersistenceLabeler:
    def __init__(self, max_horizon=100, threshold_quantile=0.2):
        self.max_horizon = max_horizon
        self.threshold_quantile = threshold_quantile
        self.threshold = None

    def fit_threshold(self, z_seq):
        sims = []
        for i in range(len(z_seq) - 10):
            sims.append(cosine(z_seq[i], z_seq[i + 5]))
        self.threshold = float(np.quantile(sims, self.threshold_quantile))

    def compute_tau(self, z_seq):
        if self.threshold is None:
            self.fit_threshold(z_seq)
        n = len(z_seq)
        tau = np.zeros(n, dtype=np.float32)
        for t in range(n):
            base = z_seq[t]
            for k in range(1, self.max_horizon):
                if t + k >= n:
                    break
                sim = cosine(base, z_seq[t + k])
                if sim < self.threshold:
                    tau[t] = k
                    break
            if tau[t] == 0:
                tau[t] = float(self.max_horizon)
        return tau
