import numpy as np

class TransitionField:
    def __init__(self, k=20, sim_threshold=0.5):
        self.k = k
        self.sim_threshold = sim_threshold
        self.z_train = None
        self.velocities = None
        self._mean = None
        self._std = None

    def _normalize(self, z):
        return (z - self._mean) / self._std

    def fit(self, z_series):
        self._mean = np.mean(z_series, axis=0)
        self._std = np.std(z_series, axis=0) + 1e-8
        z_norm = self._normalize(z_series)
        self.z_train = z_norm
        v = np.diff(z_norm, axis=0)
        v = np.vstack([v[0:1], v])
        self.velocities = v.astype(np.float64)

    def _cos_sim(self, a, b):
        na = np.linalg.norm(a) + 1e-8
        nb = np.linalg.norm(b) + 1e-8
        return float(np.dot(a, b) / (na * nb))

    def _neighbors(self, z_q):
        sims = np.array([self._cos_sim(z_q, z) for z in self.z_train])
        best_k = min(self.k, len(sims))
        idx = np.argpartition(-sims, best_k)[:best_k]
        idx = idx[sims[idx] > self.sim_threshold]
        if len(idx) == 0:
            return None, None
        return idx, sims[idx]

    def predict(self, z_q):
        z_qn = self._normalize(z_q.reshape(1, -1)).flatten()
        idx, sims = self._neighbors(z_qn)
        if idx is None:
            return None
        weights = sims / (np.sum(sims) + 1e-8)
        v_pred = np.sum(self.velocities[idx].T * weights, axis=1)
        return v_pred.astype(np.float64)

    def predict_batch(self, z_queries):
        z_qn = self._normalize(z_queries)
        field = np.zeros_like(z_qn)
        valid = np.ones(len(z_qn), dtype=bool)
        for i in range(len(z_qn)):
            v = self.predict(z_queries[i])
            if v is None:
                valid[i] = False
            else:
                field[i] = v
        return field, valid

    def flow_divergence(self, z_q, eps=0.01):
        z_qn = self._normalize(z_q.reshape(1, -1)).flatten()
        idx, sims = self._neighbors(z_qn)
        if idx is None or len(idx) < self.k // 2:
            return None
        z_neighbors = self.z_train[idx]
        v_neighbors = self.velocities[idx]
        A = z_neighbors - z_qn
        b = v_neighbors - np.mean(v_neighbors, axis=0)
        A_2d = A.reshape(-1, A.shape[-1])
        b_2d = b.reshape(-1, b.shape[-1])
        try:
            J = np.linalg.lstsq(A_2d, b_2d, rcond=None)[0]
        except np.linalg.LinAlgError:
            return None
        return float(np.trace(J))

    def flow_alignment(self, z_prev, z_curr, z_next):
        v_actual = z_next - z_curr
        v_expected = self.predict(z_curr)
        if v_expected is None:
            return None
        v_curr = z_curr - z_prev
        alignment = self._cos_sim(v_actual, v_expected)
        return alignment
