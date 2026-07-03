import numpy as np

class FlowSignalHead:
    def __init__(self, dim=8, lr=0.001):
        self.dim = dim
        self.lr = lr
        self.w = np.random.randn(dim) * 0.01
        self.w = self.w / (np.linalg.norm(self.w) + 1e-8)
        self._signal_type = "velocity_dot_w"

    def fit(self, expected_velocities, returns, epochs=10):
        w = self.w.copy()
        for epoch in range(epochs):
            epoch_lr = self.lr / (epoch + 1)
            perm = np.random.permutation(len(expected_velocities))
            for idx in perm:
                v = expected_velocities[idx]
                r = returns[idx]
                v_n = v / (np.linalg.norm(v) + 1e-8)
                pred = float(np.dot(v_n, w))
                grad = -2.0 * (pred - r) * v_n
                w -= epoch_lr * grad
                wn = np.linalg.norm(w)
                if wn > 3.0:
                    w *= (3.0 / wn)
        self.w = w.copy()
        train_preds = np.array([float(np.dot(v / (np.linalg.norm(v) + 1e-8), self.w)) for v in expected_velocities])
        ic = float(np.corrcoef(train_preds, returns)[0, 1]) if len(train_preds) > 2 else 0.0
        return ic if not np.isnan(ic) else 0.0

    def predict(self, expected_velocity):
        v_n = expected_velocity / (np.linalg.norm(expected_velocity) + 1e-8)
        return float(np.dot(v_n, self.w))

    def predict_batch(self, expected_velocities):
        scores = np.zeros(len(expected_velocities))
        for i in range(len(expected_velocities)):
            scores[i] = self.predict(expected_velocities[i])
        return scores

    def signal_via_divergence(self, expected_velocity, divergence):
        v_n = expected_velocity / (np.linalg.norm(expected_velocity) + 1e-8)
        base = float(np.dot(v_n, self.w))
        if divergence is not None:
            div_mod = np.tanh(-divergence * 2.0)
            base = base * (0.5 + 0.5 * div_mod)
        return float(np.tanh(base))

    def signal_via_flow_coherence(self, expected_velocity, coherence):
        v_n = expected_velocity / (np.linalg.norm(expected_velocity) + 1e-8)
        base = float(np.dot(v_n, self.w))
        return float(np.tanh(base * coherence))
