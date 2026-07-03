import numpy as np

class FrozenSignalProjection:
    def __init__(self, dim=8):
        self.dim = dim
        self.mu = None
        self.sd = None
        self.tcma_W = None
        self.readout_w = None

    def fit(self, z_train, records, n_tcma_epochs=3):
        self.mu = np.mean(z_train, axis=0)
        self.sd = np.std(z_train, axis=0) + 1e-8
        z_n = (z_train - self.mu) / self.sd
        z_norm = np.array([z / (np.linalg.norm(z) + 1e-8) for z in z_n])

        from research.directional_physics.dpl7.core.temporal_contrastive_aligner import TemporalContrastiveAligner
        tcma = TemporalContrastiveAligner(dim=self.dim, tau=0.5, lambda_drift=0.1)
        tcma.pretrain(z_norm, records, n_epochs=n_tcma_epochs, lr=0.001)
        self.tcma_W = tcma.W.copy()

        z_a = np.array([self._tcma_project(z) for z in z_norm])
        z_a_n = z_a / (np.linalg.norm(z_a, axis=1, keepdims=True) + 1e-8)

        rets = np.array([r[f"return_h20"] for r in records])
        w = np.mean(rets[:, None] * z_a_n, axis=0)
        w = w / (np.linalg.norm(w) + 1e-8)
        self.readout_w = w.copy()
        scores = z_a_n @ self.readout_w
        ic = float(np.corrcoef(scores, rets)[0, 1]) if len(scores) > 2 else 0.0
        return ic if not np.isnan(ic) else 0.0

    def _tcma_project(self, z):
        return (self.tcma_W @ z).flatten()

    def raw_signal(self, z):
        if self.mu is None:
            return 0.0
        z_n = (z - self.mu) / self.sd
        z_norm = z_n / (np.linalg.norm(z_n) + 1e-8)
        z_a = self._tcma_project(z_norm)
        z_a_n = z_a / (np.linalg.norm(z_a) + 1e-8)
        return float(z_a_n @ self.readout_w)

    def raw_signal_batch(self, z_seq):
        signals = np.zeros(len(z_seq))
        for i in range(len(z_seq)):
            signals[i] = self.raw_signal(z_seq[i])
        return signals
