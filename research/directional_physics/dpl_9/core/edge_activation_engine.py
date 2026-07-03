import numpy as np


def entropy(x, bins=10):
    hist, _ = np.histogram(x, bins=bins, density=True)
    hist = hist + 1e-8
    hist = hist / np.sum(hist)
    return float(-np.sum(hist * np.log(hist)))


def rolling_std(x, window=20):
    out = np.zeros(len(x))
    for i in range(window, len(x)):
        out[i] = float(np.std(x[i - window:i]))
    return out


def cosine(a, b):
    na = np.linalg.norm(a) + 1e-8
    nb = np.linalg.norm(b) + 1e-8
    return float(np.dot(a, b) / (na * nb))


class EdgeActivationEngine:
    def __init__(self, entropy_window=30, tcma_window=15, vol_window=20):
        self.entropy_window = entropy_window
        self.tcma_window = tcma_window
        self.vol_window = vol_window
        self.entropy_thresh = None
        self.tcma_thresh = None
        self.vol_thresh = None

    def compute_entropy_signal(self, z_seq):
        ent = np.zeros(len(z_seq))
        for t in range(self.entropy_window, len(z_seq)):
            window = z_seq[t - self.entropy_window:t].flatten()
            ent[t] = entropy(window)
        return ent

    def compute_tcma_stability(self, z_seq):
        stability = np.zeros(len(z_seq))
        for t in range(self.tcma_window, len(z_seq)):
            base = z_seq[t]
            sims = [cosine(base, z_seq[t - k]) for k in range(1, self.tcma_window)]
            stability[t] = float(np.mean(sims))
        return stability

    def compute_volatility_compression(self, z_seq):
        dz = np.diff(z_seq, axis=0)
        dz = np.vstack([dz[0:1], dz])
        vol = np.linalg.norm(dz, axis=1)
        return rolling_std(vol, self.vol_window)

    def fit_thresholds(self, z_seq):
        ent = self.compute_entropy_signal(z_seq)
        stab = self.compute_tcma_stability(z_seq)
        vol = self.compute_volatility_compression(z_seq)
        ent_pos = ent[ent > 0]
        stab_pos = stab[stab > 0]
        vol_pos = vol[vol > 0]
        self.entropy_thresh = float(np.quantile(ent_pos, 0.35)) if len(ent_pos) > 0 else 0.5
        self.tcma_thresh = float(np.quantile(stab_pos, 0.65)) if len(stab_pos) > 0 else 0.5
        self.vol_thresh = float(np.quantile(vol_pos, 0.40)) if len(vol_pos) > 0 else 0.5

    def compute_gate(self, z_seq):
        if self.entropy_thresh is None:
            self.fit_thresholds(z_seq)
        ent = self.compute_entropy_signal(z_seq)
        stab = self.compute_tcma_stability(z_seq)
        vol = self.compute_volatility_compression(z_seq)
        gate = np.zeros(len(z_seq), dtype=np.float64)
        for t in range(len(z_seq)):
            if (ent[t] < self.entropy_thresh and
                stab[t] > self.tcma_thresh and
                vol[t] < self.vol_thresh):
                gate[t] = 1.0
        return gate

    def edge_strength(self, z_t):
        return float(np.tanh(np.sum(z_t)))
