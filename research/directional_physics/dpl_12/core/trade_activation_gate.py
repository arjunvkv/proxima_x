import numpy as np

def cosine(a, b):
    na = np.linalg.norm(a) + 1e-8
    nb = np.linalg.norm(b) + 1e-8
    return float(np.dot(a, b) / (na * nb))


def entropy(x, bins=10):
    hist, _ = np.histogram(x, bins=bins, density=True)
    hist = hist + 1e-8
    hist = hist / np.sum(hist)
    return float(-np.sum(hist * np.log(hist)))


class TradeActivationGate:
    def __init__(self, tcma_window=15, entropy_window=30):
        self.tcma_window = tcma_window
        self.entropy_window = entropy_window

    def tcma_coherence(self, z_seq):
        coh = np.zeros(len(z_seq))
        for t in range(self.tcma_window, len(z_seq)):
            base = z_seq[t]
            sims = [cosine(base, z_seq[t - k]) for k in range(1, self.tcma_window)]
            coh[t] = float(np.mean(sims))
        return coh

    def energy_compression(self, features_list, window=10):
        comp = np.zeros(len(features_list))
        for t in range(window, len(features_list)):
            balance_hist = [features_list[i].get("energy_balance", 0.0) for i in range(t - window, t)]
            release_hist = [features_list[i].get("energy_release", 0.0) for i in range(t - window, t)]
            delta_balance = balance_hist[-1] - balance_hist[0]
            delta_release = release_hist[-1] - release_hist[0]
            comp[t] = float(np.tanh(delta_balance - delta_release))
        return comp

    def manifold_entropy(self, z_seq):
        ent = np.zeros(len(z_seq))
        for t in range(self.entropy_window, len(z_seq)):
            window = z_seq[t - self.entropy_window:t].flatten()
            ent[t] = entropy(window)
        return ent

    def gate_score(self, z_seq, features_list):
        coh = self.tcma_coherence(z_seq)
        comp = self.energy_compression(features_list)
        ent = self.manifold_entropy(z_seq)
        coh_n = (coh - np.mean(coh[coh > 0])) / (np.std(coh[coh > 0]) + 1e-8) if np.sum(coh > 0) > 0 else coh
        ent_n = (ent - np.mean(ent[ent > 0])) / (np.std(ent[ent > 0]) + 1e-8) if np.sum(ent > 0) > 0 else ent
        gate = 0.4 * np.tanh(coh_n) + 0.3 * comp + 0.3 * np.tanh(-ent_n)
        gate = np.clip(gate, -1.0, 1.0)
        return gate, coh, comp, ent

    def gate_active(self, z_seq, features_list, threshold=0.0):
        gate, _, _, _ = self.gate_score(z_seq, features_list)
        active = gate > threshold
        return active, gate
