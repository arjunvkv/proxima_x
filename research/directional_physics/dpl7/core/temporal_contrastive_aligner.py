import numpy as np


class TemporalContrastiveAligner:
    def __init__(self, dim=8, tau=0.5, lambda_drift=0.1, ema_alpha=0.1):
        self.dim = dim
        self.tau = tau
        self.lambda_drift = lambda_drift
        self.ema_alpha = ema_alpha
        self.W = np.eye(dim) * 0.1
        self.ema_z = None

    def _cos_sim(self, a, b):
        a_norm = a / (np.linalg.norm(a) + 1e-8)
        b_norm = b / (np.linalg.norm(b) + 1e-8)
        return float(np.clip(np.dot(a_norm, b_norm), -1.0, 1.0))

    def align(self, z):
        return self.W @ z

    def align_batch(self, z_batch):
        return np.array([self.align(z) for z in z_batch])

    def build_es_vec(self, record):
        return np.array([
            record.get("es_rank", 0.5),
            record.get("es_slope", 0.0),
            record.get("energy_balance", 0.0),
            record.get("returns_vol", 0.0)
        ], dtype=np.float64)

    def build_vol_feature(self, record):
        return record.get("returns_vol", 0.0)

    def sample_positive(self, z_bank, idx, es_bank, max_attempts=30):
        base_es = es_bank[idx]
        best_j = idx
        best_sim = -1.0
        for _ in range(max_attempts):
            j = np.random.randint(len(z_bank))
            if abs(j - idx) < 5 or abs(j - idx) > 30:
                continue
            es_sim = self._cos_sim(base_es, es_bank[j])
            if es_sim > 0.80 and es_sim > best_sim:
                best_sim = es_sim
                best_j = j
        if best_j == idx:
            for j in range(max(0, idx - 30), min(len(z_bank), idx + 30)):
                if abs(j - idx) < 5 or j == idx:
                    continue
                es_sim = self._cos_sim(base_es, es_bank[j])
                if es_sim > best_sim:
                    best_sim = es_sim
                    best_j = j
        return best_j

    def sample_negative(self, z_bank, idx, vol_bank, max_attempts=30):
        base_vol = vol_bank[idx]
        best_j = idx
        best_diff = -1.0
        for _ in range(max_attempts):
            j = np.random.randint(len(z_bank))
            if abs(j - idx) < 5:
                continue
            vol_diff = abs(vol_bank[j] - base_vol)
            if vol_diff > 0.3 and vol_diff > best_diff:
                best_diff = vol_diff
                best_j = j
        if best_j == idx:
            for j in range(len(z_bank)):
                if abs(j - idx) < 5 or j == idx:
                    continue
                vol_diff = abs(vol_bank[j] - base_vol)
                if vol_diff > best_diff:
                    best_diff = vol_diff
                    best_j = j
        return best_j

    def compute_loss(self, z_i, z_pos, z_negs, vol_i):
        zi = self.align(z_i)
        zp = self.align(z_pos)
        zn = np.array([self.align(n) for n in z_negs])
        pos_sim = self._cos_sim(zi, zp)
        neg_sims = np.array([self._cos_sim(zi, n) for n in zn])
        tau = max(self.tau, 0.05)
        logits = np.exp(np.clip(np.concatenate([[pos_sim], neg_sims]) / tau, -20, 20))
        loss_con = -np.log((logits[0] + 1e-8) / (np.sum(logits) + 1e-8))
        if self.ema_z is None:
            self.ema_z = zi.copy()
        self.ema_z = self.ema_alpha * zi + (1.0 - self.ema_alpha) * self.ema_z
        loss_drift = float(np.sum((zi - self.ema_z) ** 2))
        weight = 1.0 / (1.0 + abs(vol_i))
        loss = weight * loss_con + self.lambda_drift * loss_drift
        return loss, zi, zp, zn, pos_sim, neg_sims, loss_con, loss_drift

    def update(self, z_i, z_pos, z_negs, vol_i, lr=0.001):
        loss, zi, zp, zn, pos_sim, neg_sims, loss_con, loss_drift = self.compute_loss(z_i, z_pos, z_negs, vol_i)
        z_i_norm = z_i / (np.linalg.norm(z_i) + 1e-8)
        z_p_norm = z_pos / (np.linalg.norm(z_pos) + 1e-8)
        z_n_norms = np.array([n / (np.linalg.norm(n) + 1e-8) for n in z_negs])
        tau = max(self.tau, 0.05)
        all_logits = np.exp(np.clip(np.array([pos_sim] + list(neg_sims)) / tau, -20, 20))
        probs = all_logits / (np.sum(all_logits) + 1e-8)
        grad = np.zeros_like(self.W)
        for k in range(len(z_negs)):
            z_neg = z_negs[k]
            z_n_norm = z_n_norms[k]
            w_pos = (1.0 - probs[0]) / tau
            w_neg = -probs[k + 1] / tau
            grad += w_pos * np.outer(z_i_norm, z_p_norm)
            grad += w_neg * np.outer(z_i_norm, z_n_norm)
        grad /= (len(z_negs) + 1)
        if self.ema_z is not None:
            drift_grad = 2 * self.lambda_drift * np.outer(z_i_norm, (zi - self.ema_z))
            grad += drift_grad
        grad = np.clip(grad, -1.0, 1.0)
        weight = 1.0 / (1.0 + abs(vol_i))
        self.W -= lr * weight * grad
        s = np.linalg.norm(self.W, 'fro')
        if s > 3.0:
            self.W *= (3.0 / s)
        return float(loss)

    def pretrain(self, z_bank, records, n_epochs=3, lr=0.001):
        es_bank = np.array([self.build_es_vec(r) for r in records])
        vol_bank = np.array([self.build_vol_feature(r) for r in records])
        losses = []
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            n = len(z_bank)
            if n == 0:
                continue
            epoch_lr = lr / (epoch + 1)
            perm = np.random.permutation(n)
            n_updates = 0
            for idx in perm:
                pos_idx = self.sample_positive(z_bank, idx, es_bank)
                neg_indices = []
                for _ in range(2):
                    neg_idx = self.sample_negative(z_bank, idx, vol_bank)
                    if neg_idx != idx and neg_idx not in neg_indices:
                        neg_indices.append(neg_idx)
                if not neg_indices:
                    continue
                z_i = z_bank[idx]
                z_pos = z_bank[pos_idx]
                z_negs = [z_bank[ni] for ni in neg_indices]
                vol_i = vol_bank[idx]
                l = self.update(z_i, z_pos, z_negs, vol_i, lr=epoch_lr)
                epoch_loss += l
                n_updates += 1
            avg_loss = epoch_loss / max(n_updates, 1)
            losses.append(avg_loss)
        return losses
