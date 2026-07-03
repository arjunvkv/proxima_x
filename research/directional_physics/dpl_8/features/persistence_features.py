import numpy as np


class PersistenceFeatureBuilder:
    def build(self, z_seq, traj_features):
        T = len(z_seq)
        dz = np.zeros_like(z_seq)
        dz[1:] = z_seq[1:] - z_seq[:-1]
        d2z = np.zeros_like(z_seq)
        d2z[2:] = dz[2:] - dz[1:-1]
        X = []
        for t in range(T):
            feat = np.concatenate([z_seq[t], dz[t], d2z[t], traj_features[t]]).astype(np.float32)
            X.append(feat)
        return np.array(X, dtype=np.float32)
