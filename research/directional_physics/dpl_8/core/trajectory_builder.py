import numpy as np


class TrajectoryBuilder:
    def __init__(self, window=20):
        self.window = window

    def build(self, z_seq):
        dtype = np.float32
        traj = np.zeros((len(z_seq), 12), dtype=dtype)
        for t in range(self.window, len(z_seq)):
            seg = z_seq[t - self.window:t]
            mean = seg.mean(axis=0).astype(dtype)
            std = seg.std(axis=0).astype(dtype)
            trend = (seg[-1] - seg[0]).astype(dtype)
            velocity = (seg[-1] - seg[-2]).astype(dtype)
            curvature = self._curvature(seg)
            energy = float(np.mean(np.sum(seg ** 2, axis=1)))
            traj[t] = np.concatenate([
                mean[:3], std[:3], trend[:2], velocity[:2],
                np.array([curvature, energy], dtype=dtype)
            ])
        return traj

    def _curvature(self, seg):
        if len(seg) < 3:
            return 0.0
        d1 = seg[1:] - seg[:-1]
        d2 = d1[1:] - d1[:-1]
        return float(np.mean(np.linalg.norm(d2, axis=1)))
