import numpy as np

def velocity_magnitude(z_seq):
    dz = np.diff(z_seq, axis=0)
    dz = np.vstack([dz[0:1], dz])
    return np.linalg.norm(dz, axis=1)

def acceleration(z_seq):
    dz = np.diff(z_seq, axis=0)
    dz = np.vstack([dz[0:1], dz])
    d2z = np.diff(dz, axis=0)
    d2z = np.vstack([d2z[0:1], d2z])
    return np.linalg.norm(d2z, axis=1)

def curvature(z_seq):
    dz = np.diff(z_seq, axis=0)
    dz = np.vstack([dz[0:1], dz])
    d2z = np.diff(dz, axis=0)
    d2z = np.vstack([d2z[0:1], d2z])
    speed = np.linalg.norm(dz, axis=1) + 1e-8
    accel_norm = np.linalg.norm(d2z, axis=1)
    cross_dz_d2z = np.zeros(len(z_seq))
    for i in range(1, len(z_seq) - 1):
        d = dz[i]
        d2 = d2z[i]
        cross_dz_d2z[i] = float(np.linalg.norm(np.cross(d[:3], d2[:3]))) if len(d) >= 3 else 0.0
    curv = cross_dz_d2z / (speed ** 3 + 1e-8)
    return curv

def divergence_from_field(tfield, z_queries):
    divs = np.zeros(len(z_queries))
    valid = np.ones(len(z_queries), dtype=bool)
    for i, z in enumerate(z_queries):
        d = tfield.flow_divergence(z)
        if d is None:
            valid[i] = False
            divs[i] = 0.0
        else:
            divs[i] = d
    return divs, valid

def flow_coherence(velocities):
    n = len(velocities)
    if n < 5:
        return np.zeros(n)
    coh = np.zeros(n)
    for i in range(5, n):
        window = velocities[i - 5:i]
        window_n = np.array([v / (np.linalg.norm(v) + 1e-8) for v in window])
        mean_dir = np.mean(window_n, axis=0)
        mean_dir_n = mean_dir / (np.linalg.norm(mean_dir) + 1e-8)
        alignments = np.array([np.dot(v, mean_dir_n) for v in window_n])
        coh[i] = float(np.mean(alignments))
    return coh
