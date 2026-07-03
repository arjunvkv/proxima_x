import numpy as np


def compute_temporal_pace(log_ret, window=24):
    avg_abs = np.full(len(log_ret), np.nan)
    for i in range(window, len(log_ret)):
        avg_abs[i] = np.mean(np.abs(log_ret[i - window:i]))
    pace = np.abs(log_ret) / avg_abs
    pace = np.where(np.isinf(pace), np.nan, pace)
    return pace, avg_abs


def compute_displacement_efficiency(log_ret, window=12):
    n = len(log_ret)
    de = np.full(n, np.nan)
    for i in range(window, n):
        total_path = np.sum(np.abs(log_ret[i - window + 1:i + 1]))
        net_move = np.abs(np.sum(log_ret[i - window + 1:i + 1]))
        if total_path > 1e-12:
            de[i] = net_move / total_path
        else:
            de[i] = 0.0
    return de


def compute_reversal_density(log_ret, window=12):
    signs = np.sign(log_ret)
    n = len(signs)
    rd = np.full(n, np.nan)
    for i in range(window, n):
        flips = np.sum(np.abs(np.diff(signs[i - window + 1:i + 1]))) / 2
        rd[i] = flips / window
    return rd


def compute_sit_state(close, high, low, log_ret, rv, saf_val, displacement_window=12,
                      reversal_window=12, pace_window=24):
    n = len(log_ret)
    n_close = len(close)

    V = np.full(n_close, np.nan)
    V[1:] = rv

    T_pace, _ = compute_temporal_pace(log_ret, window=pace_window)
    T = np.full(n_close, np.nan)
    T[1:] = T_pace

    C = saf_val.copy()

    D_ret = compute_displacement_efficiency(log_ret, window=displacement_window)
    D = np.full(n_close, np.nan)
    D[1:] = D_ret

    R_ret = compute_reversal_density(log_ret, window=reversal_window)
    R = np.full(n_close, np.nan)
    R[1:] = R_ret

    state_matrix = np.column_stack([V, T, C, D, R])

    return {
        "V": V, "T": T, "C": C, "D": D, "R": R,
        "state_matrix": state_matrix,
    }


def normalize_state(state_matrix):
    normed = np.zeros_like(state_matrix)
    for j in range(state_matrix.shape[1]):
        col = state_matrix[:, j]
        std = np.nanstd(col)
        if std > 1e-12:
            normed[:, j] = (col - np.nanmean(col)) / std
        else:
            normed[:, j] = 0.0
    return normed


def compute_instability(state_matrix):
    n = state_matrix.shape[0]
    normed = normalize_state(state_matrix)
    prev = np.roll(normed, 1, axis=0)
    prev[0] = np.nan
    diff = normed - prev
    I = np.sqrt(np.nansum(diff ** 2, axis=1))
    J = np.full(n, np.nan)
    J[1:] = I[1:] - I[:-1]
    A = np.full(n, np.nan)
    A[2:] = J[2:] - J[1:-1]
    return {"instability": I, "jerk": J, "acceleration": A}


def compute_sit(close, high, low, log_ret, rv, saf_val, displacement_window=12,
                reversal_window=12, pace_window=24):
    state = compute_sit_state(close, high, low, log_ret, rv, saf_val,
                              displacement_window=displacement_window,
                              reversal_window=reversal_window,
                              pace_window=pace_window)
    sm = state["state_matrix"]
    inst = compute_instability(sm)
    result = {**state, **inst}
    return result


def compute_grid_profile(saf_dec, sit_dec, labels):
    n = min(len(saf_dec), len(sit_dec), len(labels))
    saf_dec = saf_dec[:n]
    sit_dec = sit_dec[:n]
    labels = labels[:n]
    grid = np.full((10, 10), np.nan)
    count = np.zeros((10, 10), dtype=int)
    for i in range(n):
        s = saf_dec[i]
        t = sit_dec[i]
        if np.isnan(s) or np.isnan(t) or np.isnan(labels[i]):
            continue
        si, ti = int(s), int(t)
        count[si, ti] += 1
        if np.isnan(grid[si, ti]):
            grid[si, ti] = 0.0
        grid[si, ti] += labels[i]
    for si in range(10):
        for ti in range(10):
            if count[si, ti] > 0:
                grid[si, ti] /= count[si, ti]
    return grid, count
