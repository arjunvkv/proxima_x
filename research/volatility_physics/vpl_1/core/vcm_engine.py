import numpy as np


def compute_burst_state(log_ret, rv, threshold_z=1.5):
    rv_z = (rv - np.nanmean(rv)) / np.nanstd(rv)
    burst = (rv_z > threshold_z).astype(np.float64)
    return rv_z, burst


def compute_burst_duration(burst):
    n = len(burst)
    dur = np.full(n, np.nan)
    current = 0
    for i in range(n):
        if np.isnan(burst[i]):
            dur[i] = 0
            current = 0
        elif burst[i] > 0.5:
            current += 1
            dur[i] = current
        else:
            dur[i] = 0
            current = 0
    return dur


def compute_burst_recency(burst, decay_lambda=0.1):
    n = len(burst)
    recency = np.full(n, 0.0)
    last_burst = -1
    for i in range(n):
        if np.isnan(burst[i]):
            continue
        if burst[i] > 0.5:
            last_burst = i
            recency[i] = 1.0
        elif last_burst >= 0:
            d = i - last_burst
            recency[i] = np.exp(-decay_lambda * d)
    return recency


def compute_burst_density(burst, window=50):
    n = len(burst)
    cum = np.zeros(n + 1)
    b_clean = np.nan_to_num(burst, nan=0.0)
    cum[1:] = np.cumsum(b_clean)
    density = np.full(n, np.nan)
    for i in range(window, n):
        density[i] = cum[i] - cum[i - window]
    return density


def compute_vmr(log_ret, rv, decay_lambda=0.05):
    n = len(log_ret)
    vmr = np.full(n, np.nan)
    rv_clean = np.nan_to_num(rv, nan=0.0)
    weight = 1.0
    for i in range(1, n):
        vmr[i] = vmr[i - 1] * (1 - decay_lambda) + rv_clean[i] * decay_lambda
        if np.isnan(vmr[i - 1]):
            vmr[i] = rv_clean[i] * decay_lambda
    return vmr


def compute_vcm(log_ret, rv, burst_z_threshold=1.5, density_window=50,
                recency_lambda=0.1, vmr_lambda=0.05):
    rv_z, burst = compute_burst_state(log_ret, rv, threshold_z=burst_z_threshold)
    duration = compute_burst_duration(burst)
    recency = compute_burst_recency(burst, decay_lambda=recency_lambda)
    density = compute_burst_density(burst, window=density_window)
    vmr = compute_vmr(log_ret, rv, decay_lambda=vmr_lambda)
    n = len(log_ret)
    max_vmr = np.nanmax(vmr) if np.nanmax(vmr) > 1e-12 else 1.0
    vcm = np.full(n, np.nan)
    for i in range(n):
        dur_n = duration[i] / (duration[i] + 1.0) if not np.isnan(duration[i]) else 0.0
        den_n = density[i] / density_window if not np.isnan(density[i]) else 0.0
        rec_n = recency[i] if not np.isnan(recency[i]) else 0.0
        vmr_n = vmr[i] / max_vmr if not np.isnan(vmr[i]) else 0.0
        vcm[i] = dur_n * 0.2 + den_n * 0.2 + rec_n * 0.3 + vmr_n * 0.3
    return {
        "rv_z": rv_z,
        "burst": burst,
        "duration": duration,
        "recency": recency,
        "density": density,
        "vmr": vmr,
        "vcm": vcm,
    }
