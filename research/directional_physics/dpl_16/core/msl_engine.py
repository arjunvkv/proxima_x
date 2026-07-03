import numpy as np
from scipy.ndimage import uniform_filter1d

DATA_BASE = "C:/Trading/Agentic_Trading/data/ticks"
SYMBOLS = ["EURJPY", "USDJPY", "GBPUSD", "EURUSD"]

def load_ticks(symbol, stride=5):
    import polars as pl
    df = pl.read_parquet(f"{DATA_BASE}/{symbol}_ticks.parquet")
    arr = df.to_numpy()
    if stride > 1:
        arr = arr[::stride]
    ts = arr[:, 0].astype(np.int64)
    bid = arr[:, 1].astype(np.float64)
    ask = arr[:, 2].astype(np.float64)
    spread = arr[:, 3].astype(np.float64)
    volume = arr[:, 4].astype(np.float64)
    mid = (bid + ask) / 2.0
    return {"symbol": symbol, "timestamp": ts, "bid": bid, "ask": ask,
            "mid": mid, "spread": spread, "volume": volume, "n": len(ts)}

def load_m5(symbol):
    import polars as pl
    path = f"C:/Trading/Agentic_Trading/data/intraday/{symbol}_M5.parquet"
    df = pl.read_parquet(path)
    df = df.sort("timestamp")  # ensure ascending for searchsorted
    cols = df.columns
    arr = df.to_numpy()
    ts = arr[:, 0].astype(np.int64)
    close = arr[:, cols.index("close")].astype(np.float64)
    result = {"symbol": symbol, "timestamp": ts, "close": close, "n": len(close)}
    if "open" in cols:
        result["open"] = arr[:, cols.index("open")].astype(np.float64)
        result["high"] = arr[:, cols.index("high")].astype(np.float64)
        result["low"] = arr[:, cols.index("low")].astype(np.float64)
        result["volume"] = arr[:, cols.index("volume")].astype(np.float64)
    return result

def align_ticks_to_bars(tick_ts_us, bar_ts_sec):
    tick_sec = tick_ts_us // 1_000_000
    starts = np.searchsorted(tick_sec, bar_ts_sec, side='left')
    ends = np.searchsorted(tick_sec, bar_ts_sec, side='right')
    return starts, ends, tick_sec

def aggregate_to_bars_vectorized(arr, tick_bar_idx, n_bars):
    """Fast groupby-mean: for each bar, compute mean of arr over its ticks."""
    valid = ~np.isnan(arr)
    if not np.any(valid):
        return np.full(n_bars, np.nan)
    binned = np.bincount(tick_bar_idx[valid], weights=arr[valid], minlength=n_bars)
    counts = np.bincount(tick_bar_idx[valid], minlength=n_bars)
    out = np.full(n_bars, np.nan)
    mask = counts > 0
    out[mask] = binned[mask] / counts[mask]
    return out

def normalize(s):
    s = np.where(np.isinf(s), np.nan, s)
    mu = np.nanmean(s)
    std = np.nanstd(s)
    if std < 1e-12:
        return np.zeros_like(s)
    return (s - mu) / std

def compute_tpi(mid, tick_windows=(50, 100, 200)):
    n = len(mid)
    delta = np.diff(mid)
    up = np.concatenate([[0], (delta > 1e-8).astype(np.float64)])
    down = np.concatenate([[0], (delta < -1e-8).astype(np.float64)])
    results = {}
    for w in tick_windows:
        u_w = uniform_filter1d(up, size=w, mode='constant', origin=0) * w
        d_w = uniform_filter1d(down, size=w, mode='constant', origin=0) * w
        total = u_w + d_w
        tpi = np.full(n, np.nan)
        valid = total > 0
        tpi[valid] = (u_w[valid] - d_w[valid]) / total[valid]
        results[f"tpi_{w}"] = tpi
        if w > 1 and n > w * 2:
            tpi_clean = np.nan_to_num(tpi)
            sq = uniform_filter1d(tpi_clean ** 2, size=w, mode='constant', origin=0) * w
            mean = uniform_filter1d(tpi_clean, size=w, mode='constant', origin=0) * w
            var = np.maximum(sq - mean ** 2, 0)
            persistence = np.full(n, np.nan)
            pv = np.sqrt(var / w)
            persistence[pv > 0] = pv[pv > 0]
            results[f"tpi_persistence_{w}"] = persistence
    tpi_accel = np.full(n, np.nan)
    tpi_base = results.get(f"tpi_{tick_windows[0]}", np.full(n, np.nan))
    if n > 2:
        tpi_accel[2:] = tpi_base[2:] - 2 * tpi_base[1:-1] + tpi_base[:-2]
    results["tpi_accel"] = tpi_accel
    return results

def compute_ssf(mid, tick_windows=(20, 50, 100)):
    n = len(mid)
    results = {}
    for w in tick_windows:
        if w >= n:
            results[f"sweep_intensity_{w}"] = np.full(n, np.nan)
            continue
        rolled = np.roll(mid, w)
        rolled[:w] = np.nan
        moves = np.abs(mid - rolled) / np.maximum(rolled, 1e-12)
        initial_std = np.nanstd(moves[w:min(n//4, n)])
        if initial_std > 0:
            moves = moves / initial_std
        results[f"sweep_intensity_{w}"] = moves
    return results

def compute_rap(mid, tick_windows=(100, 200)):
    n = len(mid)
    results = {}
    for w in tick_windows:
        if w >= n:
            results[f"rap_{w}"] = np.full(n, np.nan)
            continue
        changes = np.abs(np.diff(mid)) / np.maximum(mid[:-1], 1e-12)
        touches = np.concatenate([[0], (changes > 1e-5).astype(np.float64)])
        touch_counts = uniform_filter1d(touches, size=w, mode='constant', origin=0) * w
        mid_shift = np.roll(mid, w)
        mid_shift[:w] = np.nan
        net_disp = np.abs(mid - mid_shift) / np.maximum(mid_shift, 1e-12)
        rap = np.full(n, np.nan)
        valid = net_disp > 1e-8
        rap[valid] = touch_counts[valid] / net_disp[valid]
        results[f"rap_{w}"] = rap
    return results

def compute_mff(mid, tick_windows=(10, 30)):
    n = len(mid)
    if n < 2:
        return {f"{k}_{w}": np.full(n, np.nan) for w in tick_windows for k in ["alt", "rev", "failed_push"]}
    # All arrays padded to length n
    changes = np.concatenate([[np.nan], np.diff(mid)])
    signs = np.concatenate([[np.nan], np.sign(changes[1:])])
    abs_ch = np.concatenate([[np.nan], np.abs(changes[1:])])
    nonzero = np.concatenate([[np.nan], (signs[1:] != 0).astype(float)])
    up = np.concatenate([[np.nan], (changes[1:] > 0).astype(float)])
    down = np.concatenate([[np.nan], (changes[1:] < 0).astype(float)])
    # sign flips among consecutive non-zero signs
    signs_diff_abs = np.concatenate([[np.nan, np.nan], np.abs(np.diff(signs[1:])) / 2])
    results = {}
    for w in tick_windows:
        if w >= n:
            for k in ["alt", "rev", "failed_push"]:
                results[f"{k}_{w}"] = np.full(n, np.nan)
            continue
        flips_per = uniform_filter1d(np.nan_to_num(signs_diff_abs), size=w, mode='constant', origin=0) * w
        nonzeros_per = uniform_filter1d(np.nan_to_num(nonzero), size=w, mode='constant', origin=0) * w
        alt_density = np.full(n, np.nan)
        valid = nonzeros_per > 1
        alt_density[valid] = flips_per[valid] / nonzeros_per[valid]
        mid_shift = np.roll(mid, w)
        mid_shift[:w] = np.nan
        ac_per = uniform_filter1d(np.nan_to_num(abs_ch), size=w, mode='constant', origin=0) * w
        net = np.abs(mid - mid_shift)
        rev_density = np.full(n, np.nan)
        rv = net > 1e-8
        rev_density[rv] = ac_per[rv] / net[rv]
        rev_density[~rv] = 1.0
        up_per = uniform_filter1d(np.nan_to_num(up), size=w, mode='constant', origin=0) > 0
        down_per = uniform_filter1d(np.nan_to_num(down), size=w, mode='constant', origin=0) > 0
        failed_push = (up_per & down_per).astype(float)
        results[f"alt_{w}"] = alt_density
        results[f"rev_{w}"] = rev_density
        results[f"failed_push_{w}"] = failed_push
    return results

def compute_tfc(timestamp, mid, tick_windows=(50, 100)):
    """TFC: Time-of-Flow Compression using tick timestamps (microseconds)."""
    n = len(mid)
    if n < 2:
        return {f"tfc_{w}": np.full(n, np.nan) for w in tick_windows}
    inter_tick = np.diff(timestamp).astype(np.float64) / 1000.0  # microsec → millisec
    inter_tick = np.maximum(inter_tick, 0.001)
    it_pad = np.concatenate([[inter_tick[0]], inter_tick])
    results = {}
    for w in tick_windows:
        mean_gap = uniform_filter1d(it_pad, size=w, mode='constant', origin=0)
        tfc = np.full(n, np.nan)
        valid = mean_gap > 0
        tfc[valid] = 1.0 / mean_gap[valid]
        results[f"tfc_{w}"] = tfc
    return results

def compute_all_msl_features(mid, timestamp):
    result = {}
    result.update(compute_tpi(mid))
    result.update(compute_ssf(mid))
    result.update(compute_rap(mid))
    result.update(compute_mff(mid))
    result.update(compute_tfc(timestamp, mid))
    return result

def aggregate_features_to_bars(features, tick_bar_idx, n_bars):
    """Vectorized: groupby-mean for each feature."""
    agg = {}
    for key, arr in features.items():
        agg[key] = aggregate_to_bars_vectorized(arr, tick_bar_idx, n_bars)
    return agg

def compute_directional_labels(close, n_bars=3):
    log_ret = np.diff(np.log(close))
    n = len(close)
    label = np.full(n, np.nan)
    for i in range(n - n_bars):
        fwd_ret = np.sum(log_ret[i:i + n_bars])
        label[i] = np.sign(fwd_ret)
    return label

def compute_directional_labels_vect(close, n_bars=3):
    """Vectorized directional label computation."""
    log_ret = np.diff(np.log(close))
    n = len(close)
    label = np.full(n, np.nan)
    if n <= n_bars:
        return label
    cs = np.zeros(n)
    cs[1:] = np.cumsum(log_ret)
    fwd = cs[n_bars:] - cs[:-n_bars]
    label[:n - n_bars] = np.sign(fwd)
    return label

def evaluate_directional_accuracy(feature_matrix, labels, feature_names, min_samples=100):
    n = len(labels)
    results = {}
    for j, name in enumerate(feature_names):
        feat = feature_matrix[:, j]
        valid = ~np.isnan(feat) & ~np.isnan(labels)
        if np.sum(valid) < min_samples:
            results[name] = {"accuracy": np.nan, "n": int(np.sum(valid)), "error": "too few samples"}
            continue
        feat_v = feat[valid]
        labels_v = labels[valid]
        median = np.nanmedian(feat_v)
        pred = np.where(feat_v > median, 1, -1)
        acc = np.mean(pred == labels_v)
        results[name] = {"accuracy": acc, "n": int(np.sum(valid)), "median": float(median)}
    return results

def compute_accuracy_by_quintile(feature, labels, n_quintiles=5):
    valid = ~np.isnan(feature) & ~np.isnan(labels)
    if np.sum(valid) < 100:
        return None
    fv = feature[valid]
    lv = labels[valid]
    n = len(fv)
    order = np.argsort(fv)
    qsize = n // n_quintiles
    median = np.median(fv)
    results = {}
    for q in range(n_quintiles):
        s = q * qsize
        e = n if q == n_quintiles - 1 else (q + 1) * qsize
        idx = order[s:e]
        pred = np.where(fv[idx] > median, 1, -1)
        acc = np.mean(pred == lv[idx])
        results[f"q{q}"] = {"feat_mean": float(np.mean(fv[idx])), "accuracy": acc, "n": len(idx)}
    return results

def cross_pair_evaluation(pair_results, feature_names):
    pairs = list(pair_results.keys())
    results = {}
    for name in feature_names:
        accs = []
        for p in pairs:
            acc = pair_results[p].get(name, {}).get("accuracy", np.nan)
            if not np.isnan(acc):
                accs.append(acc)
        if len(accs) >= 2:
            results[name] = {"mean_acc": float(np.mean(accs)), "std_acc": float(np.std(accs)),
                             "min_acc": float(np.min(accs)), "max_acc": float(np.max(accs)),
                             "stable": np.std(accs) < 0.03}
    return results
