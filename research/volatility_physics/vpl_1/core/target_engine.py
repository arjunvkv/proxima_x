import numpy as np
from scipy.stats import entropy, linregress
import logging

logger = logging.getLogger("proxima_ops.vpl")

DATA_BASE = "C:/Trading/Agentic_Trading/data/intraday"
SYMBOLS_FULL = ["EURJPY", "USDJPY", "GBPJPY", "XAUUSD"]


class DataIntegrityError(Exception):
    pass


def load_m5(symbol):
    import polars as pl
    path = f"{DATA_BASE}/{symbol}_M5.parquet"
    df = pl.read_parquet(path)
    arr = df.to_numpy()
    if arr.shape[1] < 6:
        raise DataIntegrityError(
            f"{symbol}: expected >= 6 columns (timestamp,open,high,low,close,volume), got {arr.shape[1]}")
    ts = arr[:, 0].astype(np.int64)
    o, h, l, c, v = [arr[:, i].astype(np.float64) for i in range(1, 6)]
    return {
        "symbol": symbol,
        "timestamp": ts,
        "open": o, "high": h, "low": l, "close": c, "volume": v,
        "n": len(c),
    }


def compute_returns(close):
    return np.diff(np.log(close))


def realized_variance(returns, window):
    n = len(returns)
    r2 = returns ** 2
    r_cum = np.zeros(n + 1)
    r2_cum = np.zeros(n + 1)
    r_cum[1:] = np.cumsum(returns)
    r2_cum[1:] = np.cumsum(r2)
    out = np.full(n, np.nan)
    for i in range(window, n):
        s = r_cum[i] - r_cum[i - window]
        s2 = r2_cum[i] - r2_cum[i - window]
        out[i] = s2 / window - (s / window) ** 2
    return out


def variance_decay_slope(values, window):
    n = len(values)
    decay = np.full(n, np.nan)
    x = np.arange(window, dtype=np.float64)
    x_mean = x.mean()
    x2_sum = (x * x).sum()
    denom = x2_sum - window * x_mean * x_mean
    if denom < 1e-12:
        return decay
    for i in range(window, n):
        y = values[i - window:i]
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            continue
        y_mean = y.mean()
        slope = (x * y).sum() - window * x_mean * y_mean
        decay[i] = slope / denom
    return decay


def compute_expansion_labels(close, baseline_window=12, horizons=(3, 6, 12), thresholds=(1.5, 2.0, 3.0)):
    r = compute_returns(close)
    n = len(r)
    result = {}
    for N in horizons:
        bv = realized_variance(r, baseline_window)
        fv = realized_variance(r, N)
        er = np.full(n, np.nan)
        max_i = n - N - 1
        for i in range(baseline_window, max_i):
            if bv[i] > 0 and not np.isnan(bv[i]) and not np.isnan(fv[i + 1]):
                er[i] = fv[i + 1] / bv[i]
        for thresh in thresholds:
            label = np.full(n, np.nan)
            valid = ~np.isnan(er)
            label[valid] = (er[valid] >= thresh).astype(np.float64)
            result[f"expand_{thresh}_{N}"] = {"er": er, "label": label}
    return result


def crf_entropy_collapse(close, window=24, bins=20):
    r = compute_returns(close)
    n = len(r)
    global_p5, global_p95 = np.percentile(r, [5, 95])
    edges = np.linspace(global_p5, global_p95, bins + 1)
    bin_idx = np.digitize(r, edges) - 1
    bin_idx = np.clip(bin_idx, 0, bins - 1)
    one_hot = np.zeros((n, bins))
    one_hot[np.arange(n), bin_idx] = 1.0
    cum = np.zeros((n + 1, bins))
    cum[1:] = np.cumsum(one_hot, axis=0)
    ent = np.full(n, np.nan)
    for i in range(window, n):
        counts = cum[i] - cum[i - window]
        total = counts.sum()
        if total < 1 or counts.max() == total:
            ent[i] = 0.0
            continue
        probs = counts[counts > 0] / total
        ent[i] = -np.sum(probs * np.log(probs))
    ent_z = (ent - np.nanmean(ent)) / np.nanstd(ent)
    ent_close = np.full(len(close), np.nan)
    ent_close[1:] = ent
    ent_z_close = np.full(len(close), np.nan)
    ent_z_close[1:] = ent_z
    return ent_close, ent_z_close


def crf_range_persistence(close, high, low, window=24):
    range_vals = high - low
    n = len(range_vals)
    streak = np.full(n, np.nan)
    windows = np.lib.stride_tricks.sliding_window_view(range_vals, window)
    sorted_w = np.sort(windows, axis=1)
    thresholds = sorted_w[:, window // 4]
    streak[:window] = np.nan
    for i in range(window, n):
        thresh = thresholds[i - window]
        s = 0
        for j in range(i, -1, -1):
            if range_vals[j] < thresh:
                s += 1
            else:
                break
        streak[i] = s
    return streak


def crf_variance_decay(close, window=24):
    r = compute_returns(close)
    rv = realized_variance(r, window)
    decay_r = variance_decay_slope(rv, window)
    decay = np.full(len(close), np.nan)
    decay[1:] = decay_r
    return decay


def crf_path_inefficiency(close, window=12):
    r = compute_returns(close)
    n = len(r)
    pi = np.full(n, np.nan)
    for i in range(window, n):
        total_path = np.sum(np.abs(r[i - window + 1:i + 1]))
        net_move = np.abs(np.log(close[i + 1] / close[i - window + 1]))
        if net_move > 1e-12:
            pi[i] = total_path / net_move
        else:
            pi[i] = total_path / 1e-12
    pi_out = np.full(len(close), np.nan)
    pi_out[1:] = pi
    return pi_out


def normalize(series):
    s = series.copy()
    s = np.where(np.isinf(s), np.nan, s)
    mu = np.nanmean(s)
    std = np.nanstd(s)
    if std < 1e-12:
        return np.zeros_like(s)
    return (s - mu) / std


def compute_crf(close, high, low, entropy_window=24, variance_window=24, pi_window=12, range_window=24):
    ent, ent_z = crf_entropy_collapse(close, window=entropy_window)
    persistence = crf_range_persistence(close, high, low, window=range_window)
    decay = crf_variance_decay(close, window=variance_window)
    pi = crf_path_inefficiency(close, window=pi_window)
    crf_val = normalize(-ent_z) * 0.25 + normalize(persistence) * 0.25 + normalize(-decay) * 0.25 + normalize(pi) * 0.25
    return {
        "entropy": ent,
        "entropy_z": ent_z,
        "range_persistence": persistence,
        "variance_decay": decay,
        "path_inefficiency": pi,
        "crf": crf_val,
        "crf_pct": np.round(np.nanpercentile(crf_val, np.arange(0, 101, 10)), 3),
    }


def compute_crf_deciles(crf_val):
    n = len(crf_val)
    decile = np.full(n, np.nan)
    valid = ~np.isnan(crf_val)
    ranks = np.argsort(np.argsort(crf_val[valid]))
    decile[valid] = np.floor(ranks / np.sum(valid) * 10).astype(int)
    decile[valid] = np.clip(decile[valid], 0, 9)
    return decile


def compute_decile_expansion_profile(decile, label):
    profile = []
    for d in range(10):
        mask = decile == d
        freq = np.nanmean(label[mask]) if np.sum(mask) > 0 else np.nan
        count = int(np.sum(mask))
        profile.append({"decile": d, "freq": freq, "count": count})
    return profile


def compute_base_expansion_rate(close, horizons=(3, 6, 12), thresholds=(1.5, 2.0, 3.0)):
    result = compute_expansion_labels(close, horizons=horizons, thresholds=thresholds)
    rates = {}
    for key, val in result.items():
        rates[key] = np.nanmean(val["label"])
    return rates


def compute_log_returns(close):
    return compute_returns(close)


def realized_variance_from_log_returns(log_ret, window):
    n = len(log_ret)
    r2 = log_ret ** 2
    r_cum = np.zeros(n + 1)
    r2_cum = np.zeros(n + 1)
    r_cum[1:] = np.cumsum(log_ret)
    r2_cum[1:] = np.cumsum(r2)
    out = np.full(n, np.nan)
    for i in range(window, n):
        s = r_cum[i] - r_cum[i - window]
        s2 = r2_cum[i] - r2_cum[i - window]
        out[i] = s2 / window - (s / window) ** 2
    return out


def compute_forward_variance_expansion(log_ret, baseline_window=24, forward_horizons=(12,)):
    n = len(log_ret)
    results = {}
    for N in forward_horizons:
        bv = realized_variance_from_log_returns(log_ret, baseline_window)
        fv = realized_variance_from_log_returns(log_ret, N)
        er = np.full(n, np.nan)
        max_i = n - N - 1
        for i in range(baseline_window, max_i):
            if bv[i] > 0 and not np.isnan(bv[i]) and not np.isnan(fv[i + 1]):
                er[i] = fv[i + 1] / bv[i]
        results[N] = {
            "baseline_var": bv,
            "forward_var": fv,
            "expansion_ratio": er,
            "mean_er": np.nanmean(er),
            "std_er": np.nanstd(er),
            "p25": np.nanpercentile(er, 25),
            "p50": np.nanpercentile(er, 50),
            "p75": np.nanpercentile(er, 75),
            "p90": np.nanpercentile(er, 90),
        }
    return results
