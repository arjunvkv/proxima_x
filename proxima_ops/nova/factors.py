"""NOVA factors — vectorized indicator/factor tables over numpy bars.

Every function returns a float64 array aligned to the input bars (NaN where
undefined at the head). Windows use the SAME conventions as the legacy engine
(proxima_ops/backtest/engine.py) so NOVA scores are bit-comparable.
"""
import numpy as np


def bar_hour(ts: np.ndarray) -> np.ndarray:
    return (ts // 3600) % 24


def bar_day(ts: np.ndarray) -> np.ndarray:
    return ts // 86400


def weekday(ts: np.ndarray) -> np.ndarray:
    """0=Mon..6=Sun (matches engine: (day+3)%7 with epoch 1970-01-01=Thu)."""
    return (bar_day(ts) + 3) % 7


def ret_series(close: np.ndarray, lb: int) -> np.ndarray:
    """(close[i] - close[i-lb]) / close[i-lb] — engine _ret, vectorized."""
    out = np.full(close.shape, np.nan)
    if len(close) > lb:
        out[lb:] = (close[lb:] - close[:-lb]) / close[:-lb]
    return out


def trailing_atr(o, h, l, c, w: int = 168) -> np.ndarray:
    """Engine _trailing_atr (n_avg=14, n_range=12): mean TR over the trailing
    `w` closed bars strictly before i, range starts at k=max(1, i-w)."""
    n = len(c)
    tr = np.zeros(n)
    if n > 1:
        prev = np.roll(c, 1)
        prev[0] = c[0]
        tr[1:] = np.maximum(h[1:] - l[1:],
                            np.maximum(np.abs(h[1:] - prev[1:]),
                                       np.abs(l[1:] - prev[1:])))
    cs = np.concatenate([[0.0], np.cumsum(tr)])
    out = np.zeros(n)
    for i in range(1, n):
        lo = max(1, i - w)
        cnt = i - lo
        if cnt > 0:
            out[i] = (cs[i] - cs[lo]) / cnt
    return out


def rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    """Mean of x[max(0,i-w):i] — strictly-prior window ending at bar i-1."""
    n = len(x)
    cs = np.concatenate([[0.0], np.cumsum(np.nan_to_num(x, nan=0.0))])
    out = np.full(n, np.nan)
    for i in range(1, n):
        lo = max(0, i - w)
        out[i] = (cs[i] - cs[lo]) / (i - lo)
    out[0] = 0.0
    return out


def rolling_max(x: np.ndarray, w: int) -> np.ndarray:
    n = len(x)
    out = np.full(n, np.nan)
    for i in range(1, n):
        out[i] = x[max(0, i - w):i].max()
    return out


def rolling_min(x: np.ndarray, w: int) -> np.ndarray:
    n = len(x)
    out = np.full(n, np.nan)
    for i in range(1, n):
        out[i] = x[max(0, i - w):i].min()
    return out


def rolling_std(x: np.ndarray, w: int) -> np.ndarray:
    n = len(x)
    out = np.full(n, np.nan)
    for i in range(1, n):
        seg = x[max(0, i - w):i]
        if len(seg) >= 2:
            out[i] = seg.std()
    return out


def rolling_z(x: np.ndarray, w: int) -> np.ndarray:
    mu = rolling_mean(x, w)
    sd = rolling_std(x, w)
    return (x - mu) / np.where(np.isnan(sd) | (sd == 0), np.nan, sd)


def rolling_percentile(x: np.ndarray, w: int) -> np.ndarray:
    """Fraction of trailing `w` prior values <= current (0..1), NaN at head."""
    n = len(x)
    out = np.full(n, np.nan)
    for i in range(1, n):
        lo = max(0, i - w)
        seg = x[lo:i]
        out[i] = float((seg <= x[i]).mean())
    return out


def trend_efficiency(close: np.ndarray, k: int) -> np.ndarray:
    """|ret k| / sum(|1-bar ret| over the window) — 1 = pure trend, 0 = chop."""
    n = len(close)
    out = np.full(n, np.nan)
    if n <= k:
        return out
    d = np.abs(np.diff(close))
    retk = np.abs(close[k:] - close[:-k])
    for i in range(k, n):
        path = d[i - k:i].sum()
        out[i] = retk[i - k] / path if path > 0 else np.nan
    return out


def session_open_idx(ts: np.ndarray) -> np.ndarray:
    """Index of the first bar of the UTC day containing each bar (engine
    session_start_idx, vectorized)."""
    d = bar_day(ts)
    first = np.empty(len(ts), dtype=np.int64)
    first[0] = 0
    for i in range(1, len(ts)):
        first[i] = i if d[i] != d[i - 1] else first[i - 1]
    return first


def gap_open(o: np.ndarray, c: np.ndarray, ts: np.ndarray) -> np.ndarray:
    """Session-open gap = (open[i] - close of last bar of prior day)/prior close;
    0.0 for bars that do not open a new day (engine _wc_gap semantics)."""
    n = len(ts)
    out = np.zeros(n)
    d = bar_day(ts)
    for i in range(1, n):
        if d[i] != d[i - 1]:
            out[i] = (o[i] - c[i - 1]) / c[i - 1]
    return out
