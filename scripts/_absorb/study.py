"""scripts/_absorb/study.py — Hypothesis A/B: absorption -> reversal OR continuation.

For every hour-boundary bar i where the absorption->impact-transition state is
detected (causal, closed bars only), measure the FORWARD horizon returns from
the fill bar (i+1) with the transition direction d:

    cont(H) = mean( r_H * d ),  r_H = close(i+1+H) - open(i+1)

  positive cont  -> Hypothesis B (continuation: absorbed flow finally moves
                   price; price discovery persists)
  negative cont  -> Hypothesis A (reversal: the absorbed move mean-reverts)

Baselines:
  * RAW: same machinery on ALL hour-boundary bars (no absorption gate), same
    direction definition -> the natural drift of "prior T-bar displacement".
  * SHUFFLE: canonical purple null — per-symbol bar-order shuffle (24 iters x
    2 seeds), same gates, same conditional forward measure. z = (real-mean)/sd.

Horizons H in {6,12,24,48} (0.5h / 1h / 2h / 4h) — economically justified by
the holding periods the book already validates (12-24 M5 bars).
"""
from __future__ import annotations
import json, math, random, statistics as st
from typing import Optional

import numpy as np

from measures import AbsorbSignals

HORIZONS = [6, 12, 24, 48]
BLOCKS = {"Asia": range(0, 6), "London": range(7, 13),
          "NY": range(13, 21), "Late": range(21, 24)}


def _block(h: int) -> str:
    for name, hrs in BLOCKS.items():
        if h in hrs:
            return name
    return "off"


def forward_r(sig: AbsorbSignals, idx) -> dict:
    """Dict of arrays: forward close-return x dir per horizon for candidate bars."""
    out = {H: [] for H in HORIZONS}
    for i in idx:
        d = sig.dir_tr[i]
        for H in HORIZONS:
            r, _ = sig.forward_return(int(i), H)
            out[H].append(r * d)
    return {H: np.array(v) for H, v in out.items()}


def summarize(contrib: dict, tag: str) -> dict:
    out: dict = {"tag": tag}
    for H, arr in contrib.items():
        arr = np.asarray(arr, dtype=float)
        n = len(arr)
        if n == 0:
            out[str(H)] = {"n": 0, "mean": None, "t": None}
            continue
        m = float(arr.mean())
        sd = float(arr.std(ddof=1)) if n > 1 else 0.0
        out[str(H)] = {"n": n, "mean": round(m, 6),
                       "t": round(m / (sd / math.sqrt(n)), 2) if sd > 0 else None}
    return out


def study_symbol(sym: str, bars: list, W: int, T: int, D: int) -> dict:
    sig = AbsorbSignals(bars, W=W, T=T, D=D)
    idx = sig.signal_indices()
    hours = np_ts_hours(sig, idx)
    res: dict = {"symbol": sym, "n_signal": int(len(idx)),
           "gated": summarize(forward_r(sig, idx), "gated")}
    by_block = {}
    for name in BLOCKS:
        sel = idx[[h in BLOCKS[name] for h in hours]]
        by_block[name] = summarize(forward_r(sig, sel), name)
    res["by_block"] = by_block
    # RAW baseline: every hour-boundary bar with the same direction definition
    raw_idx = raw_indices(sig)
    res["raw"] = summarize(forward_r(sig, raw_idx), "raw")
    return res


def np_ts_hours(sig: AbsorbSignals, idx) -> list:
    return [int(sig.a["ts"][i]) // 3600 % 24 for i in idx]


def raw_indices(sig: AbsorbSignals) -> np.ndarray:
    """All hour-boundary bars with same-day window (no absorption gates)."""
    a = sig.a
    ts = a["ts"]
    day = ts // 86400
    out = []
    for i in range(sig.W + sig.T, len(ts)):
        if ts[i] % 3600 != 0:
            continue
        if not (day[i - 1] == day[i - sig.W - sig.T] == day[i - sig.T - 1]):
            continue
        if abs(sig.dis_tr[i]) <= 1e-12:
            continue
        out.append(i)
    return np.array(out, dtype=np.int64)


def random_null(sigs: dict, n_sig: int, n_iter: int = 1000, seed: int = 7) -> dict:
    """Honest null: same signal COUNT on random guarded hour-boundary bars with
    random +/-1 directions, over the REAL calendar (ts intact, forward returns
    real). Bar-order shuffle is degenerate here — the same-day guard makes
    shuffled signals ~impossible, so it cannot serve as the null."""
    rng = np.random.default_rng(seed)
    pool = {H: [] for H in HORIZONS}          # per-guarded-bar forward returns
    n_guarded = 0
    for s, sig in sigs.items():
        for i in raw_indices(sig):
            n_guarded += 1
            for H in HORIZONS:
                r, _ = sig.forward_return(int(i), H)
                pool[H].append(r)
    arr = {H: np.array(pool[H]) for H in HORIZONS}
    if n_guarded == 0:
        return {str(H): {"mean": 0.0, "sd": 0.0, "n_iter": 0} for H in HORIZONS}
    out = {}
    for H in HORIZONS:
        a = arr[H]
        means = np.empty(n_iter)
        for k in range(n_iter):
            idx = rng.integers(0, n_guarded, size=n_sig)
            sgn = rng.choice([-1.0, 1.0], size=n_sig)
            means[k] = float((a[idx] * sgn).mean())
        out[str(H)] = {"mean": round(float(means.mean()), 6),
                       "sd": round(float(means.std(ddof=1)), 6), "n_iter": n_iter}
    return out


def run(bars_map: dict, syms: list, W: int, T: int, D: int = 240,
        purples: bool = True, out_path: Optional[str] = None) -> dict:
    sigs = {s: AbsorbSignals(bars_map[s], W=W, T=T, D=D) for s in syms}
    res: dict = {"W": W, "T": T, "D": D, "symbols": syms,
                 "per_symbol": [study_symbol(s, bars_map[s], W, T, D) for s in syms]}
    # pooled gated vs raw
    gated = {H: [] for H in HORIZONS}
    raw = {H: [] for H in HORIZONS}
    n_sig = 0
    for s in syms:
        sig = sigs[s]
        for i in sig.signal_indices():
            n_sig += 1
            d = sig.dir_tr[i]
            for H in HORIZONS:
                r, _ = sig.forward_return(int(i), H)
                gated[H].append(r * d)
        for i in raw_indices(sig):
            d = sig.dir_tr[i]
            for H in HORIZONS:
                r, _ = sig.forward_return(int(i), H)
                raw[H].append(r * d)
    res["pooled_gated"] = summarize(gated, "gated")
    res["pooled_raw"] = summarize(raw, "raw")
    if purples:
        res["random_null"] = random_null(sigs, n_sig)
    if out_path:
        with open(out_path, "w") as f:
            json.dump({k: v for k, v in res.items()}, f, indent=2, default=str)
    return res