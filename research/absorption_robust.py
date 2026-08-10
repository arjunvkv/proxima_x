"""Robustness: is the strict-absorption reversal concentrated in one period?

Takes the strictest config (PQ=0.95, LQ=0.10, W=60) and checks the 300s/900s
signed forward return per day + leave-one-day-out t-stat, so we can tell
"one lucky week" from an effect that shows up across the tape.
"""
from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl

os.environ.setdefault("PROBE_W", "60")
os.environ.setdefault("PROBE_TRAIL", "14400")
os.environ.setdefault("PROBE_PQ", "0.95")
os.environ.setdefault("PROBE_LQ", "0.10")
os.environ.setdefault("PROBE_MINP", "10")

import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "absorption_probe", "research/absorption_probe.py"
)
probe = importlib.util.module_from_spec(spec)
# guard: main() only runs under __name__ == "__main__"
spec.loader.exec_module(probe)

W, TRAIL = probe.W, probe.TRAIL
HORIZONS = (60, 300, 900)
s = probe.load_1s(probe.SYM, probe.ARCHIVE)
t = s["time_sec"].to_numpy()
P = s["pressure"].to_numpy().astype(float)
dM = s["dM"].to_numpy()
mid = s["mid_last"].to_numpy()

n_win = int(np.floor(len(t) / W))
Pw = P[: n_win * W].reshape(n_win, W).sum(axis=1)
dMw = dM[: n_win * W].reshape(n_win, W).sum(axis=1)
tw = t[: n_win * W].reshape(n_win, W)[:, -1]
lam = np.abs(dMw) / (np.abs(Pw) + 1.0)
n_trail = TRAIL // W
valid = np.zeros(n_win, bool)
p90_absP = np.full(n_win, np.nan)
q10_lam = np.full(n_win, np.nan)
for i in range(n_win):
    if i < n_trail:
        continue
    a = max(0, i - n_trail)
    ha = np.abs(Pw[a:i])
    hl = lam[a:i]
    ha = ha[~np.isnan(ha)]
    hl = hl[~np.isnan(hl)]
    if len(ha) < 50 or len(hl) < 50:
        continue
    p90_absP[i] = np.quantile(ha, probe.P_QUANT)
    q10_lam[i] = np.quantile(hl, probe.L_QUANT)
    valid[i] = True

abs_state = (valid & (np.abs(Pw) >= p90_absP) & (np.abs(Pw) >= probe.MIN_PRESSURE)
             & (lam <= q10_lam))
print(f"strict absorption states: {int(abs_state.sum())} over {int(valid.sum())} windows")

idx0 = np.searchsorted(t, tw)
fwd = {}
for H in HORIZONS:
    fwd[H] = np.full(n_win, np.nan)
    idx = np.searchsorted(t, tw + H)
    ok = idx < len(mid)
    fwd[H][ok] = mid[idx[ok]] - mid[idx0[ok]]

days = tw[abs_state] // 86400
print(f"days with states: {len(Counter(days))} of ~20")

for H in HORIZONS:
    m = abs_state & ~np.isnan(fwd[H])
    n = int(m.sum())
    if n < 20:
        print(f"H={H}s: n={n} (skip)")
        continue
    signed = fwd[H][m] * np.sign(Pw[m])  # - = reversal
    pt = 0.001
    print(f"\nH={H}s  n={n}  mean_signed={signed.mean()/pt:+.1f}pts "
          f"(reversal={-signed.mean()/pt:+.1f}pts)")
    # per-day means
    dkey = (tw[m] // 86400)
    for d in sorted(set(dkey)):
        mm = dkey == d
        print(f"  day {d}: n={int(mm.sum()):3d} mean_signed={signed[mm].mean()/pt:+8.1f}pts")
    # leave-one-day-out t for the reversal sign flip
    def tstat1(x):
        n = len(x)
        if n < 3:
            return 0.0
        m, sd = x.mean(), x.std(ddof=1)
        return m / (sd / np.sqrt(n)) if sd > 0 else 0.0

    all_p = signed / pt
    for d in sorted(set(dkey)):
        keep = dkey != d
        tstat = tstat1(-all_p[keep])
        if tstat < 0:
            print(f"  LODO day {d}: reversal t={tstat:+.2f}  (flips to continuation!)")
        else:
            print(f"  LODO day {d}: reversal t={tstat:+.2f}")
    t_all = tstat1(-all_p)
    print(f"  ALL days: reversal t={t_all:+.2f}")