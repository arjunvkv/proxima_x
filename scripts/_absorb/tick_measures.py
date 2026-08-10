"""scripts/_absorb/tick_measures.py — flow-aware absorption->impact measures on
1-min bars built from genuine trade ticks (see tick_pull.py).

Price-only variant (T1): identical logic to measures.AbsorbSignals on minute
bars — direction from T-window displacement, impact from price activity.

Flow-aware variant (T2): the faithful institutional test —
  flow_abs = net signed flow over the W-window (tick-rule aggressor counts)
  ky λ_abs = |displacement| / (|net flow| + ε)   <- Kyle-style price impact
  a low λ with high |flow| = absorption (directional flow, small price move)
  transition = λ expands materially over the T-window with net flow present
  direction d = sign(net flow over T)

The 'flow' array (net signed flow per minute) is the only addition vs the M5
machinery. Everything else (z-score gates, hour-boundary, same-day guard,
consistent-flow requirement) mirrors measures.py so T1 vs T2 vs the M5 study
are definitionally comparable.

All windows/regimes are in MINUTES. Non-repainting, causal, executable at the
closed-bar timestamp.
"""
from __future__ import annotations
from typing import Optional

import numpy as np

EPS = 1e-9
HORIZONS_FWD = [30, 60, 120, 240]   # forward minutes (matches M5 H=6/12/24/48 x 5m)
BLOCKS = {"Asia": [0, 6], "London": [7, 12], "NY": [13, 20], "Late": [21, 23]}


class TickAbsorbSignals:
    """Causal absorption->impact-transition state on minute bars (flow-aware)."""

    def __init__(self, bars, flow, W: int = 36, T: int = 6, D: int = 240,
                 use_flow: bool = True):
        ts0 = np.asarray(bars["ts"], dtype=np.int64)
        n = len(ts0)
        ts = ts0
        o = np.asarray(bars["open"], dtype=np.float64)
        h = np.asarray(bars["high"], dtype=np.float64)
        l = np.asarray(bars["low"], dtype=np.float64)
        c = np.asarray(bars["close"], dtype=np.float64)
        rng = h - l + EPS
        self.a = {"ts": ts, "open": o, "high": h, "low": l, "close": c}
        self.n = len(ts)
        self.use_flow = use_flow
        self.flow = np.asarray(flow, dtype=np.float64) if flow is not None else None

        cs = np.concatenate([[0.0], np.cumsum(rng)])       # activity cumsum
        cf = (np.concatenate([[0.0], np.cumsum(self.flow)])
              if self.flow is not None else None)          # flow cumsum
        if self.flow is not None:
            caf = np.concatenate([[0.0], np.cumsum(np.abs(self.flow))])

        self.act_abs = np.zeros(n); self.flow_abs = np.zeros(n)
        self.act_tr = np.zeros(n);  self.flow_tr = np.zeros(n)
        self.imp_abs = np.zeros(n); self.imp_tr = np.zeros(n)
        self.lam_abs = np.zeros(n); self.lam_tr = np.zeros(n)
        self.dir_tr = np.zeros(n, dtype=np.int8)
        self.ratio = np.zeros(n)
        self.score = np.zeros(n)
        self.z_act = np.full(n, np.nan)
        self.valid = np.zeros(n, dtype=bool)
        if n <= W + T + 2:
            return

        # window aggregates (full-range vectorized, like measures.py)
        j = np.arange(n)
        A = np.where(j >= W, cs[j] - cs[j - W], 0.0)       # W activity
        E = np.where(j >= T, cs[j] - cs[j - T], 0.0)       # T activity
        dispA = np.where(j >= W + 1, c[j - 1] - o[j - W], 0.0)
        dispE = np.where(j >= T + 1, c[j - 1] - o[j - T], 0.0)
        if self.flow is not None:
            fA = np.where(j >= W, cf[j] - cf[j - W], 0.0)  # net flow W
            absA = np.where(j >= W, caf[j] - caf[j - W], 0.0)
            fE = np.where(j >= T, cf[j] - cf[j - T], 0.0)
        mA, sA = _roll_mean_sd(A, D)
        mB, sB = _roll_mean_sd(
            np.where(j >= W + 1, np.abs(dispA) / np.maximum(A, EPS), 0.0), D)
        # λ regime: price move per unit |net flow| (W-window, trailing D)
        if self.flow is not None:
            lamA = np.where(j >= W, np.abs(dispA) / (np.abs(fA) + EPS), np.nan)
            mLa, sLa = _roll_mean_sd(np.nan_to_num(lamA, nan=0.0), D)
            lamE = np.where(j >= T, np.abs(dispE) / (np.abs(fE) + EPS), np.nan)
        mE, sE = _roll_mean_sd(E, D)

        i0, i1 = W + T, n
        sl = slice(i0, i1)
        i = np.arange(i0, i1)
        # signal window aggregates for the T-moment bars
        self.act_abs[sl] = cs[i - T] - cs[i - W - T]
        self.act_tr[sl] = cs[i] - cs[i - T]
        self.imp_abs[sl] = np.abs(c[i - T - 1] - o[i - W - T]) / np.maximum(
            self.act_abs[sl], EPS)
        self.imp_tr[sl] = np.abs(c[i - 1] - o[i - T]) / np.maximum(
            self.act_tr[sl], EPS)
        disp_tr = c[i - 1] - o[i - T]
        if self.flow is not None:
            self.flow_abs[sl] = cf[i - T] - cf[i - W - T]
            self.flow_tr[sl] = cf[i] - cf[i - T]
            self.lam_abs[sl] = np.abs(c[i - T - 1] - o[i - W - T]) / np.maximum(
                np.abs(self.flow_abs[sl]), EPS)
            self.lam_tr[sl] = np.abs(c[i - 1] - o[i - T]) / np.maximum(
                np.abs(self.flow_tr[sl]), EPS)
        self.z_act[sl] = (self.act_abs[sl] - mA[i - T]) / (sA[i - T] + EPS)
        if self.use_flow and self.flow is not None:
            self.dir_tr[sl] = np.sign(self.flow_tr[sl])
            base_ratio = self.lam_tr[sl] / np.maximum(self.lam_abs[sl], EPS)
        else:
            self.dir_tr[sl] = np.sign(disp_tr)
            base_ratio = self.imp_tr[sl] / np.maximum(self.imp_abs[sl], EPS)
        self.ratio[sl] = base_ratio
        self.score[sl] = np.where(
            base_ratio > 1.0, self.dir_tr[sl] * np.maximum(base_ratio - 1.0, 0.0), 0.0)

        # gates (all causal; distributions trailing the signal moment)
        hour_edge = ts % 3600 == 0
        day = ts // 86400
        same_day = np.zeros(n, dtype=bool)
        same_day[sl] = (day[i - 1] == day[i - W - T]) & (day[i - W - T] == day[i - T - 1])
        non_zero = np.abs(disp_tr) > EPS
        gate_act = self.z_act[sl] >= 1.0
        if self.use_flow and self.flow is not None:
            gate_abs = self.lam_abs[sl] <= (mLa[i - T] - 0.25 * sLa[i - T])
            gate_tr = self.lam_tr[sl] >= (mLa[i - T] + 0.5 * sLa[i - T])
            # net flow must be material in BOTH windows (directional pressure)
            gate_flow = (np.abs(self.flow_abs[sl]) >= 1.0) & (np.abs(self.flow_tr[sl]) >= 1.0)
        else:
            gate_abs = self.imp_abs[sl] <= (mB[i - T] - 0.25 * sB[i - T])
            gate_tr = self.imp_tr[sl] >= (mE[i - T] + 0.5 * sE[i - T])
            gate_flow = np.ones(len(i), dtype=bool)
        gate_ratio = base_ratio >= 2.0
        gate_persist = np.zeros(n, dtype=bool)
        gate_persist[sl] = (self.act_tr[sl] / T) >= 0.5 * (self.act_abs[sl] / W)
        self.valid[sl] = (hour_edge[sl] & same_day[sl] & non_zero & gate_act
                          & gate_abs & gate_tr & gate_flow & gate_ratio
                          & gate_persist[sl])

    def signal_indices(self) -> np.ndarray:
        return np.flatnonzero(self.valid)

    def forward_return(self, i: int, H: int):
        """Close-to-open forward 'H' minutes; (ret_abs, direction_flag)."""
        if i + 1 + H < self.n:
            return self.a["close"][i + 1 + H] - self.a["open"][i + 1], 1
        return 0.0, 0


def _roll_mean_sd(x: np.ndarray, D: int) -> tuple[np.ndarray, np.ndarray]:
    """Rolling mean/sd of x over the previous D values (window ends at i-1)."""
    n = len(x)
    x0 = np.nan_to_num(x)
    sx = np.concatenate([[0.0], np.cumsum(x0)])
    sx2 = np.concatenate([[0.0], np.cumsum(x0 * x0)])
    m = np.full(n, np.nan)
    sd = np.full(n, np.nan)
    i = np.arange(n)
    lo = np.maximum(i - D, 0)
    valid = i >= D
    m[valid] = (sx[i[valid]] - sx[lo[valid]]) / D
    var = (sx2[i[valid]] - sx2[lo[valid]]) / D - m[valid] ** 2
    sd[valid] = np.sqrt(np.maximum(var, 0.0))
    return m, sd


def raw_indices(sig: TickAbsorbSignals) -> np.ndarray:
    """Guarded hour-boundary bars (regime + day guard, no state gates)."""
    out = []
    ts = sig.a["ts"]
    day = ts // 86400
    for i in range(sig.n):
        if ts[i] % 3600 != 0 or i < 300:
            continue
        if not (day[i - 1] == day[i - 290] == day[i - 5]):
            continue
        if sig.a["close"][i - 1] == sig.a["open"][i - 4]:
            continue
        out.append(i)
    return np.array(out, dtype=np.int64)