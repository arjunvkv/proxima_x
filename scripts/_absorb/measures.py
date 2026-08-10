"""scripts/_absorb/measures.py — causal absorption->impact-transition measures (M5 bars).

STANDALONE RESEARCH (2026-08-10). Read-only against the engine: bar tape via
feed.load_bars_cached, USD conversion via pnl.trade_to_usd, battery via
validation.*. ENGINE AND LIVE BOOK ARE NOT TOUCHED.

Phenomenon (Kyle-style liquidity rationing, retail-independent):
  * State A (absorption): unusually HIGH activity (sum of bar ranges) with
    unusually LOW price impact (|net displacement| per unit activity).
    Interpreted as flow being absorbed by available liquidity.
  * State B (transition): impact per unit activity EXPANDS (>= RATIO x the
    absorption impact) and price discovers direction (the transition window's
    net displacement with bar-level directional consistency).

Every quantity at signal bar `i` uses ONLY closed bars index <= i (no
lookahead, no repaint). Signal bars are hour-boundary bars (ts % 3600 == 0)
so a live worker polling hourly reproduces the cross-section exactly.
"""
from __future__ import annotations
import numpy as np

EPS = 1e-12


def build_arrays(bars) -> dict:
    """bar arrays from either list-of-dicts (engine tape) or dict-of-arrays."""
    if isinstance(bars, dict):
        ts = np.asarray(bars["ts"], dtype=np.int64)
        o = np.asarray(bars["open"], dtype=np.float64)
        h = np.asarray(bars["high"], dtype=np.float64)
        l = np.asarray(bars["low"], dtype=np.float64)
        c = np.asarray(bars["close"], dtype=np.float64)
    else:
        ts = np.array([b["ts"] for b in bars], dtype=np.int64)
        o = np.array([b["open"] for b in bars], dtype=np.float64)
        h = np.array([b["high"] for b in bars], dtype=np.float64)
        l = np.array([b["low"] for b in bars], dtype=np.float64)
        c = np.array([b["close"] for b in bars], dtype=np.float64)
    n = len(ts)
    rng = h - l
    cs = np.concatenate([[0.0], np.cumsum(rng)])          # cs[j] = sum rng[0:j]
    return {"ts": ts, "o": o, "h": h, "l": l, "c": c, "rng": rng, "cs": cs}


def _roll_mean_sd(x: np.ndarray, D: int) -> tuple[np.ndarray, np.ndarray]:
    """Rolling mean/sd of x over the previous D values (window ends at i-1).

    Out[i] = stats of x[i-D : i]. NaN where insufficient history.
    """
    n = len(x)
    xx = np.nan_to_num(x)
    sx = np.concatenate([[0.0], np.cumsum(xx)])     # sx[k] = sum of x[0:k], len n+1
    sx2 = np.concatenate([[0.0], np.cumsum(xx ** 2)])
    i = np.arange(n)
    lo = np.maximum(i - D, 0)
    cnt = (i - lo).astype(float)
    m = np.full(n, np.nan)
    sd = np.full(n, np.nan)
    valid = i >= D
    iv = i[valid]
    m[valid] = (sx[iv] - sx[lo[iv]]) / cnt[valid]
    var = (sx2[iv] - sx2[lo[iv]]) / cnt[valid] - m[valid] ** 2
    sd[valid] = np.sqrt(np.maximum(var, 0.0))
    return m, sd


class AbsorbSignals:
    """Vectorized per-bar signal machinery for one symbol's bar tape."""

    def __init__(self, bars: list[dict], W: int = 36, T: int = 4, D: int = 240):
        self.bars = bars
        self.W, self.T, self.D = W, T, D
        a = build_arrays(bars)
        self.a = a
        n = len(a["ts"])
        ts, o, c, cs = a["ts"], a["o"], a["c"], a["cs"]
        self.n = n

        # ---- per-bar (signal at i) quantities ----
        W, T = self.W, self.T
        self.act_abs = np.zeros(n); self.imp_abs = np.zeros(n)
        self.act_tr = np.zeros(n);  self.imp_tr = np.zeros(n)
        self.dis_tr = np.zeros(n);  self.dir_tr = np.zeros(n)
        self.ratio = np.zeros(n);   self.z_act = np.zeros(n)
        self.valid = np.zeros(n, dtype=bool)
        if n > W + T + 2:
            i0, i1 = W + T, n
            sl = slice(i0, i1)
            i = np.arange(i0, i1)
            self.act_abs[sl] = cs[i - T] - cs[i - W - T]
            self.act_tr[sl] = cs[i] - cs[i - T]
            d_abs = c[i - T - 1] - o[i - W - T]
            d_tr = c[i - 1] - o[i - T]
            self.dis_tr[sl] = d_tr
            with np.errstate(divide="ignore", invalid="ignore"):
                self.imp_abs[sl] = np.abs(d_abs) / np.maximum(self.act_abs[sl], EPS)
                self.imp_tr[sl] = np.abs(d_tr) / np.maximum(self.act_tr[sl], EPS)
            self.dir_tr[sl] = np.sign(d_tr)
            self.ratio[sl] = self.imp_tr[sl] / np.maximum(self.imp_abs[sl], EPS)

            # ---- trailing regime distributions (rolling, O(1) per bar) ----
            # A[j] = W-window activity ending at j (exclusive): sum rng[j-W:j]
            j = np.arange(n)
            A = np.where(j >= W, cs[j] - cs[j - W], 0.0)
            mA, sA = _roll_mean_sd(A, D)
            self.z_act[sl] = (self.act_abs[sl] - mA[sl]) / np.maximum(sA[sl], EPS)
            # B[j] = W-window impact ending at j (exclusive) — matches imp_abs at j=i-T
            B = np.zeros(n)
            Bj = np.where(j >= W + 1, np.abs(c[j - 1] - o[j - W]) /
                          np.maximum(cs[j] - cs[j - W], EPS), 0.0)
            B[:n] = Bj
            mB, sB = _roll_mean_sd(B, D)
            # E[j] = T-window efficiency ending at j (exclusive) — matches imp_tr at j=i
            E = np.zeros(n)
            Ej = np.where(j >= T, np.abs(c[j - 1] - o[j - T]) /
                          np.maximum(cs[j] - cs[j - T], EPS), 0.0)
            E[:n] = Ej
            mE, sE = _roll_mean_sd(E, D)

            # ---- gates (distribution-derived, not hand-tuned constants) ----
            day = ts // 86400
            hour_edge = ts % 3600 == 0
            same_day = np.zeros(n, dtype=bool)
            same_day[sl] = (day[i - 1] == day[i - W - T]) & (day[i - W - T] == day[i - T - 1])
            non_zero = np.abs(d_tr) > EPS
            # absorption: activity unusually high vs regime; impact unusually low
            gate_act = self.z_act[sl] >= 1.0
            gate_abs = self.imp_abs[sl] <= (mB[i - T] - 0.25 * sB[i - T])
            # transition: efficiency unusually high vs regime; ratio >= 2.0
            gate_tr = self.imp_tr[sl] >= (mE[sl] + 0.5 * sE[sl])
            gate_ratio = self.ratio[sl] >= 2.0
            # flow persists into the transition (activity not collapsed)
            gate_persist = np.zeros(n, dtype=bool)
            gate_persist[sl] = (self.act_tr[sl] / T) >= 0.5 * (self.act_abs[sl] / W)

            self.valid[sl] = (hour_edge[sl] & same_day[sl] & non_zero
                              & gate_act & gate_abs & gate_tr
                              & gate_ratio & gate_persist[sl])

        # directional consistency (loop — only surviving bars are cheap)
        self.consistent = np.zeros(n, dtype=bool)
        self.score = np.zeros(n)
        frac = 0.6
        for ii in np.nonzero(self.valid)[0]:
            d = self.dir_tr[ii]
            seg = slice(ii - T, ii)
            sgn = np.sign(self.a["c"][seg] - self.a["o"][seg])
            ok_last = np.sign(self.a["c"][ii - 1] - self.a["o"][ii - 1]) == d
            if (sgn == d).mean() >= frac and ok_last:
                self.consistent[ii] = True
                self.score[ii] = d * max(self.ratio[ii] - 1.0, 0.0)

    def signal_indices(self) -> np.ndarray:
        return np.nonzero(self.valid & self.consistent)[0]

    def forward_return(self, i: int, H: int) -> tuple[float, float]:
        """Causal forward horizon from the FILL bar (i+1): close and open."""
        if i + 1 + H >= self.n:
            return 0.0, 0.0
        e = self.a["o"][i + 1]
        return self.a["c"][i + 1 + H] - e, self.a["o"][i + 1 + H] - e