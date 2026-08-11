"""s4_aligned_test.py — the HONEST causal test: same-day aligned residual.

The certified battery positional-aligned dc[-n:] + ec[-n:], but the cached
EURUSD file ends 3 trading days before DXY.cash, so the cert paired DXY(t)
with EURUSD(t-3). Here we intersect day axes (same-day pairing) and re-run:
  * cached   : DXY + EURUSD from cache, day-aligned (overlap window)
  * terminal : DXY + EURUSD from live terminal, day-aligned
If the edge (PF ~11.8, n=33) survives same-day alignment, S4 is real.
If it collapses, the certified edge was a stale-file lag artifact.
"""
import sys, os, json
import numpy as np

sys.path.insert(0, "scripts"); sys.path.insert(0, "scripts/_absorb"); sys.path.insert(0, ".")
from proxima_ops.backtest.feed import load_bars_cached
from proxima_ops.nova import feed as nfeed
from proxima_ops.nova import factors as F

def lastc(s):
    b = nfeed.bars_list_to_arrays(load_bars_cached(s))
    d = F.bar_day(b["ts"])
    dd = np.unique(d)
    return dd, b["open"][np.searchsorted(d, dd, side="left")], \
        b["close"][np.searchsorted(d, dd, side="right") - 1]

def rolling_z(x, w=20):
    out = np.full(len(x), np.nan)
    for i in range(1, len(x)):
        seg = x[max(0, i - w): i]
        sd = seg.std()
        out[i] = (x[i] - seg.mean()) / sd if sd > 0 else 0.0
    return out

def evaluate(days, odx, dcx, ode, ece, tag):
    r = np.log(dcx) + 0.576 * np.log(ece)
    z = rolling_z(r, 20)
    sig = np.where(z < -1.5, 1.0, np.where(z > 1.5, -1.0, 0.0))
    idx = np.where(sig[:-2] != 0)[0]
    nets, tdays, dirs = [], [], []
    for k in idx:
        d = sig[k]
        pts = (ode[k + 2] - ode[k + 1]) * d
        nets.append(pts); tdays.append(int(days[k + 1])); dirs.append("L" if d > 0 else "S")
    nets = np.array(nets) if nets else np.array([0.0])
    w = sum(1 for x in nets if x > 0) / len(nets) * 100
    g = nets[nets > 0].sum(); l = -nets[nets < 0].sum()
    pf = g / l if l > 0 else float("inf")
    print(f"[{tag}] n={len(nets)} win={w:.1f}% pts_sum={nets.sum():+.3f} avg={nets.mean():+.4f} PF={pf:.2f}")
    print(f"[{tag}] trades(day,dir): {list(zip(tdays, dirs))}")
    return tdays

def main_cached():
    dd, od, dc = lastc("DXY.cash")
    de, oe, ec = lastc("EURUSD")
    common = np.intersect1d(dd, de)
    i1 = np.searchsorted(dd, common); i2 = np.searchsorted(de, common)
    print(f"[cache] DXY n={len(dd)} {dd[0]}..{dd[-1]}  EUR n={len(de)} {de[0]}..{de[-1]}  overlap={len(common)}")
    evaluate(common, od[i1], dc[i1], oe[i2], ec[i2], "cache-aligned")

def main_terminal():
    T = json.load(open(r"C:\Users\arjun\AppData\Local\Temp\s4_term_closes.json"))
    td = np.array(T["DXY.cash"]["days"]); to = np.array(T["DXY.cash"]["open"]); tc = np.array(T["DXY.cash"]["close"])
    ed = np.array(T["EURUSD"]["days"]); eo = np.array(T["EURUSD"]["open"]); ec2 = np.array(T["EURUSD"]["close"])
    common = np.intersect1d(td, ed)
    i1 = np.searchsorted(td, common); i2 = np.searchsorted(ed, common)
    print(f"[term] DXY n={len(td)} {td[0]}..{td[-1]}  EUR n={len(ed)} {ed[0]}..{ed[-1]}  overlap={len(common)}")
    evaluate(common, to[i1], tc[i1], eo[i2], ec2[i2], "terminal-aligned")

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("cached", "both"):
        main_cached()
    if which in ("terminal", "both"):
        main_terminal()
