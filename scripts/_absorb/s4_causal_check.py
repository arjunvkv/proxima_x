"""s4_causal_check.py — can S4 fire from symbols ALREADY live on the worker?

The DXY identity: log(DXY) = ln(50.1435) - 0.576*ln(EURUSD) + 0.136*ln(USDJPY)
- 0.119*ln(GBPUSD) + 0.091*ln(USDCAD) + 0.042*ln(USDSEK) + 0.036*ln(USDCHF).

battery_dxy's residual r = log(DXY) + 0.576*log(EURUSD) - mean, so the EURUSD
term CANCELS: r is computable from the non-EUR components alone. If the
reconstructed r tracks the DXY.cash r closely, S4 needs NO new symbol feed —
every input is already in the worker's M5 universe (except USDSEK, 4.2% wt).

Checks: (1) correlation of r_rec vs r_dxy, (2) signal overlap at z20/1.5,
(3) full battery-style net for both on the same fills.
"""
import sys
import numpy as np
sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scripts/_absorb")

from costmaps_r3 import corrected_maps
from proxima_ops.backtest.feed import load_bars_cached
from proxima_ops.nova import feed as nfeed
from proxima_ops.nova import factors as F
from battery_dxy import trades, _pf, VOL, COMM, SYM, TICK, SPR

TICK, SPR = corrected_maps()


def lastc(s):
    b = nfeed.bars_list_to_arrays(load_bars_cached(s))
    d = F.bar_day(b["ts"])
    dd = np.unique(d)
    return dd, b["open"][np.searchsorted(d, dd, side="left")], b["close"][np.searchsorted(d, dd, side="right") - 1]


def residual_dxy():
    dd, od, dc = lastc("DXY.cash")
    de, oe, ec = lastc("EURUSD")
    n = min(len(dc), len(ec))
    r = np.log(dc[-n:]) + 0.576 * np.log(ec[-n:])
    return de[-n:], oe[-n:], ec[-n:], r - np.mean(r)


def residual_rec():
    """Reconstruct r from non-EUR components (day-intersected)."""
    syms = ["USDJPY", "GBPUSD", "USDCAD", "USDSEK", "USDCHF"]
    wts = {"USDJPY": 0.136, "GBPUSD": -0.119, "USDCAD": 0.091,
           "USDSEK": 0.042, "USDCHF": 0.036}
    closes = {}
    for s in syms:
        try:
            dd, od, dc = lastc(s)
            closes[s] = (dd, dc)
        except Exception as e:
            print(f"  [no {s}: {e}]")
    days = set.intersection(*[set(v[0]) for v in closes.values()])
    days = np.array(sorted(days))
    r = np.zeros(len(days))
    for s, (dd, dc) in closes.items():
        idx = np.searchsorted(dd, days)
        r += wts[s] * np.log(dc[idx])
    return days, r - np.mean(r)


def main():
    de, oe, ec, r1 = residual_dxy()
    days2, r2 = residual_rec()
    # intersect on days
    common = np.intersect1d(de[-len(r1):], days2)
    i1 = np.searchsorted(de[-len(r1):], common)
    i2 = np.searchsorted(days2, common)
    r1c, r2c = r1[i1], r2[i2]
    oe_c = oe[-len(r1):][i1]
    corr = np.corrcoef(r1c, r2c)[0, 1]
    print(f"days common={len(common)}  corr(r_dxy, r_rec)={corr:.4f}")

    # signal overlap at the certified z20/1.5
    z1 = F.rolling_z(r1c, 20)
    z2 = F.rolling_z(r2c, 20)
    # certified battery convention: z<-1.5 -> LONG EURUSD (+1), z>+1.5 -> SHORT (-1)
    s1 = np.where(z1 < -1.5, 1.0, np.where(z1 > 1.5, -1.0, 0.0))
    s2 = np.where(z2 < -1.5, 1.0, np.where(z2 > 1.5, -1.0, 0.0))
    n1, n2 = int(np.sum(s1 != 0)), int(np.sum(s2 != 0))
    agree = int(np.sum((s1 != 0) & (s1 == s2)))
    print(f"signals: dxy={n1} rec={n2}  same-day-same-dir={agree}  "
          f"overlap={agree / max(1, n1) * 100:.1f}% of dxy")

    # battery-style nets on the SAME day axis (fills next-day open, +1d exit)
    net1 = trades(s1, oe_c, len(oe_c))
    net2 = trades(s2, oe_c, len(oe_c))
    for tag, nn in (("DXY.cash ", net1), ("reconstr ", net2)):
        if nn:
            w = sum(1 for x in nn if x > 0) / len(nn) * 100
            print(f"  {tag}: n={len(nn):4d} win={w:4.1f}% net=${sum(nn):7.0f} "
                  f"exp/lot=${sum(nn) / len(nn) / VOL:6.1f} PF={_pf(nn):5.2f}")

    # what does the reconstructed signal need live? list inputs
    print("live inputs needed: EURUSD (fill sym) + USDJPY GBPUSD USDCAD USDSEK USDCHF "
          "(closes; all in worker M5 universe except USDSEK)")
    print("=> S4 fire path: daily close (last M5 bar of day), r via weights, z20,")
    print("   |z|>1.5 -> EURUSD at next day open, exit following day open.")


if __name__ == "__main__":
    main()
