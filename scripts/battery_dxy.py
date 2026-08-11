"""battery_dxy.py — deep battery for the DXY-implied divergence factor (#21).

Signal: daily residual r = log(DXY) + 0.576*log(EURUSD) - mean (DXY = 50.14 *
EURUSD^-0.576 * ...; EURUSD-implied DXY removed). When the non-EUR basket is
stretched vs its 20d window (|z| > thr), EURUSD mean-reverts over the next day.

Checks: LODO, monthly, plateau grid (thr x window), regime halves, cost stress
ladder, MC reshuffle (DD distribution + FTMO $5k breach prob), FTMO sim,
entry-timing sensitivity (next-day open vs +1 bar).
"""
import sys

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "scripts/_absorb")

from costmaps_r3 import corrected_maps
from proxima_ops.backtest.feed import load_bars_cached
from proxima_ops.backtest.pnl import trade_to_usd
from proxima_ops.nova import feed as nfeed
from proxima_ops.nova import factors as F

TICK, SPR = corrected_maps()
VOL = 0.15
COMM = 3.0
SYM = "EURUSD"
LIMIT = 5000.0
MAXDD = 10000.0


def lastc(s):
    b = nfeed.bars_list_to_arrays(load_bars_cached(s))
    d = F.bar_day(b["ts"])
    dd = np.unique(d)
    return dd, b["open"][np.searchsorted(d, dd, side="left")], b["close"][np.searchsorted(d, dd, side="right") - 1]


def residual():
    dd, od, dc = lastc("DXY.cash")
    de, oe, ec = lastc("EURUSD")
    n = min(len(dc), len(ec))
    r = np.log(dc[-n:]) + 0.576 * np.log(ec[-n:])
    r = r - np.mean(r)
    return de[-n:], oe[-n:], ec[-n:], r


def trades(sig, oe, n, entry_shift=0):
    """sig[i]!=0 -> fill at open of day i+1+shift, exit open of day i+2+shift."""
    i = np.where(sig[:-2 - entry_shift] != 0)[0]
    if len(i) == 0:
        return []
    dirn = sig[i]
    entry = oe[-(n):][i + 1 + entry_shift]
    ex = oe[-(n):][i + 2 + entry_shift]
    pts = (ex - entry) * dirn / TICK[SYM]
    return [trade_to_usd(dict(symbol=SYM, side="BUY" if dirn[k] > 0 else "SELL",
                              entry=float(entry[k]), entry_ts=int(1), exit_ts=int(2),
                              reason="HOLD", pnl_pts=float(pts[k])),
                         VOL, {SYM: TICK[SYM]}, COMM, {SYM: SPR[SYM]})["net"]
            for k in range(len(i))]


def stats(net, tag):
    if not net:
        print(f"{tag}: NO TRADES"); return None
    w = sum(1 for x in net if x > 0) / len(net) * 100
    print(f"{tag}: n={len(net):4d} win={w:4.1f}% net=${sum(net):7.0f} "
          f"exp/lot=${sum(net)/len(net)/VOL:6.1f} PF={_pf(net):5.2f}")
    return np.array(net)


def _pf(net):
    wins = [x for x in net if x > 0]; losses = [x for x in net if x <= 0]
    return sum(wins) / abs(sum(losses)) if losses else float("inf")


def ftmo_sim(net, tag):
    """FTMO $100k sim: size so worst 2x-stress day = $5,000 daily limit."""
    net = np.array(net)
    daily = np.array([net[np.array([i])].sum() for i in range(len(net))]) if False else None
    # daily grouping by position index is not available; use per-trade MC instead
    worst = np.abs(net).max()
    v = max(0.15, 0.15 * LIMIT / (worst * 2.0))
    scaled = net * v / VOL
    dd = 0.0; peak = 0.0; maxdd = 0.0; streak = 0; worst_streak = 0
    cum = 0.0
    for x in scaled:
        cum += x; peak = max(peak, cum); maxdd = max(maxdd, peak - cum)
        streak = streak + 1 if x < 0 else 0
        worst_streak = max(worst_streak, streak)
    print(f"{tag}: size={v:.2f} lots  worst1x=${worst*v/VOL:7.0f}  maxDD=${maxdd:7.0f} "
          f"(limit ${MAXDD:.0f})  lossStreak={worst_streak}d  net=${cum:7.0f} (~{cum/10000*100:.1f}%/mo @7mo)")
    return v


def main():
    de, oe, ec, r = residual()
    n = len(r)
    z = F.rolling_z(r, 20)
    sig = np.where(z < -1.5, 1.0, np.where(z > 1.5, -1.0, 0.0))
    net = trades(sig, oe, n)
    stats(net, "BASE thr1.5 w20")
    base = np.array(net)

    # LODO: drop each day's signal, recompute full net -> all positive?
    lodo_ok = 0
    for drop in np.where(sig != 0)[0]:
        sig2 = sig.copy(); sig2[drop] = 0.0
        n2 = trades(sig2, oe, n)
        if n2 and sum(n2) > 0:
            lodo_ok += 1
    ntr = len(np.where(sig != 0)[0])
    print(f"LODO: {lodo_ok}/{ntr} drops keep net>0")

    # monthly via signal days (day numbers -> 30-day buckets)
    import collections
    sig_days = np.where(sig != 0)[0]
    mon = collections.defaultdict(list)
    for k in sig_days:
        mon[int(de[k] // 30)].append(k)
    print("monthly (net$):", {m: round(sum(trades(np.where(np.isin(np.arange(n), idx), sig, 0.0), oe, n)), 0) for m, idx in sorted(mon.items())})

    # plateau grid: window x threshold
    print("PLATEAU (thr x window):")
    cells = 0
    for w in (10, 20, 40):
        for thr in (1.0, 1.5, 2.0):
            zw = F.rolling_z(r, w)
            sg = np.where(zw < -thr, 1.0, np.where(zw > thr, -1.0, 0.0))
            nn = trades(sg, oe, n)
            tot = sum(nn) if nn else 0.0
            ok = "PASS" if tot > 0 else "die"
            cells += 1
            print(f"  w{w}/thr{thr}: n={len(nn):3d} net=${tot:7.0f} {ok}")
    print(f"  cells: {cells}")

    # regime halves
    half = n // 2
    for name, sl in (("H1", slice(0, half)), ("H2", slice(half, n))):
        sig2 = sig.copy(); sig2[sl.stop:] = 0.0 if sl.stop < n else sig2[sl.stop:]
        # keep only signals in slice
        sig3 = np.zeros(n); sig3[sl] = sig[sl]
        nn = trades(sig3, oe, n)
        stats(nn, f"REGIME {name}")

    # cost stress ladder
    for x in (1.25, 1.5, 2.0):
        spr = SPR[SYM] * x
        i = np.where(sig[:-2] != 0)[0]
        dirn = sig[i]; entry = oe[-(n):][i + 1]; ex = oe[-(n):][i + 2]
        pts = (ex - entry) * dirn / TICK[SYM]
        nn = [trade_to_usd(dict(symbol=SYM, side="BUY" if dirn[k] > 0 else "SELL",
                                entry=float(entry[k]), entry_ts=1, exit_ts=2, reason="HOLD",
                                pnl_pts=float(pts[k])), VOL, {SYM: TICK[SYM]}, COMM, {SYM: spr})["net"]
              for k in range(len(i))]
        print(f"stress {x}x spread: net=${sum(nn):7.0f} exp/lot=${sum(nn)/len(nn)/VOL:6.1f}")

    # MC reshuffle: embed nets into the 200-day series, shuffle DAYS 1000x,
    # scale to sim size -> DD distribution + $10k maxDD breach prob
    v = max(0.15, 0.15 * LIMIT / (np.abs(base).max() * 2.0))
    daily = np.zeros(n)
    daily[np.where(sig[:-2] != 0)[0]] = base * v / VOL
    rng = np.random.default_rng(7)
    dds = []
    for _ in range(1000):
        sh = rng.permutation(daily)
        cum = np.cumsum(sh)
        dd = np.max(np.maximum.accumulate(cum) - cum)
        dds.append(dd)
    dds = np.array(dds)
    print(f"MC1000 (daily shuffle @ {v:.2f} lots): mean maxDD=${dds.mean():6.0f} "
          f"p95=${np.percentile(dds, 95):6.0f} breach>{MAXDD:.0f}: {(dds > MAXDD).mean()*100:.1f}%")

    # FTMO sim (size chosen so worst 2x-stress day = $5,000 daily limit)
    worst = np.abs(base).max()
    v = max(0.15, 0.15 * LIMIT / (worst * 2.0))
    scaled = base * v / VOL
    cum = np.cumsum(scaled)
    maxdd = np.max(np.maximum.accumulate(cum) - cum)
    streak = 0; worst_streak = 0
    for x in scaled:
        streak = streak + 1 if x < 0 else 0
        worst_streak = max(worst_streak, streak)
    nmonths = max(1, n / 21)
    print(f"FTMO thr1.5: size={v:.2f} lots  worst1x=${worst*v/VOL:7.0f} (daily limit ${LIMIT:.0f}, 2x=${LIMIT:.0f})  "
          f"maxDD=${maxdd:7.0f} (limit ${MAXDD:.0f})  lossStreak={worst_streak}d  "
          f"net=${cum[-1]:7.0f} ({cum[-1]/10000*100:.1f}% total / {cum[-1]/10000/nmonths*100:.1f}%/mo @{nmonths:.1f}mo)")

    # entry-timing sensitivity: +1 bar (shift fill by one more day)
    nn = trades(sig, oe, n, entry_shift=1)
    stats(nn, "ENTRY +1 day")


if __name__ == "__main__":
    main()
