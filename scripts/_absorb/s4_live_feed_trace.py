"""s4_live_feed_trace.py — causal-replay gate for S4 (R3-3b equivalent).

Proves the LIVE-computable signal reproduces the certified battery trades:
  * cached : daily closes from the audit cache (battery-identical baseline)
  * terminal: D1 bars fetched from the live FTMO terminal (what the worker
             will actually see at fire time)

Both compute r = log(DXY.close) + 0.576*log(EURUSD.close), z = trailing-20
z-score (no full-sample mean — it cancels in the z, verified), signal = |z|>1.5,
fill at NEXT day open, exit at day+2 open (battery semantics).

Prints trade days for cross-source comparison + full stats for each source.
"""
import argparse
import sys
import json

import numpy as np


def rolling_z(x, w=20):
    """NOVA-exact causal z: strictly-prior window ending at bar i-1."""
    out = np.full(len(x), np.nan)
    for i in range(1, len(x)):
        seg = x[max(0, i - w): i]
        if len(seg) >= 2:
            sd = seg.std()
            out[i] = (x[i] - seg.mean()) / sd if sd > 0 else np.nan
    return out


def pf(net):
    w = [x for x in net if x > 0]
    l = [x for x in net if x <= 0]
    return sum(w) / abs(sum(l)) if l else float("inf")


def daily_closes_cached():
    sys.path.insert(0, ".")
    sys.path.insert(0, "scripts")
    sys.path.insert(0, "scripts/_absorb")
    from proxima_ops.backtest.feed import load_bars_cached
    from proxima_ops.nova import feed as nfeed
    from proxima_ops.nova import factors as F

    def lastc(s):
        b = nfeed.bars_list_to_arrays(load_bars_cached(s))
        d = F.bar_day(b["ts"])
        dd = np.unique(d)
        return dd, b["open"][np.searchsorted(d, dd, side="left")], \
            b["close"][np.searchsorted(d, dd, side="right") - 1]

    ddx, odx, dcx = lastc("DXY.cash")
    dde, ode, ece = lastc("EURUSD")
    n = min(len(dcx), len(ece))
    return ddx[-n:], odx[-n:], dcx[-n:], ode[-n:], ece[-n:]


def daily_closes_terminal():
    import os
    import MetaTrader5 as mt5
    if not mt5.initialize(path=os.environ["MT5_PATH"], timeout=4000):
        print("MT5 init FAILED"); sys.exit(1)
    for s in ("DXY.cash", "EURUSD"):
        mt5.symbol_select(s, True)

    def m5_to_daily(sym, n=15000):
        """Worker-exact daily closes: last M5 close of each server day."""
        r = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_M5, 0, n)
        if r is None:
            print(f"no M5 for {sym}: {mt5.last_error()}"); sys.exit(1)
        days, opens, closes, last_ts = [], [], [], None
        for x in r:
            d = int(x["time"]) // 86400
            if last_ts is not None and d != last_ts:
                days.append(last_ts); opens.append(o); closes.append(c)
            last_ts, o, c = d, float(x["open"]), float(x["close"])
        days.append(last_ts); opens.append(o); closes.append(c)
        return np.array(days), np.array(opens), np.array(closes)

    ddx, odx, dcx = m5_to_daily("DXY.cash")
    dde, ode, ece = m5_to_daily("EURUSD")
    print(f"[dxy daily] n={len(ddx)} range={ddx[0]}..{ddx[-1]}", file=sys.stderr)
    print(f"[eur daily] n={len(dde)} range={dde[0]}..{dde[-1]}", file=sys.stderr)
    common = np.intersect1d(ddx, dde)
    i1 = np.searchsorted(ddx, common); i2 = np.searchsorted(dde, common)
    print(f"[align] common={len(common)} i1max={i1.max()} i2max={i2.max()} "
          f"ddx_sorted={bool(np.all(np.diff(ddx) > 0))} "
          f"dde_sorted={bool(np.all(np.diff(dde) > 0))}", file=sys.stderr)
    # fire-hour check: which hour does DXY.cash's LAST M5 bar of the day carry?
    r0 = mt5.copy_rates_from_pos("DXY.cash", mt5.TIMEFRAME_M5, 0, 5)
    print(f"[dxy last M5 ts] {[int(x['time']) for x in r0]} "
          f"(hour {(int(r0[-1]['time']) // 3600) % 24}, "
          f"server-now {mt5.symbol_info_tick('EURUSD').time})", file=sys.stderr)
    # per-day last-bar hour for DXY.cash (fire-hour envelope)
    rq = mt5.copy_rates_from_pos("DXY.cash", mt5.TIMEFRAME_M5, 0, 15000)
    hh = {}
    for x in rq:
        d = int(x["time"]) // 86400
        hh[d] = (int(x["time"]) // 3600) % 24
    hours = sorted(set(hh.values()))
    print(f"[dxy last-bar hour per day] seen={hours} "
          f"last3={list(hh.items())[-3:]}", file=sys.stderr)
    mt5.shutdown()
    return common, odx[i1], dcx[i1], ode[i2], ece[i2]


def run(days, odx, dcx, ode, ece, tag, want_trades=False):
    r = np.log(dcx) + 0.576 * np.log(ece)
    # full-sample mean subtraction is a constant shift -> cancels in z.
    z = rolling_z(r, 20)
    sig = np.where(z < -1.5, 1.0, np.where(z > 1.5, -1.0, 0.0))
    idx = np.where(sig[:-2] != 0)[0]
    nets, tdays, dirs = [], [], []
    for k in idx:
        d = sig[k]
        entry, ex = ode[k + 1], ode[k + 2]
        pts = (ex - entry) * d
        nets.append(pts)
        tdays.append(int(days[k + 1]))
        dirs.append("L" if d > 0 else "S")
    nets = np.array(nets) if nets else np.array([0.0])
    w = sum(1 for x in nets if x > 0) / len(nets) * 100
    print(f"[{tag}] n={len(nets)} win={w:.1f}% gross_pts_sum={nets.sum():.2f} "
          f"avg_pts={nets.mean():.3f} PF={pf(nets):.2f}")
    print(f"[{tag}] signal days: {tdays}")
    print(f"[{tag}] dirs: {''.join(dirs)}")
    if want_trades:
        with open(f"s4_{tag}_trades.json", "w") as f:
            json.dump({"tag": tag, "days": tdays, "dirs": dirs,
                       "nets_pts": [round(float(x), 4) for x in nets]}, f)
    return tdays, dirs


def compare(cached_json="s4_cached_closes.json"):
    """Overlap test: cached-vs-terminal DXY.cash closes, residual corr, signal agreement."""
    with open(cached_json) as f:
        J = json.load(f)
    d1 = np.array(J["days"]); c1 = np.array(J["dcx"]); f1 = np.array(J["ece"])
    d2, o2, c2, e2, f2 = daily_closes_terminal()
    common = np.intersect1d(d1, d2)
    i1 = np.searchsorted(d1, common); i2 = np.searchsorted(d2, common)
    if len(common) == 0:
        print("NO OVERLAP"); return
    rc = np.log(c1[i1]) + 0.576 * np.log(f1[i1])
    rt = np.log(c2[i2]) + 0.576 * np.log(f2[i2])
    print(f"[compare] overlap n={len(common)} days {common[0]}..{common[-1]}")
    print(f"[compare] corr(dxy close cached, terminal) = {np.corrcoef(c1[i1], c2[i2])[0, 1]:.4f}")
    print(f"[compare] corr(residual cached, terminal)  = {np.corrcoef(rc, rt)[0, 1]:.4f}")
    zc = rolling_z(rc, 20); zt = rolling_z(rt, 20)
    sc = (zc < -1.5) | (zc > 1.5); st = (zt < -1.5) | (zt > 1.5)
    full = np.arange(len(common)) >= 21  # full 20-day prior window both sides
    agree = np.mean(sc[full] == st[full])
    print(f"[compare] full-window days={int(full.sum())} signal-agreement={100 * agree:.1f}%")
    print(f"[compare] cached fires in overlap: {list(common[sc & full])}")
    print(f"[compare] term   fires in overlap: {list(common[st & full])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["cached", "terminal", "compare"], required=True)
    ap.add_argument("--dump", action="store_true", help="write cached closes to s4_cached_closes.json")
    ap.add_argument("--cached-json", default="s4_cached_closes.json", help="cached closes JSON for compare")
    args = ap.parse_args()
    if args.source == "cached":
        days, odx, dcx, ode, ece = daily_closes_cached()
        if args.dump:
            with open(args.cached_json, "w") as f:
                json.dump({"days": [int(x) for x in days], "odx": [float(x) for x in odx],
                           "dcx": [float(x) for x in dcx], "ode": [float(x) for x in ode],
                           "ece": [float(x) for x in ece]}, f)
            print(f"[cached] dumped {len(days)} days to {args.cached_json}")
        run(days, odx, dcx, ode, ece, "cached", want_trades=True)
    elif args.source == "terminal":
        days, odx, dcx, ode, ece = daily_closes_terminal()
        run(days, odx, dcx, ode, ece, "terminal", want_trades=True)
    else:
        compare(args.cached_json)


if __name__ == "__main__":
    main()
