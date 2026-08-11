"""gold_fix_study.py — LBMA PM-fix mechanism on XAUUSD/XAGUSD (server 17:00 = 15:00 UTC).

The FX fix-window family was dead (audited 20/20), but FX fixes are indicative
auctions; gold has a REAL auction (LBMA). Pre-registered:
  T0 = pre-fix drift [16:40,16:55] server
  T1 = fix window     [16:55,17:10]
  T2 = post-fix       [17:10,17:40]
  H_a: T0 direction continues into T2 (fix confirms drift)
  H_b: T0 direction reverses into T2 (fix absorbs positioning)
  H_c: T1 direction continues into T2 (auction print carries)
Per-side pips (bid-side longs / ask-side shorts), LODO, honest null (random
sign on the SIGNAL), cost check (measured spreads). Research-only.
"""
import sys, os
import numpy as np
import polars as pl
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MARKET = "audit_7_eas/market"
SPREAD = {"XAUUSD": 0.45, "XAGUSD": 0.059}   # price units (measured live)

def load(sym):
    return pl.read_parquet(f"{MARKET}/{sym}.pqt").sort("time")

def study(sym, w0=(40, 55), w1=(55, 10), w2=(10, 40), fwd=(40, 100)):
    s = load(sym)
    ts = s["time"].to_numpy(); cl = s["close"].to_numpy()
    n = len(ts)
    def minute_of(tt):
        return (tt // 60) % 60
    def hour_of(tt):
        return (tt // 3600) % 24
    # signal bar: the bar whose minute == end of T1 (17:10) — T0/T1 fully closed
    events = []
    for i in range(2, n - 1):
        if hour_of(ts[i]) == 17 and minute_of(ts[i]) == w1[1]:
            events.append(i)
    if len(events) < 60:
        print(f"{sym}: only {len(events)} fix events"); return None
    t0 = np.array([cl[i - 15] / cl[i - 3] - 1.0 for i in events])   # 16:40-16:55
    t1 = np.array([cl[i] / cl[i - 15] - 1.0 for i in events])       # 16:55-17:10
    t2 = np.array([cl[i + 6] / cl[i] - 1.0 for i in events])        # 17:10-17:40
    fw = np.array([cl[i + 12] / cl[i] - 1.0 for i in events])       # 17:10-18:10
    days = np.array([int(ts[i]) // 86400 for i in events])
    sp = SPREAD[sym]
    out = {}
    for name, sig, tgt, H in [("a_T0->T2", t0, t2, "17:10-17:40"),
                              ("a_T0->fwd", t0, fw, "17:10-18:10"),
                              ("c_T1->T2", t1, t2, "17:10-17:40"),
                              ("c_T1->fwd", t1, fw, "17:10-18:10")]:
        # per-side: long side = sig>0 trades (fwd continuation), short side = sig<0
        mask = np.abs(sig) > 1e-12
        sgn = np.sign(sig[mask]); f = tgt[mask]; d = days[mask]
        longs = f[sgn > 0]; shorts = f[sgn < 0]
        net_all = (sgn * f)
        gross = net_all.mean()
        net = gross - 2 * sp
        wins = net_all[net_all > 0].sum(); losses = -net_all[net_all < 0].sum()
        pf = wins / losses if losses > 0 else 99
        z = net_all.mean() / (net_all.std() / np.sqrt(len(net_all)))
        lodo = sum(1 for dd in set(d) if (net_all[d != dd]).mean() <= 0)
        rng = np.random.default_rng(7)
        null = np.mean([(rng.choice([-1.0, 1.0], size=len(net_all)) * net_all).mean()
                        for _ in range(300)])
        out[name] = (gross, net, pf, z, lodo, len(net_all), sgn.size,
                     longs.mean(), shorts.mean())
        print(f"{sym} {name} (fwd {H}): gross={gross*1e4:+.1f}e-4 net={net*1e4:+.1f}e-4 "
              f"PF={pf:.2f} z={z:+.2f} LODO={lodo}/{len(set(d))} n={len(net_all)} "
              f"long={longs.mean()*1e4:+.1f} short={shorts.mean()*1e4:+.1f} null={null*1e4:+.1f}")
    return out

print("=== LBMA PM fix (server 17:00) ===")
study("XAUUSD")
study("XAGUSD")
