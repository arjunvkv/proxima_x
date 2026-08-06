"""Aligned economics for the remaining strategies (same lens as MSV):
full universe, declared lot, $7/lot RT commission, gross/comm/net per trade,
trade density, and purple verdict."""
from __future__ import annotations
import os, sys, random, statistics as st
ROOT = r"C:\Trading\Proxima_X"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "audit_7_eas"))
from run_audit import STRATEGIES, load_bars, trade_to_usd, split_by_ts, metrics, gate
import ea_ports as EP

SKIP = {"msv_asian", "test_min_fire"}
DAYS = 200


for name, cfg in STRATEGIES.items():
    if name in SKIP:
        continue
    lot = cfg.get("lot", 0.15)
    bmap = {s: load_bars(s) for s in cfg["pairs"]}
    pf = getattr(EP, cfg["port"])
    trades = pf(bmap)
    usd = [trade_to_usd(t, lot) for t in trades if t is not None]
    usd1 = [trade_to_usd(t, 1.0) for t in trades if t is not None]
    n = len(usd)
    if not n:
        print(f"\n=== {name} ({len(cfg['pairs'])} pairs) ===  NO TRADES"); continue
    gross = sum(t["gross_usd"] for t in usd)
    comm = sum(t["commission"] for t in usd)
    net = sum(t["net"] for t in usd)
    wins = sum(1 for t in usd if t["net"] > 0)
    real_mean = sum(t["net"] for t in usd1) / n          # 1.0-lot basis
    per_day = n / DAYS
    print(f"\n=== {name} ({len(cfg['pairs'])} pairs, lot={lot}) ===")
    print(f"  trades={n} ({per_day:.1f}/day)  win%={wins/n*100:.1f}  "
          f"gross=${gross:,.0f}  comm=${comm:,.0f}  net=${net:,.0f} (${net/21:,.0f}/mo)")
    print(f"  per-trade: gross=${gross/n:.2f}  comm=${comm/n:.2f}  net=${net/n:.3f}  "
          f"comm-share={comm/gross*100:.0f}%")
    # purple (1.0 lot basis, compared to real 1.0-lot basis)
    rng = random.Random(42)
    means, ns = [], []
    for _ in range(10):
        sh = {}
        for sy, bars in bmap.items():
            sh[sy] = bars[:]
            rng.shuffle(sh[sy])
        tr = pf(sh)
        if not tr:
            continue
        u2 = [trade_to_usd(t, 1.0) for t in tr if t is not None]
        ns.append(len(u2))
        means.append(sum(t["net"] for t in u2) / len(u2))
    if means:
        sm = sum(means) / len(means)
        sd = st.stdev(means) if len(means) > 1 else 0.0
        tag = "REAL-EDGE" if real_mean > sm + 2 * sd else "NO-EDGE"
        print(f"  purple: shuffled={sm:+.2f}/tr (n={sum(ns)//len(ns)}) vs real={real_mean:+.2f}/tr -> {tag}")
    else:
        print(f"  purple: shuffle produced 0 trades")
    tr, va = split_by_ts(usd)
    for w, m in (("train", tr), ("val", va)):
        if not m:
            continue
        met = metrics(m)
        g = gate(met, lot=lot)
        print(f"  {w}: wr={met['win_rate']:.2f} PF={met['profit_factor']:.2f} "
              f"exp=${met['expectancy']:.2f} -> "
              f"{'PASS' if g['passed'] else 'REJECT ' + str(g['reject'][:2])}")