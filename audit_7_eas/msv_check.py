"""Focused re-verification of MSV_Asian with its FULL 18-pair universe, aligned
to the EA (BASE_LOT=0.18, $7/lot RT commission, hours 0-6, 12-bar ret +/-0.0002,
SL/TP 0.35/0.45 JPY else 0.0035/0.0045, hold 12). Compares real vs shuffled
PER-TRADE expectation (robust purple diagnostic) + per-pair support check."""
from __future__ import annotations
import os, sys, random, statistics as st
ROOT = r"C:\Trading\Proxima_X"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "audit_7_eas"))
from run_audit import STRATEGIES, load_bars, trade_to_usd, split_by_ts, metrics, gate
import ea_ports as EP

MSV_LOT = 0.18
pairs = STRATEGIES["msv_asian"]["pairs"]


def ev_per_trade(tl, lot):
    usd = [trade_to_usd(t, lot) for t in tl if t is not None]
    n = len(usd)
    mean = sum(t["net"] for t in usd) / n if n else 0.0
    win = sum(1 for t in usd if t["net"] > 0)
    return {"n": n, "mean_usd": mean, "wr": win / n if n else 0.0,
            "net": round(sum(t["net"] for t in usd), 2)}


bmap = {p: load_bars(p) for p in pairs}
pf = getattr(EP, STRATEGIES["msv_asian"]["port"])

support = all(bool(b) for b in bmap.values()) and len(bmap) == 18
print(f"MSV 18-pair support: ALL PRESENT -> pair_support={support}")

real = pf(bmap)
r = ev_per_trade(real, MSV_LOT)
print("\n--- REAL tape ---")
print(f"  trades={r['n']}  win%={r['wr']*100:.2f}  ${r['mean_usd']:.2f}/trade  "
      f"net=${r['net']}")

# robust purple: 20 shuffled tapes, compare PER-TRADE mean distribution
rng = random.Random(42)
shuf_means, shuf_ns = [], []
for _ in range(20):
    sh = {}
    for sy, bars in bmap.items():
        sh[sy] = bars[:]
        rng.shuffle(sh[sy])
    s_ev = ev_per_trade(pf(sh), MSV_LOT)
    shuf_means.append(s_ev["mean_usd"]); shuf_ns.append(s_ev["n"])
sm = sum(shuf_means) / len(shuf_means)
sd = st.stdev(shuf_means)
real_mean = r["mean_usd"]
print("\n--- PURPLE (20 shuffled tapes) ---")
print(f"  shuffled trades/run: mean={sum(shuf_ns)/len(shuf_ns):.0f}")
print(f"  shuffled per-trade mean: {sm:.4f} +- {sd:.4f}  (real={real_mean:.4f})")
verdict = ("PRESENT (real >> shuffled)" if real_mean > sm + 2 * sd
           else "NOT ROBUST (real ~= shuffled or shuffled >= real)")
print(f"  verdict: real edge {verdict}")

# walkforward gate at real lot
frames = split_by_ts([trade_to_usd(t, MSV_LOT) for t in real if t is not None])
for w, m in (("train", frames[0]), ("val", frames[1])):
    if not m:
        print(f"  {w}: no trades"); continue
    met = metrics(m)
    g = gate(met, lot=MSV_LOT)
    print(f"  {w}: n={met['trades']} wr={met['win_rate']} PF={met['profit_factor']} "
          f"exp=${met['expectancy']} -> "
          f"{'PASS' if g['passed'] else 'REJECT ' + str(g['reject'])}")