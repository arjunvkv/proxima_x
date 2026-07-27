"""Find combination of features that predicts positive PnL in forward period."""
import re
import numpy as np
from itertools import combinations

path = "C:\\Users\\Arjun Sasi\\AppData\\Roaming\\MetaQuotes\\Tester\\D0E8209F77C8CF37AD8BF550E51FF075\\Agent-127.0.0.1-3000\\logs\\20260726.log"
with open(path, "rb") as f:
    raw = f.read()
text = raw.decode("utf-16-le", errors="replace")
lines = text.split("\n")

# Use the last run only
runs = []
current = []
for l in lines:
    if "final balance" in l:
        current.append(l)
        runs.append(current)
        current = []
    else:
        current.append(l)

run = runs[3]  # Last run (index 3)

# Parse trade data
deals = []
for l in run:
    m = re.search(r"deal #(\d+) (buy|sell) [\d.]+ (\w+) at ([\d.]+) done", l)
    if m:
        deals.append({"tkt": int(m.group(1)), "side": m.group(2), "sym": m.group(3), "pr": float(m.group(4))})

pnls = []
i = 0
while i < len(deals) - 1:
    for j in range(i+1, len(deals)):
        if deals[j]["side"] != deals[i]["side"] and deals[j]["sym"] == deals[i]["sym"]:
            if deals[i]["side"] == "buy":
                p = (deals[j]["pr"] - deals[i]["pr"]) * 0.75 * 100000
            else:
                p = (deals[i]["pr"] - deals[j]["pr"]) * 0.75 * 100000
            pnls.append(p)
            i = j + 1
            break
    else:
        i += 1

# Parse entry bar features
features_list = []
for l in run:
    m = re.search(r"z=([-\d.eE+]+) open=([\d.]+) high=([\d.]+) low=([\d.]+) close=([\d.]+) tv=([\d.]+) sp=([\d.]+)", l)
    if m:
        z = float(m.group(1))
        o = float(m.group(2))
        h = float(m.group(3))
        lv = float(m.group(4))
        c = float(m.group(5))
        tv = float(m.group(6))
        sp = float(m.group(7))
        rng = h - lv
        body = abs(c - o)
        sm = body / rng if rng > 1e-10 else 1.0
        features_list.append({
            "abs_z": abs(z),
            "smoothness": sm,
            "body_pips": body * 10000,
            "range_pips": rng * 10000,
            "spread_pips": sp / 10.0,
            "tick_vol": tv,
        })

n = min(len(pnls), len(features_list))
p = np.array(pnls[:n])
feats = features_list[:n]

print(f"Total trades: {n}, PnL: ${p.sum():+.2f}, WR: {np.mean(p>0)*100:.0f}%\n")

# Define candidate binary filters
candidates = {}
for fi in ["abs_z", "smoothness", "body_pips", "range_pips", "spread_pips", "tick_vol"]:
    vals = np.array([f[fi] for f in feats])
    for pct in [20, 30, 40, 50, 60, 70, 80]:
        th = np.percentile(vals, pct)
        # Greater than threshold
        candidates[f"{fi}_>{pct}p"] = vals > th
        # Less than threshold
        candidates[f"{fi}_<{pct}p"] = vals < th

# Also add fixed meaningful thresholds
for fi, th in [("abs_z", 4.0), ("abs_z", 5.0), ("abs_z", 6.0), 
               ("smoothness", 0.85), ("smoothness", 0.90), ("smoothness", 0.95),
               ("body_pips", 2.0), ("body_pips", 3.0), ("body_pips", 4.0), ("body_pips", 5.0),
               ("range_pips", 3.0), ("range_pips", 4.0), ("range_pips", 5.0)]:
    vals = np.array([f[fi] for f in feats])
    candidates[f"{fi}>={th}"] = vals >= th
    candidates[f"{fi}<{th}"] = vals < th

print(f"Total binary candidates: {len(candidates)}")
print("\nTop single filters (highest mean PnL):")
results = []
for name, mask in candidates.items():
    if mask.sum() < 3 or (~mask).sum() < 3:
        continue
    r = {"name": name, "n": mask.sum(), "pnl": p[mask].sum(), "mean": p[mask].mean(), "wr": np.mean(p[mask] > 0)}
    results.append(r)
results.sort(key=lambda x: x["mean"], reverse=True)
for r in results[:15]:
    print(f"  {r['name']:25s}: N={r['n']:2d}  ${r['pnl']:+8.2f}  ${r['mean']:+7.2f}/trade  WR={r['wr']*100:.0f}%")

print("\nTop 2-filter combinations:")
best_combo_results = []
keys = list(candidates.keys())
n_keys = len(keys)
for i in range(n_keys):
    for j in range(i+1, n_keys):
        m = candidates[keys[i]] & candidates[keys[j]]
        if m.sum() < 3 or (~m).sum() < 3:
            continue
        mean = p[m].mean()
        if mean > 0:
            best_combo_results.append({"name": f"{keys[i]} & {keys[j]}", "n": m.sum(), "pnl": p[m].sum(), "mean": mean, "wr": np.mean(p[m] > 0)})

best_combo_results.sort(key=lambda x: x["mean"], reverse=True)
for r in best_combo_results[:10]:
    print(f"  {r['name']:45s}: N={r['n']:2d}  ${r['pnl']:+8.2f}  ${r['mean']:+7.2f}/trade  WR={r['wr']*100:.0f}%")

# Also check the reverse: combinations that filter BAD trades
print("\n\nBest 2-filter for PROFITABLE subset (mean > +$5):")
pos_combos = [r for r in best_combo_results if r["mean"] > 5.0]
for r in pos_combos[:10]:
    print(f"  {r['name']:45s}: N={r['n']:2d}  ${r['pnl']:+8.2f}  ${r['mean']:+7.2f}/trade  WR={r['wr']*100:.0f}%")
