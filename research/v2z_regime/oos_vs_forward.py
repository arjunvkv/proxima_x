"""Compare feature predictive power across OOS vs Forward periods."""
import re
import numpy as np

path = "C:\\Users\\Arjun Sasi\\AppData\\Roaming\\MetaQuotes\\Tester\\D0E8209F77C8CF37AD8BF550E51FF075\\Agent-127.0.0.1-3000\\logs\\20260726.log"
with open(path, "rb") as f:
    raw = f.read()
text = raw.decode("utf-16-le", errors="replace")
lines = text.split("\n")

# Split into runs
runs = []
current = []
for l in lines:
    if "final balance" in l:
        current.append(l)
        runs.append(current)
        current = []
    else:
        current.append(l)
if current:
    runs.append(current)

print(f"Runs: {len(runs)}")

def parse_run(run):
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
    
    feats = []
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
            feats.append({
                "abs_z": abs(z),
                "smoothness": sm,
                "body_pips": body * 10000,
                "range_pips": rng * 10000,
                "spread_pips": sp / 10.0,
                "tick_vol": tv,
            })
    
    n = min(len(pnls), len(feats))
    return np.array(pnls[:n]), feats[:n]

# Forward = run 3 (last forward run, with real ENTRYBAR data), OOS = run 4
for name, run_idx in [("OOS (Feb-Mar)", 4), ("Forward (Jun-Jul)", 3)]:
    p, feats = parse_run(runs[run_idx])
    n = len(p)
    
    print(f"\n{'='*70}")
    print(f"  {name}: {n} trades  total=${p.sum():+.2f}  WR={np.mean(p>0)*100:.0f}%")
    print(f"{'='*70}")
    
    # Feature means by win/loss
    print(f"\n  {'Feature':<20s} {'Winners':<12s} {'Losers':<12s} {'t-stat':<8s} {'Direction'}")
    print(f"  {'-'*60}")
    for col in ["abs_z", "smoothness", "body_pips", "range_pips", "spread_pips", "tick_vol"]:
        vals = np.array([f[col] for f in feats])
        w = vals[p > 0]
        l_vals = vals[p <= 0]
        if len(w) < 3 or len(l_vals) < 3:
            continue
        wm, lm = w.mean(), l_vals.mean()
        t = (wm - lm) / np.sqrt(w.var()/len(w) + l_vals.var()/len(l_vals))
        arrow = "WIN>LOSE  " if wm > lm else "LOSE>WIN  "
        print(f"  {col:<20s} {wm:>8.4f}    {lm:>8.4f}    {t:+7.2f}  {arrow}")
    
    # Key filters
    print(f"\n  {'Filter':<35s} {'N':<4s} {'PnL':<10s} {'$/trade':<8s} {'WR':<6s}")
    print(f"  {'-'*60}")
    
    smooth_vals = np.array([f["smoothness"] for f in feats])
    body_vals = np.array([f["body_pips"] for f in feats])
    tv_vals = np.array([f["tick_vol"] for f in feats])
    absz_vals = np.array([f["abs_z"] for f in feats])
    
    filters = [
        ("All trades", np.ones(n, dtype=bool)),
        ("smoothness >= 0.85", smooth_vals >= 0.85),
        ("smoothness >= 0.90", smooth_vals >= 0.90),
        ("smoothness >= 0.95", smooth_vals >= 0.95),
        ("body_pips >= 3.0", body_vals >= 3.0),
        ("body_pips >= 4.0", body_vals >= 4.0),
        ("tick_vol > median", tv_vals > np.median(tv_vals)),
        ("tick_vol > 75p", tv_vals > np.percentile(tv_vals, 75)),
        ("smooth>=0.90 & tv>median", (smooth_vals >= 0.90) & (tv_vals > np.median(tv_vals))),
        ("smooth>=0.85 & tv>75p", (smooth_vals >= 0.85) & (tv_vals > np.percentile(tv_vals, 75))),
        ("smooth>=0.90 & body>=3.0", (smooth_vals >= 0.90) & (body_vals >= 3.0)),
        ("smooth>=0.85 & body>=4.0", (smooth_vals >= 0.85) & (body_vals >= 4.0)),
        ("abs_z >= 4.0", absz_vals >= 4.0),
        ("abs_z >= 5.0", absz_vals >= 5.0),
    ]
    
    for fname, mask in filters:
        if mask.sum() < 2:
            continue
        print(f"  {fname:<35s} {mask.sum():<4d} ${p[mask].sum():+8.2f} ${p[mask].mean():+7.2f} {np.mean(p[mask]>0)*100:.0f}%")
    
    # Best 3-filter combo with at least 5 trades
    print(f"\n  Best combos (>=5 trades, profitable):")
    best = []
    for sm_th in [0.80, 0.85, 0.90, 0.95]:
        for bd_th in [2.0, 3.0, 4.0, 5.0]:
            for tv_pct in [50, 60, 70, 80]:
                tv_th = np.percentile(tv_vals, tv_pct)
                mask = (smooth_vals >= sm_th) & (body_vals >= bd_th) & (tv_vals > tv_th)
                if mask.sum() >= 5 and p[mask].mean() > 0:
                    best.append((p[mask].mean(), mask.sum(), sm_th, bd_th, tv_pct, p[mask].sum()))
    best.sort(reverse=True)
    for mean, cnt, sm_th, bd_th, tv_pct, tot in best[:5]:
        print(f"    sm>={sm_th:.2f} body>={bd_th:.1f}p tv>{tv_pct}p: N={cnt:3d} ${tot:+8.2f} ${mean:+7.2f}/trade")

print("\n\n=== Key insight: Do features work SAME or OPPOSITE direction in OOS vs Forward? ===")
for col in ["abs_z", "smoothness", "body_pips", "range_pips", "spread_pips", "tick_vol"]:
    for name, run_idx in [("OOS", 4), ("Forward", 3)]:
        p, feats = parse_run(runs[run_idx])
        vals = np.array([f[col] for f in feats])
        w_mean = vals[p > 0].mean()
        l_mean = vals[p <= 0].mean()
        if name == "OOS":
            oos_dir = "WIN>LOSE" if w_mean > l_mean else "LOSE>WIN"
            oos_d = w_mean - l_mean
        else:
            fwd_dir = "WIN>LOSE" if w_mean > l_mean else "LOSE>WIN"
            fwd_d = w_mean - l_mean
    agreement = "SAME" if (oos_d > 0) == (fwd_d > 0) else "OPPOSITE"
    print(f"  {col:<20s}: OOS={oos_dir:<12s} Forward={fwd_dir:<12s} -> {agreement}")
