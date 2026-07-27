"""Per-run analysis from agent log - split by final balance."""
import re
import numpy as np

path = "C:\\Users\\Arjun Sasi\\AppData\\Roaming\\MetaQuotes\\Tester\\D0E8209F77C8CF37AD8BF550E51FF075\\Agent-127.0.0.1-3000\\logs\\20260726.log"
with open(path, "rb") as f:
    raw = f.read()
text = raw.decode("utf-16-le", errors="replace")
lines = text.split("\n")

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

print(f"Total runs: {len(runs)}\n")
for ri, run in enumerate(runs):
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
    
    abs_zs = []
    for l in run:
        m = re.search(r"z=([-\d.eE+]+)", l)
        if m:
            abs_zs.append(abs(float(m.group(1))))
    
    smooths = []
    for l in run:
        m = re.search(r"open=([\d.]+) high=([\d.]+) low=([\d.]+) close=([\d.]+)", l)
        if m:
            o, h, lv, c = float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))
            rng = h - lv
            body = abs(c - o)
            smooths.append(body / rng if rng > 1e-10 else 1.0)
    
    bodies = []
    for l in run:
        m = re.search(r"open=([\d.]+) .* close=([\d.]+)", l)
        if m:
            o, c = float(m.group(1)), float(m.group(2))
            bodies.append(abs(c - o))
    
    n = min(len(pnls), len(abs_zs), len(smooths))
    if n == 0:
        continue
    
    p, z, sm, bd = np.array(pnls[:n]), np.array(abs_zs[:n]), np.array(smooths[:n]), np.array(bodies[:n])
    
    bal = [l for l in run if "balance" in l]
    fb = bal[-1] if bal else "?"
    
    print(f"  Run {ri}: {n} trades  total=${p.sum():+.2f}  WR={np.mean(p>0)*100:.0f}%  {fb.strip()[:60]}")
    
    # z-score split
    for th in [4.0, 5.0, 6.0]:
        lo = z < th
        hi = z >= th
        if lo.sum() >= 3:
            print(f"    z<{th:.0f}: {lo.sum():3d} trades  ${p[lo].sum():+8.2f}  WR={np.mean(p[lo]>0)*100:.0f}%")
        if hi.sum() >= 3:
            print(f"    z>={th:.0f}: {hi.sum():3d} trades  ${p[hi].sum():+8.2f}  WR={np.mean(p[hi]>0)*100:.0f}%")
    
    # Smoothness
    for th in [0.85, 0.9, 0.95]:
        sm_hi = sm > th
        sm_lo = sm <= th
        if sm_hi.sum() >= 3:
            print(f"    sm>{th:.2f}: {sm_hi.sum():3d} trades  ${p[sm_hi].sum():+8.2f}  WR={np.mean(p[sm_hi]>0)*100:.0f}%")
        if sm_lo.sum() >= 3:
            print(f"    sm<={th:.2f}: {sm_lo.sum():3d} trades  ${p[sm_lo].sum():+8.2f}  WR={np.mean(p[sm_lo]>0)*100:.0f}%")
    
    # Body size (pips)
    bd_pips = bd * 10000
    for th in [1.0, 2.0, 3.0, 4.0]:
        b_hi = bd_pips > th
        b_lo = bd_pips <= th
        if b_hi.sum() >= 3:
            print(f"    body>{th:.0f}p: {b_hi.sum():3d} trades  ${p[b_hi].sum():+8.2f}  WR={np.mean(p[b_hi]>0)*100:.0f}%")
        if b_lo.sum() >= 3:
            print(f"    body<={th:.0f}p: {b_lo.sum():3d} trades  ${p[b_lo].sum():+8.2f}  WR={np.mean(p[b_lo]>0)*100:.0f}%")
    print()
