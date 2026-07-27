"""Check GBPAUD filtered run details."""
import re, numpy as np

path = "C:\\Users\\Arjun Sasi\\AppData\\Roaming\\MetaQuotes\\Tester\\D0E8209F77C8CF37AD8BF550E51FF075\\Agent-127.0.0.1-3000\\logs\\20260726.log"
with open(path, "rb") as f:
    raw = f.read()
text = raw.decode("utf-16-le", errors="replace")
lines = text.split("\n")

fbs = [i for i,l in enumerate(lines) if "final balance" in l]

# GBPAUD unfiltered = run 8, GBPAUD filtered = run 9
for label, ri in [("GBPAUD unfiltered", 8), ("GBPAUD filtered", 9)]:
    start = fbs[ri-1]+1 if ri > 0 else 0
    run = lines[start:fbs[ri]]
    deals = []
    for l in run:
        m = re.search(r"deal #(\d+) (buy|sell) [\d.]+ (\w+) at ([\d.]+) done", l)
        if m:
            deals.append({"tkt":int(m.group(1)),"side":m.group(2),"sym":m.group(3),"pr":float(m.group(4))})
    pnls = []
    ii = 0
    while ii < len(deals)-1:
        for j in range(ii+1, len(deals)):
            if deals[j]["side"] != deals[ii]["side"] and deals[j]["sym"] == deals[ii]["sym"]:
                if deals[ii]["side"] == "buy":
                    p = (deals[j]["pr"] - deals[ii]["pr"]) * 0.75 * 100000
                else:
                    p = (deals[ii]["pr"] - deals[j]["pr"]) * 0.75 * 100000
                pnls.append(p)
                ii = j+1
                break
        else:
            ii += 1
    a = np.array(pnls)
    comm = len(pnls) * 0.75 * 5 * 2
    bal = lines[fbs[ri]].strip()[:70]
    print(f"{label:20s}: {len(pnls):2d} tr  gross=${a.sum():+8.2f}  ${a.mean():+7.2f}/tr  WR={np.mean(a>0)*100:.0f}%  comm=${comm:5.2f}  net=${a.sum()-comm:+8.2f}  {bal}")
    if ri == 9:  # filtered
        for i, p in enumerate(pnls):
            print(f"    [{i}] ${p:+.2f}")
