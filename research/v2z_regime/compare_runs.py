"""Compare all filtered runs."""
import re, numpy as np

path = "C:\\Users\\Arjun Sasi\\AppData\\Roaming\\MetaQuotes\\Tester\\D0E8209F77C8CF37AD8BF550E51FF075\\Agent-127.0.0.1-3000\\logs\\20260726.log"
with open(path, "rb") as f:
    raw = f.read()
text = raw.decode("utf-16-le", errors="replace")
lines = text.split("\n")

fbs = [i for i,l in enumerate(lines) if "final balance" in l]

for ri in range(4, 8):
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
    bal = lines[fbs[ri]].strip()[:70]
    label = ["OOS(no filter)", "Fwd(body+sm)", "Fwd(+tv)", "OOS(+tv)"][ri-4]
    comm = len(pnls) * 0.75 * 5 * 2
    net = a.sum() - comm
    print(f"{label:16s}: {len(pnls):2d} tr  gross=${a.sum():+7.2f}  ${a.mean():+6.2f}/tr  comm=${comm:5.2f}  net=${net:+7.2f}  {bal}")
