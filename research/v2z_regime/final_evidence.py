"""Final evidence with correct run indexing."""
import re
import numpy as np

path = "C:\\Users\\Arjun Sasi\\AppData\\Roaming\\MetaQuotes\\Tester\\D0E8209F77C8CF37AD8BF550E51FF075\\Agent-127.0.0.1-3000\\logs\\20260726.log"
with open(path, "rb") as f:
    raw = f.read()
text = raw.decode("utf-16-le", errors="replace")
lines = text.split("\n")

fbs = [i for i,l in enumerate(lines) if "final balance" in l]
print(f"Found {len(fbs)} 'final balance' lines")

# Parse each run
for ri in range(len(fbs)):
    start = fbs[ri-1]+1 if ri > 0 else 0
    run = lines[start:fbs[ri]]
    bal_line = lines[fbs[ri]]
    bal = float(re.search(r"balance ([\d.]+)", bal_line).group(1))
    gross = bal - 10000.0
    
    # Count OPENs and VOL REGIME lines
    opens = len([l for l in run if "OPEN EURAUD" in l])
    vols = len([l for l in run if "VOL REGIME" in l])
    micro_on = len([l for l in run if "VOL REGIME" in l and "use_micro=true" in l])
    
    # Parse deals
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
    
    comm = len(pnls) * 0.75 * 5 * 2
    net = gross - comm
    wr = np.mean(np.array(pnls) > 0) * 100 if pnls else 0
    print(f"Run {ri:2d}: bal=${bal:.2f} gross=${gross:+7.2f} trades={len(pnls):2d} "
          f"comm=${comm:5.2f} net=${net:+8.2f} wr={wr:.0f}% "
          f"opens={opens:2d} vol_regime={vols:2d} micro_on={micro_on:2d}")

print("\nFocus on LAST 2 RUNS (vol5.0 threshold):")
print(f"  Fwd vol5.0 = Run {len(fbs)-2}")
print(f"  OOS vol5.0 = Run {len(fbs)-1}")
