"""s4_align_local.py — cached vs terminal daily closes, EURUSD + DXY.cash.

Reads: s4_cached_closes.json (local dump) and /tmp/s4_term_closes.json (VPS dump).
Prints per-day close deltas + corr for both symbols, and the residual impact.
"""
import json, sys
import numpy as np

C = json.load(open("s4_cached_closes.json"))
T = json.load(open(r"C:\Users\arjun\AppData\Local\Temp\s4_term_closes.json"))

for sym, ckey, tkey in [("DXY.cash", "dcx", "DXY.cash"), ("EURUSD", "ece", "EURUSD")]:
    cd = np.array(C["days"]); cc = np.array(C[ckey])
    td = np.array(T[tkey]["days"]); tc = np.array(T[tkey]["close"])
    common = np.intersect1d(cd, td)
    i1 = np.searchsorted(cd, common); i2 = np.searchsorted(td, common)
    d = cc[i1] - tc[i2]
    print(f"== {sym}: common={len(common)} days {common[0]}..{common[-1]}")
    print(f"   corr={np.corrcoef(cc[i1], tc[i2])[0,1]:.6f}  max|d|={np.abs(d).max():.5f}  mean|d|={np.abs(d).mean():.5f}")
    for k in range(max(0, len(common) - 8), len(common)):
        print(f"   {common[k]}  cached={cc[i1[k]]:.5f}  term={tc[i2[k]]:.5f}  d={cc[i1[k]]-tc[i2[k]]:+.5f}")
