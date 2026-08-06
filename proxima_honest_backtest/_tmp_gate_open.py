import sys, time, importlib.util
sys.path.insert(0, "proxima_honest_backtest")

spec = importlib.util.spec_from_file_location("run_all", "proxima_honest_backtest/strategies/run_all.py")
ra = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ra)

import pandas as pd

CAP = 12000

def slice_last(df, bars):
    if df is None or df.empty:
        return df
    return df.tail(bars).reset_index(drop=True)

results = {}
for rec in ra.STRATEGIES:
    data = ra.load_all(rec["pairs"])
    data = {s: slice_last(d, CAP) for s, d in data.items()}
    if not data:
        print(f"  {rec['name']:28s} SKIP")
        continue
    t0 = time.time()
    gate = ra.run_lookahead_gate(rec, data)
    results[rec["key"]] = gate
    print(f"  {gate.report_line()}  ({time.time()-t0:.1f}s)")

all_pass = all(g.passed for g in results.values())
print(f"\nALL GATE PASS (open-exit): {all_pass}")