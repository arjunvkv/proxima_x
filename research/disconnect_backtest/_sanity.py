"""Quick sanity check for the gap backtest pipeline."""
import numpy as np, pandas as pd, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'research', 'disconnect_backtest'))
os.chdir(os.path.join(os.path.dirname(__file__)))
from currency_pressure_gap_test import fast_rolling_z, load_and_precompute, run_fast, make_gaps, Z_THRESH, MIN_HIST, HOLD_MIN, SPREADS_BPS, DEFAULT_SPREAD

# Load
data = load_and_precompute()
print("Loaded %d bars, %d pairs" % (len(data["prices"]), len(data["pairs_list"])))
print("curr_wm keys:", sorted(data["curr_wm"].keys()))
print("best_j:", {k: "OK" if v is not None else "NONE" for k, v in data["best_j"].items()})
print()

# Take May
mask = data["month_labels"].strftime("%Y-%m") == "2026-05"
mp, mt = data["prices"][mask], data["ts"][mask]
print("May bars:", len(mp))

# Run WITHOUT gaps
lr = np.diff(np.log(mp + 1e-15), axis=0)

# Currency returns
cnames = sorted(data["curr_wm"].keys())
cr_list = []
for c in cnames:
    cols, signs, w = data["curr_wm"][c]
    cr_list.append(np.sum(lr[:, cols] * signs * w, axis=1))
cr = np.column_stack(cr_list)

z = fast_rolling_z(cr)
print("Z shape:", z.shape)

# Count signals
for ci, c in enumerate(cnames):
    n_sig = np.sum(~np.isnan(z[:, ci]) & (np.abs(z[:, ci]) >= Z_THRESH))
    j = data["best_j"].get(c)
    print("  %s: %d Z>2.0 signals (best_j=%s)" % (c, n_sig, "OK" if j is not None else "NONE"))

# Run full pipeline
dirty, clean = run_fast(mp, mt, data)
print("\nDirty trades:", len(dirty))
print("Clean trades:", len(clean))
if dirty:
    print("Dirty WR: %.1f%%" % (np.mean([t["correct"] for t in dirty]) * 100))
if clean:
    print("Clean WR: %.1f%%" % (np.mean([t["correct"] for t in clean]) * 100))
