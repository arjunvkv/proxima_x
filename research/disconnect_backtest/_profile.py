import numpy as np, time
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'research', 'disconnect_backtest'))
os.chdir(os.path.join(os.path.dirname(__file__)))
from currency_pressure_gap_test import *

# Load
data = load_and_precompute()
print("Data: %d bars" % len(data["prices"]))

# Take May
mask = data["month_labels"].strftime("%Y-%m") == "2026-05"
mp, mt = data["prices"][mask], data["ts"][mask]
print("May: %d bars" % len(mp))

lr = np.diff(np.log(mp + 1e-15), axis=0)

# Precompute currency returns
cr_list = []
for c in sorted(data["curr_wm"].keys()):
    cols, signs, w = data["curr_wm"][c]
    cr_list.append(np.sum(lr[:, cols] * signs * w, axis=1))
cr = np.column_stack(cr_list)

t = time.time()
z = fast_rolling_z(cr)
print("fast_rolling_z: %.4fs" % (time.time() - t))

# Time get_trades
t = time.time()
cnames = sorted(data["curr_wm"].keys())
gap = np.zeros(len(lr), dtype=bool)
n_trades = 0
for i in range(MIN_HIST, len(lr) - HOLD_MIN):
    for ci, c in enumerate(cnames):
        zv = z[i, ci]
        if np.isnan(zv) or abs(zv) < Z_THRESH:
            continue
        j = data["best_j"].get(c)
        if j is None:
            continue
        direction = 1 if zv > 0 else -1
        fwd_ret = np.sum(lr[i+1:i+1+HOLD_MIN, j])
        pnl = fwd_ret * direction
        n_trades += 1
t2 = time.time()
print("get_trades: %.4fs, %d trades" % (t2 - t, n_trades))

# Check how many Z vals exceed threshold
for ci, c in enumerate(cnames):
    n_sig = np.sum(~np.isnan(z[:, ci]) & (np.abs(z[:, ci]) >= Z_THRESH))
    print("  %s: %d Z>2.0 signals" % (c, n_sig))
