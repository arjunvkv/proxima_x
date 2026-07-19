"""
Cross-Pair Response Deficit Test.

Theory: When pair A moves X pips but pair B (correlated) moves
less than expected, the deficit predicts B catching up later.

This is NOT lead-lag. Lead-lag asks "does A predict B directionally."
Response deficit asks "when B fails to respond, does convergence happen?"

Works at M1 because it measures RELATIVE displacement, not tick ordering.
"""
import sys, os
import numpy as np
from datetime import datetime
from numba import jit

sys.path.insert(0, os.path.dirname(__file__))
from data import TempCache, PAIRS, pip

LEADER_FOLLOWER_PAIRS = [
    ("EURUSD", "GBPUSD"),  # EUR bloc
    ("AUDUSD", "NZDUSD"),  # AUD/NZD bloc
    ("EURJPY", "GBPJPY"),  # JPY crosses
]

PAIR_IDX = {p: i for i, p in enumerate(PAIRS)}

@jit(nopython=True)
def rolling_beta(x, y, lookback=20):
    """Rolling beta: how much y moves per unit x."""
    n = len(x)
    beta = np.zeros(n)
    for i in range(lookback, n):
        xw = x[i-lookback:i]
        yw = y[i-lookback:i]
        xm = np.mean(xw)
        ym = np.mean(yw)
        num = np.sum((xw - xm) * (yw - ym))
        den = np.sum((xw - xm) ** 2)
        beta[i] = num / den if den != 0 else 0
    return beta

@jit(nopython=True)
def rolling_r2(x, y, lookback=20):
    """Rolling R² of x vs y."""
    n = len(x)
    r2 = np.zeros(n)
    for i in range(lookback, n):
        xw = x[i-lookback:i]
        yw = y[i-lookback:i]
        xm = np.mean(xw)
        ym = np.mean(yw)
        num = np.sum((xw - xm) * (yw - ym))
        den_x = np.sum((xw - xm) ** 2)
        den_y = np.sum((yw - ym) ** 2)
        if den_x * den_y > 0:
            r = num / np.sqrt(den_x * den_y)
            r2[i] = r * r
    return r2


print("=" * 70)
print("CROSS-PAIR RESPONSE DEFICIT TEST")
print("=" * 70)

cache = TempCache(7)
aligned, times, _ = cache.get()
n = len(times)

ohlc = {}
for pi, pair in enumerate(PAIRS):
    ohlc[pair] = {
        "close": aligned[:, pi, 3],
        "high":  aligned[:, pi, 1],
        "low":   aligned[:, pi, 2],
    }

all_results = []

for leader_name, follower_name in LEADER_FOLLOWER_PAIRS:
    print(f"\n{'─'*60}")
    print(f"  {leader_name} → {follower_name}")
    print(f"{'─'*60}")

    lc = ohlc[leader_name]["close"]
    fc = ohlc[follower_name]["close"]

    # Returns: ret[i] = close[i+1] - close[i] (return from bar i to i+1)
    l_ret = np.diff(lc)
    f_ret = np.diff(fc)

    for lookback in [10, 20, 30, 50]:
        if lookback >= len(l_ret):
            continue

        beta = rolling_beta(l_ret, f_ret, lookback)
        _ = rolling_r2(l_ret, f_ret, lookback)

        # Signal at index i (using returns up to bar i):
        #   leader return = l_ret[i-1] (return from bar i-1 to i)
        #   beta based on [i-lookback:i] returns
        #   expected follower return = beta[i-1] * leader_return
        #   actual follower return = f_ret[i-1]
        #   deficit = expected - actual
        # Predict: follower return at next bar = f_ret[i]

        sig_idx = lookback  # first index where we have beta
        deficit = np.zeros(len(l_ret) - 1)
        deficit_pips = np.zeros(len(l_ret) - 1)
        f_ret_next = np.zeros(len(l_ret) - 1)
        leader_scale = 10000 if leader_name in ("EURUSD", "GBPUSD", "AUDUSD", "NZDUSD") else 100

        for i in range(sig_idx, len(l_ret) - 1):
            # Expected follower return at bar i-1 to i
            exp_ret = beta[i] * l_ret[i-1]  # beta[i] uses returns [i-lb:i]
            act_ret = f_ret[i-1]            # what follower actually did
            deficit[i-1] = exp_ret - act_ret
            deficit_pips[i-1] = deficit[i-1] * leader_scale
            f_ret_next[i-1] = f_ret[i]       # follower return at next bar

        deficit_std = np.std(deficit)
        if deficit_std == 0:
            continue

        for z_thresh in [0.5, 1.0, 1.5, 2.0]:
            sig_long = deficit > z_thresh * deficit_std
            sig_short = deficit < -z_thresh * deficit_std

            n_long = np.sum(sig_long)
            if n_long >= 20:
                correct_long = np.sum(f_ret_next[sig_long] > 0)
                wr_long = correct_long / n_long
                avg_catchup = np.mean(f_ret_next[sig_long]) * leader_scale
                all_results.append({
                    "pair": f"{leader_name}→{follower_name}",
                    "type": "LAG_CATCHUP", "lb": lookback,
                    "z": z_thresh, "wr": wr_long, "trades": n_long,
                    "avg_catchup_pips": avg_catchup,
                })

            n_short = np.sum(sig_short)
            if n_short >= 20:
                correct_short = np.sum(f_ret_next[sig_short] < 0)
                wr_short = correct_short / n_short
                avg_catchup_s = np.mean(f_ret_next[sig_short]) * leader_scale
                all_results.append({
                    "pair": f"{leader_name}→{follower_name}",
                    "type": "LEAD_CATCHUP", "lb": lookback,
                    "z": z_thresh, "wr": wr_short, "trades": n_short,
                    "avg_catchup_pips": avg_catchup_s,
                })

        # Print summary for this lookback
        beta_mean = np.mean(beta[lookback:lookback+5])
        sig = np.abs(deficit) > 1.0 * deficit_std
        ns = np.sum(sig)
        if ns >= 50:
            dir_pred = np.sign(deficit)
            dir_actual = np.sign(f_ret_next)
            correct = np.sum(dir_pred[sig] == dir_actual[sig])
            wr = correct / ns
            avg_def = np.mean(np.abs(deficit_pips[sig]))
            print(f"  lb={lookback}: β={beta_mean:.2f} z>1: WR={wr:.1%}({ns}) avg|deficit|={avg_def:.1f}pips")

print()
print("=" * 70)
print("BEST RESULTS (sorted by WR, min 30 trades)")
print("=" * 70)
print()

all_results.sort(key=lambda r: (-r["wr"], -r["trades"]))
seen = set()
for r in all_results:
    if r["trades"] < 30:
        continue
    key = (r["pair"], r["type"], r["lb"], r["z"])
    if key in seen:
        continue
    seen.add(key)
    print(f"  {r['wr']:.1%}  {r['pair']}/{r['type']} lb={r['lb']} z>{r['z']}  "
          f"({r['trades']} trades, avg_catchup={r['avg_catchup_pips']:.2f}pips)")

print()
print("=" * 70)
print("VERDICT")
print("=" * 70)
print()
print("Response deficit tests if the FOLLOWER catches up when it fails")
print("to respond to the LEADER. This is structurally different from")
print("lead-lag because it measures CONVERGENCE, not prediction.")
print()
print("If WR > 55% with >200 trades, response deficit is real at M1.")
print("If WR ~ 50%, even convergence doesn't happen at M1 resolution.")
