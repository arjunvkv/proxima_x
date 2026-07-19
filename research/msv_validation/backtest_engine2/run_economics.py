"""Economic viability: EURJPY→GBPJPY response deficit."""
import sys, os, numpy as np
from datetime import datetime, timezone
sys.path.insert(0, os.path.dirname(__file__))
from data import TempCache, PAIRS
from numba import jit

PAIR_IDX = {p: i for i, p in enumerate(PAIRS)}

@jit(nopython=True)
def rolling_beta(x, y, lookback):
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

cache = TempCache(7)
aligned, times, _ = cache.get()
n = len(times)

leader_name, follower_name = 'EURJPY', 'GBPJPY'
lc = aligned[:, PAIR_IDX[leader_name], 3]
fc = aligned[:, PAIR_IDX[follower_name], 3]
l_ret = np.diff(lc)
f_ret = np.diff(fc)

dates = [datetime.fromtimestamp(t, tz=timezone.utc) for t in times]
date_labels = [d.strftime('%Y-%m-%d') for d in dates]
unique_dates = sorted(set(date_labels))
bar_date = np.array(date_labels[:len(l_ret)])

# Best config
lb, hold, zt = 10, 20, 2.0
beta = rolling_beta(l_ret, f_ret, lb)

print('=' * 70)
print('ECONOMIC ANALYSIS: EURJPY→GBPJPY Response Deficit')
print(f'Config: lb={lb} hold={hold} z>{zt}')
print('=' * 70)

for test_date in unique_dates:
    mask = bar_date == test_date
    deficits, catchups = [], []
    for i in range(lb, len(l_ret) - hold):
        if not mask[i]:
            continue
        exp_ret = beta[i] * l_ret[i-1]
        act_ret = f_ret[i-1]
        deficits.append((exp_ret - act_ret) * 100)
        cum_ret = np.sum(f_ret[i:i+hold])
        catchups.append(cum_ret * 100)

    deficits = np.array(deficits)
    catchups = np.array(catchups)
    if len(deficits) == 0 or np.std(deficits) == 0:
        continue

    sig = deficits > zt * np.std(deficits)
    ns = np.sum(sig)
    if ns < 3:
        continue

    wins = catchups[sig] > 0
    wr = np.sum(wins) / ns
    avg_win = np.mean(catchups[sig][wins]) if np.any(wins) else 0
    avg_loss = np.mean(catchups[sig][~wins]) if np.any(~wins) else 0
    avg_all = np.mean(catchups[sig])
    weekday = datetime.strptime(test_date, '%Y-%m-%d').strftime('%a')

    print(f'  {test_date} ({weekday}): WR={wr:.1%} trades={ns} '
          f'avg={avg_all:.2f}p win_avg={avg_win:.2f}p loss_avg={avg_loss:.2f}p')

print()
deficits, catchups = [], []
for i in range(lb, len(l_ret) - hold):
    exp_ret = beta[i] * l_ret[i-1]
    act_ret = f_ret[i-1]
    deficits.append((exp_ret - act_ret) * 100)
    cum_ret = np.sum(f_ret[i:i+hold])
    catchups.append(cum_ret * 100)

deficits = np.array(deficits)
catchups = np.array(catchups)
sig = deficits > zt * np.std(deficits)
ns = np.sum(sig)
wins = catchups[sig] > 0
wr = np.sum(wins) / ns
avg_win = np.mean(catchups[sig][wins])
avg_loss = np.mean(catchups[sig][~wins])
avg_all = np.mean(catchups[sig])

print(f'FULL DATASET:')
print(f'  WR={wr:.1%} ({ns} trades)')
print(f'  Avg win:  {avg_win:.2f} pips')
print(f'  Avg loss: {avg_loss:.2f} pips')
print(f'  Avg all:  {avg_all:.2f} pips')
print(f'  Win/loss ratio: {abs(avg_win/avg_loss):.2f}x')
print(f'  Gross expectancy: {wr*avg_win + (1-wr)*avg_loss:.2f} pips/trade')

for spread_bps in [0.3, 0.5, 0.7, 1.0]:
    net = wr * avg_win + (1-wr) * avg_loss - spread_bps
    net_usd = net * 10
    print(f'  After {spread_bps:.1f}p spread: {net:.2f} pips ≈ ${net_usd:.0f}/trade (1 lot)')
