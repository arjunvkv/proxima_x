"""Combined 3-signal response deficit across ALL days via yfinance M1 data."""
import sys, numpy as np, pandas as pd
import yfinance as yf
from datetime import datetime
from numba import jit

@jit(nopython=True)
def rolling_beta(x, y, lb):
    n = len(x)
    beta = np.zeros(n)
    for i in range(lb, n):
        xw = x[i-lb:i]; yw = y[i-lb:i]
        xm = np.mean(xw); ym = np.mean(yw)
        num = np.sum((xw-xm)*(yw-ym))
        den = np.sum((xw-xm)**2)
        beta[i] = num/den if den != 0 else 0
    return beta

print("Downloading M1 data...")
tickers = {
    'EURJPY': yf.Ticker('EURJPY=X').history(period='5d', interval='1m'),
    'GBPJPY': yf.Ticker('GBPJPY=X').history(period='5d', interval='1m'),
    'EURUSD': yf.Ticker('EURUSD=X').history(period='5d', interval='1m'),
}

common_idx = tickers['EURJPY'].index.intersection(tickers['GBPJPY'].index).intersection(tickers['EURUSD'].index)
print(f"Aligned bars: {len(common_idx)}: {common_idx[0]} to {common_idx[-1]}")

close = {pair: df.loc[common_idx, 'Close'].values.astype(np.float64) for pair, df in tickers.items()}
days = [t.strftime('%a') for t in common_idx]
unique_days = sorted(set(days), key=lambda d: ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'].index(d))
print(f"Days: {unique_days}")
print(f"Bars per day: {[days.count(d) for d in unique_days]}")

ej_c, gj_c, eu_c = close['EURJPY'], close['GBPJPY'], close['EURUSD']
ej_ret, gj_ret, eu_ret = np.diff(ej_c), np.diff(gj_c), np.diff(eu_c)
ns = len(ej_ret)

LB = 10
HOLD = 20

beta_eu_ej = rolling_beta(eu_ret, ej_ret, LB)
beta_ej_gj = rolling_beta(ej_ret, gj_ret, LB)
beta_eu_gj = rolling_beta(eu_ret, gj_ret, LB)

def_eu_ej = np.array([(beta_eu_ej[i]*eu_ret[i-1]-ej_ret[i-1])*100 if i>=LB else 0 for i in range(ns)])
def_ej_gj = np.array([(beta_ej_gj[i]*ej_ret[i-1]-gj_ret[i-1])*100 if i>=LB else 0 for i in range(ns)])
def_eu_gj = np.array([(beta_eu_gj[i]*eu_ret[i-1]-gj_ret[i-1])*100 if i>=LB else 0 for i in range(ns)])

catch_gj = np.array([np.sum(gj_ret[i:i+HOLD])*100 if i<=ns-HOLD else 0 for i in range(ns)])
catch_ej = np.array([np.sum(ej_ret[i:i+HOLD])*100 if i<=ns-HOLD else 0 for i in range(ns)])

# Trading day index: sig[i] is evaluated at bar i (using returns up to i-1)
# Entry at bar i, day = days[i+1] (since days is aligned with close price, bar i entry corresponds to day of bar i+0)
# Actually: sig is computed using returns ending at bar i-1. 
# For entry at bar i, exit at bar i+HOLD, we use days[i] as the entry day.
# sig[i] is valid for entry at time days[i+1] (1 bar after the last return in the lookback)
# Let's just use days[i] for entry at sig[i].

# sig[LB:] are valid entries (need LB bars of history)
# Entry at bar i produces trade on day days[i]
# Exit at bar i+HOLD (20 min later) - may cross into next day

print("\n" + "=" * 70)
print("SINGLE vs COMBINED SIGNALS — 5-DAY OVERALL")
print("=" * 70)

for zt in [1.5, 2.0, 2.5, 3.0]:
    s1 = def_eu_ej > zt * np.std(def_eu_ej[LB:])
    s2 = def_ej_gj > zt * np.std(def_ej_gj[LB:])
    s3 = def_eu_gj > zt * np.std(def_eu_gj[LB:])
    
    configs = [
        ("1. EURUSD->EURJPY", s1, catch_ej),
        ("2. EURJPY->GBPJPY", s2, catch_gj),
        ("3. EURUSD->GBPJPY", s3, catch_gj),
        ("1 AND 2", s1 & s2, catch_gj),
        ("1 OR 2", s1 | s2, catch_gj),
        ("2 AND 3", s2 & s3, catch_gj),
        ("2 OR 3", s2 | s3, catch_gj),
        ("1 AND 2 AND 3 (ALL 3)", s1 & s2 & s3, catch_gj),
        ("1 OR 2 OR 3 (ANY)", s1 | s2 | s3, catch_gj),
    ]
    
    print(f"\n--- z > {zt:.0f} ---")
    print(f"{'Signal':<22} {'WR':>6} {'Trades':>7} {'Avg(p)':>8} {'EV(p)':>8}")
    print("-" * 53)
    for name, sig, catch_arr in configs:
        mask = (sig) & (catch_arr != 0)
        idx = np.where(mask & (np.arange(ns) >= LB))[0]
        n = len(idx)
        if n < 5:
            continue
        vals = catch_arr[idx]
        wr = np.mean(vals > 0)
        avg = np.mean(vals)
        wins = vals[vals > 0]
        losses = vals[vals <= 0]
        ev = wr*np.mean(wins)+(1-wr)*np.mean(losses) if len(wins)>0 and len(losses)>0 else 0
        print(f"{name:<22} {wr:>5.0%} {n:>7d} {avg:>7.2f} {ev:>+7.2f}")

# ============================================================
# DAY-BY-DAY for best configs
# ============================================================
print("\n" + "=" * 70)
print("DAY-BY-DAY — EURJPY->GBPJPY AND EURUSD->GBPJPY (z>2.0)")
print("=" * 70)

std2 = np.std(def_ej_gj[LB:])
std3 = np.std(def_eu_gj[LB:])
s2 = def_ej_gj > 2.0 * std2
s3 = def_eu_gj > 2.0 * std3

for name, sig in [
    ("EURJPY->GBPJPY alone", s2),
    ("EURUSD->GBPJPY alone", s3),
    ("BOTH AND", s2 & s3),
    ("BOTH OR", s2 | s3),
]:
    print(f"\n  {name}:")
    print(f"  {'Day':<6} {'WR':>6} {'Trades':>7} {'Avg(p)':>8}")
    print("  " + "-" * 30)
    total_n = 0
    total_wins = 0
    for d in unique_days:
        day_trades = []
        for i in range(LB, ns - HOLD):
            if days[i] == d and sig[i] and catch_gj[i] != 0:
                day_trades.append(catch_gj[i])
        n = len(day_trades)
        if n < 2:
            continue
        wr = np.mean(np.array(day_trades) > 0)
        avg = np.mean(day_trades)
        print(f"  {d:<6} {wr:>5.0%} {n:>7d} {avg:>7.2f}")
        total_n += n
        total_wins += np.sum(np.array(day_trades) > 0)
    
    if total_n > 0:
        print(f"  {'TOTAL':<6} {total_wins/total_n:>5.0%} {total_n:>7d}")

# ============================================================
# ALL 3 AND day-by-day
# ============================================================
print("\n" + "=" * 70)
print("ALL 3 AND — DAY-BY-DAY (z>1.5 and z>2.0)")
print("=" * 70)

for zt in [1.5, 2.0]:
    s1 = def_eu_ej > zt * np.std(def_eu_ej[LB:])
    s2 = def_ej_gj > zt * np.std(def_ej_gj[LB:])
    s3 = def_eu_gj > zt * np.std(def_eu_gj[LB:])
    sig_all = s1 & s2 & s3
    
    print(f"\n  z > {zt:.0f}:")
    print(f"  {'Day':<6} {'WR':>6} {'Trades':>7} {'Avg(p)':>8} {'EV(p)':>8}")
    print("  " + "-" * 38)
    total_n = 0
    all_returns = []
    for d in unique_days:
        day_returns = []
        for i in range(LB, ns - HOLD):
            if days[i] == d and sig_all[i] and catch_gj[i] != 0:
                day_returns.append(catch_gj[i])
        n = len(day_returns)
        if n < 2:
            continue
        wr = np.mean(np.array(day_returns) > 0)
        avg = np.mean(day_returns)
        wins = np.array([r for r in day_returns if r > 0])
        losses = np.array([r for r in day_returns if r <= 0])
        ev = wr*np.mean(wins)+(1-wr)*np.mean(losses) if len(wins)>0 and len(losses)>0 else 0
        print(f"  {d:<6} {wr:>5.0%} {n:>7d} {avg:>7.2f} {ev:>+7.2f}")
        total_n += n
        all_returns.extend(day_returns)
    
    if total_n > 0:
        all_arr = np.array(all_returns)
        wr_t = np.mean(all_arr > 0)
        avg_t = np.mean(all_arr)
        wins_t = all_arr[all_arr > 0]
        losses_t = all_arr[all_arr <= 0]
        ev_t = wr_t*np.mean(wins_t)+(1-wr_t)*np.mean(losses_t) if len(wins_t)>0 and len(losses_t)>0 else 0
        print(f"  {'TOTAL':<6} {wr_t:>5.0%} {total_n:>7d} {avg_t:>7.2f} {ev_t:>+7.2f}")
