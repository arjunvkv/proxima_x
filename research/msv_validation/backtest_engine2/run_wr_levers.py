"""WR improvement levers for response deficit (all except day filtering)."""
import sys, os, numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime
from numba import jit

PROJ = Path(__file__).resolve().parents[3]
PAIRS = ['EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'NZDUSD', 'EURJPY', 'GBPJPY']

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

def load_market_data():
    """Load M1 data from market parquets for ALL available pairs."""
    mkt_dir = PROJ / 'data' / 'market'
    data = {}
    times = None
    for pair in PAIRS:
        fp = mkt_dir / f'{pair}.parquet'
        if not fp.exists():
            print(f"WARN: {pair} parquet not found")
            continue
        df = pd.read_parquet(fp)
        ts = pd.to_datetime(df.timestamp.values)
        if times is None:
            times = ts
        data[pair] = {
            'open': df.open.values,
            'high': df.high.values, 
            'low': df.low.values,
            'close': df.close.values,
        }
        print(f"  {pair}: {len(df)} bars, {ts[0]} to {ts[-1]}")
    return data, times

def analyze_pair(leader, follower, ohlc, lb=10, hold=10):
    lc = ohlc[leader]['close']
    fc = ohlc[follower]['close']
    l_ret = np.diff(lc)
    f_ret = np.diff(fc)
    beta = rolling_beta(l_ret, f_ret, lb)
    ns = len(l_ret)
    
    deficits = np.zeros(ns)
    for i in range(lb, ns):
        deficits[i] = (beta[i] * l_ret[i-1] - f_ret[i-1]) * 100
    
    catchups = np.zeros(ns)
    for i in range(lb, ns - hold):
        catchups[i] = np.sum(f_ret[i:i+hold]) * 100
    
    return deficits, catchups, f_ret

print("=" * 70)
print("WR IMPROVEMENT LEVERS FOR RESPONSE DEFICIT")
print("=" * 70)
print("\nLoading M1 market data (1 day, 1440 bars)...")
data, times = load_market_data()
hours = np.array([t.hour for t in times])

# ---- TEST CONFIGS ----
configs = [
    ("EURJPY", "GBPJPY", 10, 20),
    ("EURJPY", "GBPJPY", 10, 10),
    ("EURJPY", "GBPJPY", 20, 20),
    ("EURUSD", "EURJPY", 10, 20),
    ("EURUSD", "EURJPY", 20, 20),
    ("GBPUSD", "GBPJPY", 10, 10),
    ("USDJPY", "EURJPY", 10, 20),
    ("GBPUSD", "EURUSD", 10, 5),
    ("EURUSD", "GBPUSD", 10, 10),
]

# A. Z-THRESHOLD SWEEP
print("\n" + "-" * 70)
print("A. Z-THRESHOLD SWEEP (tighter = higher WR, fewer trades)")
print("-" * 70)
print(f"{'Pair':<22} {'lb':>3} {'hold':>4} | z=1.5 WR  Tr | z=2.0 WR  Tr | z=2.5 WR  Tr | z=3.0 WR  Tr | net")
print("-" * 85)
for leader, follower, lb, hold in configs:
    if leader not in data or follower not in data:
        continue
    deficits, catchups, _ = analyze_pair(leader, follower, data, lb, hold)
    std = np.std(deficits[lb:])
    if std == 0:
        continue
    parts = []
    for zt in [1.0, 1.5, 2.0, 2.5, 3.0]:
        sig = deficits > zt * std
        idx = np.where(sig & (catchups != 0))[0]
        n = len(idx)
        if n < 5:
            parts.append(f"z={zt:.0f}:  ---")
            continue
        wr = np.mean(catchups[idx] > 0)
        avg = np.mean(catchups[idx])
        # Estimate net profit: (avg_win * winrate - avg_loss * (1-winrate)) per trade
        wins = catchups[idx][catchups[idx] > 0]
        losses = catchups[idx][catchups[idx] <= 0]
        avg_w = np.mean(wins) if len(wins) > 0 else 0
        avg_l = np.mean(losses) if len(losses) > 0 else 0
        ev = wr * avg_w + (1-wr) * avg_l
        parts.append(f"{wr:.0%} {n:3d} ev={ev:+.1f}")
    print(f"{leader}->{follower:<14} {lb:>3} {hold:>4} | " + " | ".join(parts))

# B. SESSION FILTERING
print("\n" + "-" * 70)
print("B. SESSION FILTERING (best session for EURJPY->GBPJPY)")
print("-" * 70)
leader, follower = "EURJPY", "GBPJPY"
deficits, catchups, _ = analyze_pair(leader, follower, data, 10, 20)
std = np.std(deficits[10:])
sig_all = deficits > 2.0 * std

print(f"{'Session':<16} {'Hours':>8} {'WR':>6} {'Trades':>7} {'Avg(p)':>8} {'EV(p)':>8}")
print("-" * 55)
for sname, hr_range in [
    ("Tokyo", range(0, 9)),
    ("London", range(7, 17)),
    ("NY", range(12, 22)),
    ("Asia overlap", range(0, 5)),
    ("Tokyo-London", range(7, 10)),
    ("London-NY", range(12, 17)),
    ("All (no filter)", range(0, 24)),
]:
    hr_mask = np.array([h in hr_range for h in hours[1:]])
    mask = sig_all & hr_mask & (catchups != 0)
    n = np.sum(mask)
    if n < 5:
        continue
    wr = np.mean(catchups[mask] > 0)
    avg = np.mean(catchups[mask])
    wins = catchups[mask][catchups[mask] > 0]
    losses = catchups[mask][catchups[mask] <= 0]
    ev = wr * np.mean(wins) + (1-wr) * np.mean(losses) if len(wins) > 0 and len(losses) > 0 else 0
    print(f"{sname:<16} {hr_range[0]:>2d}-{hr_range[-1]:>2d}h   {wr:.0%} {n:7d} {avg:>7.2f} {ev:>+7.2f}")

# C. COMBINED SIGNALS
print("\n" + "-" * 70)
print("C. COMBINED SIGNALS (multiple confirmations)")
print("-" * 70)

# Use EURUSD AND EURJPY as confirmations for GBPJPY direction
eu_close = data.get("EURUSD", {}).get("close")
ej_close = data.get("EURJPY", {}).get("close")
gj_close = data.get("GBPJPY", {}).get("close")

if all(x is not None for x in [eu_close, ej_close, gj_close]):
    eu_ret = np.diff(eu_close)
    ej_ret = np.diff(ej_close)
    gj_ret = np.diff(gj_close)
    ns = len(eu_ret)
    
    # EURUSD -> EURJPY deficit
    bej = rolling_beta(eu_ret, ej_ret, 10)
    dej_eu = np.zeros(ns)
    for i in range(10, ns):
        dej_eu[i] = (bej[i] * eu_ret[i-1] - ej_ret[i-1]) * 100
    
    # EURJPY -> GBPJPY deficit
    bgj = rolling_beta(ej_ret, gj_ret, 10)
    dgj_ej = np.zeros(ns)
    for i in range(10, ns):
        dgj_ej[i] = (bgj[i] * ej_ret[i-1] - gj_ret[i-1]) * 100
    
    # GBP catchups
    catch_gj = np.zeros(ns)
    for i in range(10, ns - 20):
        catch_gj[i] = np.sum(gj_ret[i:i+20]) * 100
    
    # EURJPY catchups
    catch_ej = np.zeros(ns)
    for i in range(10, ns - 20):
        catch_ej[i] = np.sum(ej_ret[i:i+20]) * 100
    
    sd1 = np.std(dej_eu[10:])
    sd2 = np.std(dgj_ej[10:])
    
    print(f"{'Signal Type':<30} {'WR':>6} {'Trades':>7} {'Avg(p)':>8} {'EV(p)':>8}")
    print("-" * 60)
    
    for zt in [1.0, 1.5, 2.0]:
        s1 = dej_eu > zt * sd1
        s2 = dgj_ej > zt * sd2
        
        # Single signals
        for name, sig, catch in [("EURUSD->EURJPY only", s1, catch_ej), 
                                  ("EURJPY->GBPJPY only", s2, catch_gj)]:
            idx = np.where(sig & (catch != 0))[0]
            n = len(idx)
            if n < 5: continue
            wr = np.mean(catch[idx] > 0)
            avg = np.mean(catch[idx])
            wins = catch[idx][catch[idx] > 0]
            losses = catch[idx][catch[idx] <= 0]
            ev = wr*np.mean(wins)+(1-wr)*np.mean(losses) if len(wins)>0 and len(losses)>0 else 0
            print(f"  {name:<28} {wr:.0%} {n:7d} {avg:>7.2f} {ev:>+7.2f}")
        
        # AND: BOTH must agree
        idx_and = np.where(s1 & s2 & (catch_gj != 0))[0]
        n_and = len(idx_and)
        if n_and >= 5:
            wr_and = np.mean(catch_gj[idx_and] > 0)
            avg_and = np.mean(catch_gj[idx_and])
            wins = catch_gj[idx_and][catch_gj[idx_and] > 0]
            losses = catch_gj[idx_and][catch_gj[idx_and] <= 0]
            ev = wr_and*np.mean(wins)+(1-wr_and)*np.mean(losses) if len(wins)>0 and len(losses)>0 else 0
            print(f"  EURUSD+EURJPY->GBPJPY (AND): {wr_and:.0%} {n_and:7d} {avg_and:>7.2f} {ev:>+7.2f}")
        
        # OR: at least one
        idx_or = np.where((s1 | s2) & (catch_gj != 0))[0]
        n_or = len(idx_or)
        if n_or >= 5:
            wr_or = np.mean(catch_gj[idx_or] > 0)
            avg_or = np.mean(catch_gj[idx_or])
            wins = catch_gj[idx_or][catch_gj[idx_or] > 0]
            losses = catch_gj[idx_or][catch_gj[idx_or] <= 0]
            ev = wr_or*np.mean(wins)+(1-wr_or)*np.mean(losses) if len(wins)>0 and len(losses)>0 else 0
            print(f"  EURUSD+EURJPY->GBPJPY (OR):  {wr_or:.0%} {n_or:7d} {avg_or:>7.2f} {ev:>+7.2f}")
        
        # EURUSD -> GBPUSD (direct USD-driven)
        gu_close = data.get("GBPUSD", {}).get("close")
        if gu_close is not None:
            gu_ret = np.diff(gu_close)
            bgu = rolling_beta(eu_ret, gu_ret, 10)
            dgu_eu = np.zeros(len(eu_ret))
            for i in range(10, len(eu_ret)):
                dgu_eu[i] = (bgu[i] * eu_ret[i-1] - gu_ret[i-1]) * 10000
            catch_gu = np.zeros(len(eu_ret))
            for i in range(10, len(eu_ret)-10):
                catch_gu[i] = np.sum(gu_ret[i:i+10]) * 10000
            sd3 = np.std(dgu_eu[10:])
            s3 = dgu_eu > zt * sd3
            s_gu_gj = s3 & s2  # EURUSD->GBPUSD AND EURJPY->GBPJPY
            idx_triple = np.where(s_gu_gj & (catch_gj != 0))[0]
            if len(idx_triple) >= 5:
                wr_t = np.mean(catch_gj[idx_triple] > 0)
                avg_t = np.mean(catch_gj[idx_triple])
                print(f"  EURUSD->GBPUSD + EURJPY->GBPJPY: {wr_t:.0%} {len(idx_triple):7d} {avg_t:>7.2f}")

# D. NEGATIVE DEFICITS (follower OVERSHOT)
print("\n" + "-" * 70)
print("D. NEGATIVE DEFICITS (follower OVERSHOT → revert DOWN)")
print("-" * 70)
print(f"{'Pair':<22} {'pos WR':>7} {'pos Tr':>7} {'pos EV':>7} | {'neg WR':>7} {'neg Tr':>7} {'neg EV':>7}")
print("-" * 70)
for leader, follower, lb, hold in configs:
    if leader not in data or follower not in data:
        continue
    deficits, catchups, _ = analyze_pair(leader, follower, data, lb, hold)
    std = np.std(deficits[lb:])
    if std == 0:
        continue
    
    sig_pos = deficits > 2.0 * std
    sig_neg = deficits < -2.0 * std
    
    for sig, label in [(sig_pos, "pos"), (sig_neg, "neg")]:
        idx = np.where(sig & (catchups != 0))[0]
        n = len(idx)
        if n < 5:
            continue
        wr = np.mean(catchups[idx] > 0)
        wins = catchups[idx][catchups[idx] > 0]
        losses = catchups[idx][catchups[idx] <= 0]
        ev = wr*np.mean(wins)+(1-wr)*np.mean(losses) if len(wins)>0 and len(losses)>0 else 0
        
        if label == "pos":
            pos_wr, pos_n, pos_ev = wr, n, ev
        else:
            neg_wr, neg_n, neg_ev = wr, n, ev
    
    print(f"{leader}->{follower:<14} {pos_wr:.0%} {pos_n:6d} {pos_ev:+6.2f} | {neg_wr:.0%} {neg_n:6d} {neg_ev:+6.2f}")

# E. EURUSD MAGNITUDE FILTER
print("\n" + "-" * 70)
print("E. EURUSD MOMENTUM MAGNITUDE FILTER")
print("-" * 70)
eu_ret2 = np.diff(data["EURUSD"]["close"]) * 10000  # pips
print(f"{'Filter':<20} {'WR':>6} {'Trades':>7} {'Avg(p)':>8} {'EV(p)':>8}")
print("-" * 50)
for q_name, lo, hi in [
    ("Q1 (quiet)", 0, 25),
    ("Q2 (mild)", 25, 50), 
    ("Q3 (active)", 50, 75),
    ("Q4 (volatile)", 75, 100),
]:
    q_lo = np.percentile(np.abs(eu_ret2[10:]), lo)
    q_hi = np.percentile(np.abs(eu_ret2[10:]), hi)
    q_mask = (np.abs(eu_ret2) >= q_lo) & (np.abs(eu_ret2) < q_hi)
    
    # Apply to EURJPY->GBPJPY
    deficits_ejgj, catchups_ejgj, _ = analyze_pair("EURJPY", "GBPJPY", data, 10, 20)
    std = np.std(deficits_ejgj[10:])
    sig = deficits_ejgj > 2.0 * std
    mask = sig & q_mask[:len(sig)] & (catchups_ejgj != 0)
    n = np.sum(mask)
    if n < 5:
        continue
    wr = np.mean(catchups_ejgj[mask] > 0)
    avg = np.mean(catchups_ejgj[mask])
    wins = catchups_ejgj[mask][catchups_ejgj[mask] > 0]
    losses = catchups_ejgj[mask][catchups_ejgj[mask] <= 0]
    ev = wr*np.mean(wins)+(1-wr)*np.mean(losses) if len(wins)>0 and len(losses)>0 else 0
    print(f"EUR volatility {q_name:<10} {wr:.0%} {n:7d} {avg:>7.2f} {ev:>+7.2f}")

# Also try EURUSD DIRECTION filter
print()
for direction, cond in [
    ("EURUSD UP", eu_ret2 > 0),
    ("EURUSD DOWN", eu_ret2 < 0),
    ("|EURUSD| > 5p", np.abs(eu_ret2) > 5),
    ("|EURUSD| < 3p", np.abs(eu_ret2) < 3),
]:
    mask = sig_all & cond[:len(sig_all)] & (catchups_ejgj != 0)
    n = np.sum(mask)
    if n < 5:
        continue
    wr = np.mean(catchups_ejgj[mask] > 0)
    avg = np.mean(catchups_ejgj[mask])
    wins = catchups_ejgj[mask][catchups_ejgj[mask] > 0]
    losses = catchups_ejgj[mask][catchups_ejgj[mask] <= 0]
    ev = wr*np.mean(wins)+(1-wr)*np.mean(losses) if len(wins)>0 and len(losses)>0 else 0
    print(f"  {direction:<20} {wr:.0%} {n:7d} {avg:>7.2f} {ev:>+7.2f}")
