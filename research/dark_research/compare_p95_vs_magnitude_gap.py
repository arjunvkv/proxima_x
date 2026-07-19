#!/usr/bin/env python3
"""Side-by-side: P95 Consensus vs Live Magnitude Gap on identical Dukascopy data."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, os, calendar

DATA = r"C:\Trading\Agentic_Trading\proxima_x\research\dark_research\dukascopy_data"
HALF_SPREAD_PIPS = np.array([0.5, 0.3, 0.7])
ECN_COMM, LOT, MAG95 = 7, 100000, 0.00018741
PIPS = np.array([0.01, 0.0001, 0.01])
PAIR_NAMES = ["EURJPY", "EURUSD", "GBPJPY"]

def pip_val(p, usdjpy):
    return 10.0 if p == 1 else 1000.0 / usdjpy

def load_all():
    frames = {}
    for p, pn in [("eurjpy","EURJPY"),("eurusd","EURUSD"),("gbpjpy","GBPJPY")]:
        dfs = []
        for y in [2024, 2026]:
            for m in range(1, 13):
                if (y==2024 and m<10) or (y==2026 and m>6): continue
                ld = calendar.monthrange(y, m)[1]
                f = os.path.join(DATA, f"{p}-m1-bid-{y}-{m:02d}-01-{y}-{m:02d}-{ld}.csv")
                if not os.path.exists(f): continue
                df = pd.read_csv(f)
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                dfs.append(df)
        frames[pn] = pd.concat(dfs).sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    common = sorted(set(frames["EURJPY"]["timestamp"]) & set(frames["EURUSD"]["timestamp"]) & set(frames["GBPJPY"]["timestamp"]))
    tmap = {p: {t: i for i, t in enumerate(frames[p]["timestamp"])} for p in frames}
    close = np.column_stack([frames[p]["close"].values[[tmap[p][t] for t in common]] for p in ["EURJPY","EURUSD","GBPJPY"]])
    opens = np.column_stack([frames[p]["open"].values[[tmap[p][t] for t in common]] for p in ["EURJPY","EURUSD","GBPJPY"]])
    high = np.column_stack([frames[p]["high"].values[[tmap[p][t] for t in common]] for p in ["EURJPY","EURUSD","GBPJPY"]])
    low = np.column_stack([frames[p]["low"].values[[tmap[p][t] for t in common]] for p in ["EURJPY","EURUSD","GBPJPY"]])
    times = np.array([int(t.timestamp()) for t in common], dtype=np.int64)
    return close, opens, high, low, times

close, opens, high, low, times = load_all()
T = close.shape[0]
rets = np.diff(np.log(close), axis=0)
up = rets > 0
consensus = up.all(axis=1) | (~up).all(axis=1)
direction = np.where(up.all(axis=1), 1.0, -1.0)
avg_mag = np.mean(np.abs(rets), axis=1)
pair_mags = np.abs(rets)
dt_all = pd.to_datetime(times, unit="s")
hour_arr = dt_all.hour.values[1:]
usdjpy_proxy = close[:,0] / close[:,1]

# ============================================================
# P95 CONSENSUS BACKTEST
# ============================================================
def run_config(mag_pct, hold, session_start, session_end, exec_type="best_pair", cost_slip=0.5, cost_spread=1.5):
    mag_thresh = np.percentile(avg_mag, mag_pct) if mag_pct < 100 else MAG95
    te_idx = np.where(consensus & (hour_arr >= session_start) & (hour_arr <= session_end) & (avg_mag > mag_thresh))[0]
    te_idx = te_idx[te_idx + hold < T - 1]
    if len(te_idx) < 5: return 0, 0, 0, 0, 0, np.array([]), [], []
    bi = np.argmax(pair_mags[te_idx], axis=1)
    avg_usdjpy = np.mean(usdjpy_proxy[te_idx])
    pnls = []
    for j,i in enumerate(te_idx):
        p = bi[j]
        gross = np.log(close[i+hold,p]/close[i,p])*direction[te_idx[j]]
        spread = HALF_SPREAD_PIPS[p]*2*1.5*pip_val(p, avg_usdjpy)
        slp = cost_slip*2*pip_val(p, avg_usdjpy)
        gusd = LOT*gross if p==1 else LOT*gross*close[i,p]/usdjpy_proxy[i]
        pnls.append(gusd - spread - slp - ECN_COMM)
    pnls = np.array(pnls)
    n = len(pnls); wr = np.mean(pnls>0)*100
    sh = np.mean(pnls)/(np.std(pnls)+1e-10)*np.sqrt(1440/hold)
    return n, wr, sh, np.mean(pnls), np.sum(pnls), pnls, te_idx, bi

def magnitude_gap_backtest(hold=3, session_start=1, session_end=23,
                           min_opportunity=0.30, window_size=50,
                           cost_slip=0.5, cost_spread=1.5):
    """Replicate the live strategy.py generate_signal logic exactly."""
    # Rolling price history per pair (deque, maxlen=60 as in strategy.py)
    _history_size = 60
    _price_history = {p: [] for p in PAIR_NAMES}

    def _returns(arr):
        if len(arr) < 2:
            return np.array([])
        return np.diff(arr) / arr[:-1]

    def _magnitude_gap(ret, cross_ret):
        m = np.mean(np.abs(ret)) + 1e-10
        cm = np.mean(np.abs(cross_ret)) + 1e-10
        return (m - cm) / cm

    avg_usdjpy = np.mean(usdjpy_proxy)
    pnls = []
    t_idx = []
    bi_list = []

    for t in range(1, T):
        h = dt_all[t].hour
        if h < session_start or h > session_end:
            continue

        # Update rolling prices
        for pi, pn in enumerate(PAIR_NAMES):
            _price_history[pn].append(close[t, pi])
            if len(_price_history[pn]) > _history_size:
                _price_history[pn].pop(0)

        # Need at least 2 prices to compute returns
        if any(len(_price_history[p]) < 2 for p in PAIR_NAMES):
            continue

        # Compute returns
        rets_dict = {}
        for pn in PAIR_NAMES:
            arr = np.array(_price_history[pn])
            r = np.diff(arr) / arr[:-1]
            rets_dict[pn] = r[-1] if len(r) > 0 else 0.0

        # Cross-pair average magnitude
        cross_mag = np.mean([abs(v) for v in rets_dict.values()])

        # Score each pair
        scored = []
        for pn in PAIR_NAMES:
            r = rets_dict[pn]
            all_rets = np.array(list(rets_dict.values()))
            mg = (np.mean(np.abs([r])) + 1e-10 - (np.mean(np.abs(all_rets)) + 1e-10)) / (np.mean(np.abs(all_rets)) + 1e-10)
            sign = np.sign(r)
            if sign == 0:
                continue
            confirmers = sum(1 for v in rets_dict.values() if np.sign(v) == sign)
            opportunity = abs(mg) * (confirmers / max(len(rets_dict), 1))
            scored.append((opportunity, pn, sign, mg, confirmers))

        if not scored:
            continue

        scored.sort(key=lambda x: x[0], reverse=True)
        best = scored[0]
        opp, pair, sign, mg, conf = best

        if opp < 0.30:  # min_opportunity
            continue

        # Execute trade
        pi = PAIR_NAMES.index(pair)
        i = t  # entry at close[t]
        if i + 3 >= T:
            continue

        gross = np.log(close[i+3, pi] / close[i, pi]) * sign
        avg_usdjpy = np.mean(usdjpy_proxy)
        spread = HALF_SPREAD_PIPS[pi]*2*1.5*pip_val(pi, avg_usdjpy)
        slp = cost_slip*2*pip_val(pi, avg_usdjpy)
        gusd = LOT*gross if pi==1 else LOT*gross*close[i,pi]/usdjpy_proxy[i]
        pnl = gusd - spread - slp - ECN_COMM
        pnls.append(pnl)
        t_idx.append(i)
        bi_list.append(pi)

    pnls = np.array(pnls)
    n = len(pnls)
    if n < 5:
        return 0, 0, 0, 0, 0, np.array([]), [], []
    wr = np.mean(pnls>0)*100
    sh = np.mean(pnls)/(np.std(pnls)+1e-10)*np.sqrt(1440/hold)
    return n, wr, sh, np.mean(pnls), np.sum(pnls), pnls, t_idx, bi_list

# ============================================================
# SIDE-BY-SIDE COMPARISON
# ============================================================
print("\n" + "=" * 90)
print("SIDE-BY-SIDE: P95 CONSENSUS vs LIVE MAGNITUDE GAP")
print("=" * 90)

configs = [
    ("P95 Consensus (3p, H07-H21, best_pair)", lambda: run_config(95, 3, 7, 21)),
    ("Mag Gap Live (3p, H01-H23, opp≥0.30)", lambda: magnitude_gap_backtest(3, 1, 23)),
]

results = {}
for label, fn in configs:
    n, wr, sh, avg, tot, pnls, t_idx, bi = fn()
    results[label] = {
        "n": n, "wr": wr, "sharpe": sh, "avg": avg, "tot": tot,
        "pnls": pnls, "t_idx": t_idx, "bi": bi
    }

print("\n" + "=" * 90)
print("SIDE-BY-SIDE: P95 CONSENSUS vs LIVE MAGNITUDE GAP")
print("=" * 90)
print(f"{'Metric':<30s} {'P95 Consensus':>18s} {'Mag Gap Live':>18s}")
print(f"{'─'*70}")
for metric in ["n", "wr", "sharpe", "avg", "tot"]:
    v1 = results[configs[0][0]][metric]
    v2 = results[configs[1][0]][metric]
    if metric == "n":
        print(f"{'Trades':<30s} {v1:>18,d} {v2:>18,d}")
    elif metric == "wr":
        print(f"{'Win Rate':<30s} {v1:>17.1f}% {v2:>17.1f}%")
    elif metric == "sharpe":
        print(f"{'Sharpe':<30s} {v1:>18.2f} {v2:>18.2f}")
    elif metric == "avg":
        print(f"{'Avg PnL':<30s} {v1:>17.2f} {v2:>17.2f}")
    elif metric == "tot":
        print(f"{'Total PnL':<30s} {v1:>17,.0f} {v2:>17,.0f}")

# Monthly breakdown
print(f"\n{'Monthly Comparison':<30s} {'P95 Consensus':>18s} {'Mag Gap Live':>18s}")
print(f"{'-'*70}")
ym_set = sorted(set((dt_all[ti].year, dt_all[ti].month) for ti in results["P95 Consensus (3p, H07-H21, best_pair)"]["t_idx"]))
for y, m in ym_set:
    p95_pnls = []
    mg_pnls = []
    for i, ti in enumerate(results["P95 Consensus (3p, H07-H21, best_pair)"]["t_idx"]):
        if dt_all[ti].year == y and dt_all[ti].month == m:
            p95_pnls.append(results["P95 Consensus (3p, H07-H21, best_pair)"]["pnls"][i])
    for i, ti in enumerate(results["Mag Gap Live (3p, H01-H23, opp≥0.30)"]["t_idx"]):
        if dt_all[ti].year == y and dt_all[ti].month == m:
            mg_pnls.append(results["Mag Gap Live (3p, H01-H23, opp≥0.30)"]["pnls"][i])
    p95_arr = np.array(p95_pnls)
    mg_arr = np.array(mg_pnls)
    p95_s = np.mean(p95_arr)/(np.std(p95_arr)+1e-10)*np.sqrt(1440/3) if len(p95_arr) > 5 else 0
    mg_s = np.mean(mg_arr)/(np.std(mg_arr)+1e-10)*np.sqrt(1440/3) if len(mg_arr) > 5 else 0
    print(f"  {pd.to_datetime(f'{y}-{m:02d}-01').strftime('%b %Y'):>8s}: P95 n={len(p95_arr):4d} S={p95_s:.2f} ${np.sum(p95_arr):>8,.0f}  |  MG n={len(mg_arr):4d} S={mg_s:.2f} ${np.sum(mg_arr):>8,.0f}")

print(f"\n{'='*90}")
print("SIDE-BY-SIDE SUMMARY")
print("=" * 90)
print(f"{'Metric':<30s} {'P95 Consensus':>18s} {'Mag Gap Live':>18s}")
print(f"{'-'*70}")
for label, fn in configs:
    r = results[label]
    print(f"\n{label}:")
    print(f"  Trades:       {r['n']:>8,d}")
    print(f"  Win Rate:     {r['wr']:>7.1f}%")
    print(f"  Sharpe:       {r['sharpe']:>8.2f}")
    print(f"  Avg PnL:      ${r['avg']:>7.2f}")
    print(f"  Total PnL:    ${r['tot']:>9,.0f}")
    print(f"  Trades/day:   {r['n']/(T/1440):>7.1f}")
