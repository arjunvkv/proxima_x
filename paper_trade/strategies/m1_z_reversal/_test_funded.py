"""Test failed extreme with FundedNext-compatible holds (120s+)."""
import sys, time
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, r"C:\Trading\Agentic_Trading\proxima_x")
TICK_DIR = Path(r"C:\Trading\Agentic_Trading\proxima_x\data\exness_ticks")

def load(pair, months):
    dfs = []
    for y, m in months:
        p = TICK_DIR / f"{pair}_Raw_Spread_{y}_{m:02d}.zip"
        d = pd.read_csv(p, names=["E","S","Ts","B","A"], skiprows=1, header=None,
                        dtype={"Ts": str, "B": np.float64, "A": np.float64})
        d["Ts"] = pd.to_datetime(d["Ts"].str.replace("Z","",regex=False), format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")
        dfs.append(d.dropna(subset=["Ts"]))
    t = pd.concat(dfs, ignore_index=True).sort_values("Ts").reset_index(drop=True)
    t["ts_s"] = t["Ts"].astype(np.int64) // 10**9
    return t

def detect(ticks, pip_size, thresh_pips=5, thresh_sec=10):
    mid = (ticks["B"] + ticks["A"]).values / 2.0
    ts = ticks["ts_s"].values; n = len(ticks)
    raw = thresh_pips * pip_size; events = []; i = 0
    while i < n:
        end = min(int(np.searchsorted(ts, ts[i] + thresh_sec, side="right")), n)
        w = mid[i:end]
        if len(w) < 2: i += 1; continue
        hp = np.max(w) - w[0]; lp = w[0] - np.min(w)
        if max(hp, lp) >= raw:
            d2 = 1 if hp >= lp else -1
            ei = i + (np.argmax(w) if d2 == 1 else np.argmin(w))
            events.append({"time": ts[i], "extreme_idx": ei,
                          "impulse_pips": max(hp, lp) / pip_size,
                          "impulse_sec": ts[ei] - ts[i], "direction": d2,
                          "price_start": w[0], "price_extreme": mid[ei]})
            i = ei
        else: i += 1
    return pd.DataFrame(events)

def sim_funded(events, ticks, entry_delay=1, min_hold=120, max_hold=300,
               cost=0.00003, pip=0.0001, stop_pips=None):
    """FundedNext simulation: stop active from entry; no manual close before min_hold."""
    bids = ticks["B"].values; asks = ticks["A"].values; ts = ticks["ts_s"].values; n = len(ticks)

    trades = []
    for _, ev in events.iterrows():
        ed = -ev["direction"]          # +1=long, -1=short
        ei = int(ev["extreme_idx"]) + entry_delay
        if ei >= n: continue
        ep = asks[ei] if ed == 1 else bids[ei]
        et = ts[ei]
        min_t = et + min_hold          # earliest manual close
        max_t = et + max_hold          # latest close (fallback)

        stop_raw = stop_pips * pip if stop_pips is not None else None

        # Find tick range to scan: from entry to max_hold
        start_j = ei
        end_j = min(int(np.searchsorted(ts, max_t, side="right")), n)
        if end_j <= start_j + 1: continue

        exit_reason = None; exit_price = None

        for j in range(start_j, end_j):
            tick_t = ts[j]
            px = bids[j] if ed == 1 else asks[j]

            # Stop check (active from entry — broker can close early)
            if stop_raw is not None:
                if ed == 1 and px <= ep - stop_raw:
                    exit_price = px; exit_reason = "stop"; break
                if ed == -1 and px >= ep + stop_raw:
                    exit_price = px; exit_reason = "stop"; break

            # Manual close allowed after min_hold
            if tick_t >= min_t:
                exit_price = px; exit_reason = "market"; break

        if exit_price is None: continue
        pnl = (exit_price - ep) * ed - cost
        trades.append({"pnl_pips": pnl / pip, "exit_reason": exit_reason})
    return pd.DataFrame(trades)


print("=" * 70)
print("FUNDEDNEXT-COMPATIBLE TEST (min_hold=120-300s)")
print("=" * 70)

for pair, pip_sz, cost_r, label in [
    ("EURUSD", 0.0001, 0.00003, "EURUSD"),
    ("EURJPY", 0.01, 0.006, "EURJPY"),
]:
    print(f"\n{label} (Oct 2025):")
    t = load(pair, [(2025,10)])

    if pair == "EURUSD":
        configs = [(5, 10), (7, 10)]
    else:
        configs = [(7, 10), (10, 10)]

    for tp, ts_ in configs:
        ev = detect(t, pip_sz, tp, ts_)
        ne = len(ev)
        if ne < 10: continue

        print(f"\n  {tp}p/{ts_}s ({ne} events):")
        hdr = f"  {'Config':>22s} | {'n':>4s} {'WR':>5s} {'Gross(p)':>9s} {'Avg(p)':>7s} {'Stop%':>5s}"
        print(hdr); print("  " + "-" * len(hdr))

        # Baseline: no stop, close at min_hold
        for mh in [60, 120, 180, 300]:
            tr = sim_funded(ev, t, min_hold=mh, max_hold=mh+60, cost=cost_r, pip=pip_sz, stop_pips=None)
            if len(tr) == 0: continue
            p = tr["pnl_pips"]; n = len(p); w = int((p > 0).sum())
            print(f"  {'no_stop hold='+str(mh)+'s':>22s} | {n:>4d} {w/n*100:>5.1f}% {p.sum():>+9.1f}p {p.mean():>+7.2f}p {'0.0%':>5s}")

        # Stop at various levels, min_hold=120s
        for stop in [1.0, 2.0, 3.0, 5.0, 8.0]:
            tr = sim_funded(ev, t, min_hold=120, max_hold=300, cost=cost_r, pip=pip_sz, stop_pips=stop)
            if len(tr) == 0: continue
            p = tr["pnl_pips"]; n = len(p); w = int((p > 0).sum())
            pct_s = (tr["exit_reason"] == "stop").mean() * 100
            print(f"  {'stop='+str(stop)+'p hold=120s':>22s} | {n:>4d} {w/n*100:>5.1f}% {p.sum():>+9.1f}p {p.mean():>+7.2f}p {pct_s:>4.0f}%")

        # Stop at 3p with different holds
        for hold in [60, 180, 300]:
            tr = sim_funded(ev, t, min_hold=hold, max_hold=hold+60, cost=cost_r, pip=pip_sz, stop_pips=3.0)
            if len(tr) == 0: continue
            p = tr["pnl_pips"]; n = len(p); w = int((p > 0).sum())
            pct_s = (tr["exit_reason"] == "stop").mean() * 100
            print(f"  {'stop=3.0p hold='+str(hold)+'s':>22s} | {n:>4d} {w/n*100:>5.1f}% {p.sum():>+9.1f}p {p.mean():>+7.2f}p {pct_s:>4.0f}%")

# ── Monthly breakdown ──
print("\n" + "=" * 70)
print("MONTHLY BREAKDOWN (best configs)")
print("=" * 70)

if True:
    for pair, pip_sz, cost_r, tp, ts_, stop, hold, label in [
        ("EURUSD", 0.0001, 0.00003, 5, 10, 2.0, 120, "EURUSD 5p/10s stop=2p"),
        ("EURUSD", 0.0001, 0.00003, 7, 10, 3.0, 120, "EURUSD 7p/10s stop=3p"),
        ("EURJPY", 0.01, 0.006, 10, 10, 3.0, 120, "EURJPY 10p/10s stop=3p"),
    ]:
        print(f"\n{label}:")
        for months, mlabel in [([(2025,10)],"Oct"),([(2025,11)],"Nov"),([(2025,12)],"Dec")]:
            t_m = load(pair, months)
            ev = detect(t_m, pip_sz, tp, ts_)
            if len(ev) < 5: continue
            tr = sim_funded(ev, t_m, min_hold=hold, max_hold=300, cost=cost_r, pip=pip_sz, stop_pips=stop)
            if len(tr) == 0: continue
            p = tr["pnl_pips"]; n = len(p); w = int((p > 0).sum())
            pct_s = (tr["exit_reason"] == "stop").mean() * 100
            print(f"    {mlabel}: n={n:>3d} WR={w/n*100:.1f}% Gross={p.sum():+7.1f}p Avg={p.mean():+7.2f}p Stop={pct_s:.0f}%")

# ── Walk-forward with FundedNext configs ──
print("\n" + "=" * 70)
print("WALK-FORWARD (train Oct+Nov, test Dec)")
print("=" * 70)

for pair, pip_sz, cost_r, tp, ts_, stop, hold, label in [
    ("EURUSD", 0.0001, 0.00003, 5, 10, 2.0, 120, "EURUSD 5p/10s stop=2p"),
    ("EURUSD", 0.0001, 0.00003, 7, 10, 3.0, 120, "EURUSD 7p/10s stop=3p"),
    ("EURJPY", 0.01, 0.006, 10, 10, 3.0, 120, "EURJPY 10p/10s stop=3p"),
]:
    print(f"\n  {label}:")
    for months, mlabel in [([(2025,10),(2025,11)],"Train"),([(2025,12)],"Test")]:
        t_m = load(pair, months)
        ev = detect(t_m, pip_sz, tp, ts_)
        if len(ev) < 5: continue
        tr = sim_funded(ev, t_m, min_hold=hold, max_hold=300, cost=cost_r, pip=pip_sz, stop_pips=stop)
        if len(tr) == 0: continue
        p = tr["pnl_pips"]; n = len(p); w = int((p > 0).sum())
        print(f"    {mlabel}: n={n:>3d} WR={w/n*100:.1f}% Gross={p.sum():+7.1f}p Avg={p.mean():+7.2f}p")
