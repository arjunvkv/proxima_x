"""Test early loss cut on failed extreme signal."""
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

def sim_with_stop(events, ticks, hold=15, cost=0.00003, pip=0.0001, stop_pips=None, entry_delay=1):
    """Simulate fade with optional early loss cut.
    If stop_pips is set, scans tick-by-tick for stop hit.
    """
    bids = ticks["B"].values; asks = ticks["A"].values; ts = ticks["ts_s"].values; n = len(ticks)
    trades = []
    for _, ev in events.iterrows():
        ed = -ev["direction"]
        ei = int(ev["extreme_idx"]) + entry_delay
        if ei >= n: continue
        ep = asks[ei] if ed == 1 else bids[ei]
        et = ts[ei]
        exit_time = et + hold
        
        if stop_pips is not None:
            # Scan tick-by-tick for stop hit
            stop_raw = stop_pips * pip
            exit_price = None
            exit_reason = "hold"
            j = ei
            while j < n and ts[j] <= exit_time:
                price = bids[j] if ed == 1 else asks[j]
                if ed == 1:  # BUY: stop below entry
                    if price <= ep - stop_raw:
                        exit_price = price
                        exit_reason = "stop"
                        break
                else:  # SELL: stop above entry
                    if price >= ep + stop_raw:
                        exit_price = price
                        exit_reason = "stop"
                        break
                j += 1
            if exit_price is None:
                # Didn't hit stop, close at hold time
                xi = min(int(np.searchsorted(ts, exit_time, side="right")), n - 1)
                if xi <= ei: continue
                exit_price = bids[xi] if ed == 1 else asks[xi]
                exit_reason = "hold"
        else:
            # No stop: close at hold time
            xi = min(int(np.searchsorted(ts, exit_time, side="right")), n - 1)
            if xi <= ei: continue
            exit_price = bids[xi] if ed == 1 else asks[xi]
            exit_reason = "hold"
        
        pnl = (exit_price - ep) * ed - cost
        trades.append({"pnl_pips": pnl / pip, "pnl": pnl, "exit_reason": exit_reason})
    return pd.DataFrame(trades)


print("=" * 70)
print("EARLY LOSS CUT TEST — EURUSD (Oct only)")
print("=" * 70)

pair, pip_sz, cost_r = "EURUSD", 0.0001, 0.00003
t = load(pair, [(2025, 10)])
ev = detect(t, pip_sz)

for tp, ts_, hold in [(5, 10, 15), (7, 10, 15), (5, 10, 30)]:
    mask = (ev["impulse_pips"].values >= tp) & (ev["impulse_sec"].values <= ts_)
    if mask.sum() < 5: continue
    
    print(f"\n{pair} {tp}p/{ts_}s hold={hold}s (n={mask.sum()}):")
    print(f"  {'Stop(p)':>8s} | {'n':>4s} {'WR':>5s} {'Gross(p)':>9s} {'Avg(p)':>7s} | {'Stop%':>6s} {'Hold%':>6s} {'StopWR':>7s}")
    print(f"  {'-'*60}")
    
    for stop in [None, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
        tr = sim_with_stop(ev[mask], t, hold, cost_r, pip_sz, stop_pips=stop)
        if len(tr) == 0: continue
        p = tr["pnl_pips"].values
        n = len(p); w = int(np.sum(p > 0))
        wr = w / n * 100; gp = p.sum(); ap = p.mean()
        
        pct_stop = (tr["exit_reason"].values == "stop").mean() * 100
        pct_hold = (tr["exit_reason"].values == "hold").mean() * 100
        
        s_pnls = tr[tr["exit_reason"] == "stop"]["pnl_pips"].values
        h_pnls = tr[tr["exit_reason"] == "hold"]["pnl_pips"].values
        swr = np.sum(s_pnls > 0) / len(s_pnls) * 100 if len(s_pnls) > 0 else 0
        hwr = np.sum(h_pnls > 0) / len(h_pnls) * 100 if len(h_pnls) > 0 else 0
        
        stop_label = f"{stop}p" if stop is not None else "none"
        print(f"  {stop_label:>8s} | {n:>4d} {wr:>5.1f}% {gp:>+9.1f}p {ap:>+7.2f}p | "
              f"{pct_stop:>5.1f}% {pct_hold:>5.1f}% {swr:>6.1f}%")

print("\n" + "=" * 70)
print("EARLY LOSS CUT TEST — EURJPY (Oct only)")
print("=" * 70)

pair, pip_sz, cost_r = "EURJPY", 0.01, 0.006
t = load(pair, [(2025, 10)])
ev = detect(t, pip_sz)

for tp, ts_, hold in [(10, 10, 15), (10, 10, 30), (7, 10, 15)]:
    mask = (ev["impulse_pips"].values >= tp) & (ev["impulse_sec"].values <= ts_)
    if mask.sum() < 5: continue
    
    print(f"\n{pair} {tp}p/{ts_}s hold={hold}s (n={mask.sum()}):")
    print(f"  {'Stop(p)':>8s} | {'n':>4s} {'WR':>5s} {'Gross(p)':>9s} {'Avg(p)':>7s} | {'Stop%':>6s} {'Hold%':>6s} {'StopWR':>7s}")
    print(f"  {'-'*60}")
    
    for stop in [None, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0]:
        tr = sim_with_stop(ev[mask], t, hold, cost_r, pip_sz, stop_pips=stop)
        if len(tr) == 0: continue
        p = tr["pnl_pips"].values
        n = len(p); w = int(np.sum(p > 0))
        wr = w / n * 100; gp = p.sum(); ap = p.mean()
        
        pct_stop = (tr["exit_reason"].values == "stop").mean() * 100
        pct_hold = (tr["exit_reason"].values == "hold").mean() * 100
        
        s_pnls = tr[tr["exit_reason"] == "stop"]["pnl_pips"].values
        h_pnls = tr[tr["exit_reason"] == "hold"]["pnl_pips"].values
        swr = np.sum(s_pnls > 0) / len(s_pnls) * 100 if len(s_pnls) > 0 else 0
        
        stop_label = f"{stop}p" if stop is not None else "none"
        print(f"  {stop_label:>8s} | {n:>4d} {wr:>5.1f}% {gp:>+9.1f}p {ap:>+7.2f}p | "
              f"{pct_stop:>5.1f}% {pct_hold:>5.1f}% {swr:>6.1f}%")
