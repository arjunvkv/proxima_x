"""Check the remaining untested failure modes from RESEARCH_PLAN_v2."""
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
    ts = ticks["ts_s"].values
    n = len(ticks)
    raw = thresh_pips * pip_size
    events = []
    i = 0
    while i < n:
        end = min(int(np.searchsorted(ts, ts[i] + thresh_sec, side="right")), n)
        w = mid[i:end]
        if len(w) < 2:
            i += 1; continue
        hp = np.max(w) - w[0]
        lp = w[0] - np.min(w)
        if max(hp, lp) >= raw:
            d2 = 1 if hp >= lp else -1
            ei = i + (np.argmax(w) if d2 == 1 else np.argmin(w))
            events.append({"time": ts[i], "extreme_idx": ei,
                          "impulse_pips": max(hp, lp) / pip_size,
                          "impulse_sec": ts[ei] - ts[i], "direction": d2,
                          "price_start": w[0], "price_extreme": mid[ei]})
            i = ei
        else:
            i += 1
    return pd.DataFrame(events)

def sim(events, ticks, hold, cost, pip, entry_delay=1):
    bids = ticks["B"].values
    asks = ticks["A"].values
    ts = ticks["ts_s"].values
    n = len(ticks)
    pnls = []
    for _, ev in events.iterrows():
        ed = -ev["direction"]
        ei = int(ev["extreme_idx"]) + entry_delay
        if ei >= n: continue
        ep = asks[ei] if ed == 1 else bids[ei]
        et = ts[ei]
        xi = min(int(np.searchsorted(ts, et + hold, side="right")), n - 1)
        if xi <= ei: continue
        xp = bids[xi] if ed == 1 else asks[xi]
        pnls.append(((xp - ep) * ed - cost) / pip)
    return np.array(pnls)

# ── Problem #14: Monthly consistency ──
print("=" * 70)
print("PROBLEM #14: MONTHLY CONSISTENCY")
print("=" * 70)

for pair, pip_sz, cost_r in [("EURUSD", 0.0001, 0.00003), ("EURJPY", 0.01, 0.006)]:
    print(f"\n{pair}:")
    for months, mlabel in [([(2025, 10)], "Oct"), ([(2025, 11)], "Nov"), ([(2025, 12)], "Dec")]:
        t = load(pair, months)
        ev = detect(t, pip_sz)
        configs = [(5, 10), (7, 10)] if pair == "EURUSD" else [(7, 10), (10, 10)]
        for tp, ts_ in configs:
            mask = (ev["impulse_pips"].values >= tp) & (ev["impulse_sec"].values <= ts_)
            if mask.sum() < 5: continue
            pnls = sim(ev[mask], t, 15, cost_r, pip_sz)
            n = len(pnls)
            w = int(np.sum(pnls > 0))
            print(f"  {mlabel} {tp}p/{ts_}s: n={n:>4d} WR={w/n*100:.1f}% Gross={pnls.sum():+.1f}p", flush=True)

# ── Problem #4: Entry price sensitivity ──
print("\n" + "=" * 70)
print("PROBLEM #4: ENTRY PRICE SENSITIVITY")
print("=" * 70)

for pair, pip_sz, cost_r in [("EURUSD", 0.0001, 0.00003), ("EURJPY", 0.01, 0.006)]:
    print(f"\n{pair} (Oct only):")
    t = load(pair, [(2025, 10)])
    ev = detect(t, pip_sz)
    tp = 7 if pair == "EURUSD" else 10
    mask = (ev["impulse_pips"].values >= tp) & (ev["impulse_sec"].values <= 10)
    if mask.sum() < 5: continue
    base = sim(ev[mask], t, 15, cost_r, pip_sz, 1)
    for delay in [1, 2, 5, 10]:
        pnls = sim(ev[mask], t, 15, cost_r, pip_sz, entry_delay=delay)
        n = len(pnls)
        w = int(np.sum(pnls > 0))
        gp = pnls.sum()
        chg = gp - base.sum()
        print(f"  delay={delay:>2d}tick: n={n} WR={w/n*100:.1f}% Gross={gp:+.1f}p (vs delay=1: {chg:+.1f}p)")

# ── Problem #13: Longer holds ──
print("\n" + "=" * 70)
print("PROBLEM #13: LONGER HOLDS")
print("=" * 70)

for pair, pip_sz, cost_r in [("EURUSD", 0.0001, 0.00003), ("EURJPY", 0.01, 0.006)]:
    print(f"\n{pair} (Oct):")
    t = load(pair, [(2025, 10)])
    ev = detect(t, pip_sz)
    tp = 7 if pair == "EURUSD" else 10
    mask = (ev["impulse_pips"].values >= tp) & (ev["impulse_sec"].values <= 10)
    if mask.sum() < 5: continue
    for hold in [5, 15, 30, 60, 120, 300]:
        pnls = sim(ev[mask], t, hold, cost_r, pip_sz)
        n = len(pnls)
        w = int(np.sum(pnls > 0))
        print(f"  hold={hold:>3d}s: n={n:>4d} WR={w/n*100:.1f}% Gross={pnls.sum():+.1f}p")

# ── Walk-forward: Oct+Nov train → Dec test ──
print("\n" + "=" * 70)
print("WALK-FORWARD: Train Oct+Nov, Test Dec")
print("=" * 70)

for pair, pip_sz, cost_r, tp, ts_ in [
    ("EURUSD", 0.0001, 0.00003, 5, 10),
    ("EURJPY", 0.01, 0.006, 10, 10),
]:
    print(f"\n{pair} {tp}p/{ts_}s hold=15s:")
    train_months = [(2025, 10), (2025, 11)]
    test_months = [(2025, 12)]
    
    tr = load(pair, train_months)
    ev_tr = detect(tr, pip_sz)
    mask_tr = (ev_tr["impulse_pips"].values >= tp) & (ev_tr["impulse_sec"].values <= ts_)
    if mask_tr.sum() >= 5:
        p_tr = sim(ev_tr[mask_tr], tr, 15, cost_r, pip_sz)
        print(f"  Train (Oct+Nov): n={len(p_tr)} WR={np.sum(p_tr>0)/len(p_tr)*100:.1f}% Gross={p_tr.sum():+.1f}p")
    
    te = load(pair, test_months)
    ev_te = detect(te, pip_sz)
    mask_te = (ev_te["impulse_pips"].values >= tp) & (ev_te["impulse_sec"].values <= ts_)
    if mask_te.sum() >= 5:
        p_te = sim(ev_te[mask_te], te, 15, cost_r, pip_sz)
        print(f"  Test (Dec):     n={len(p_te)} WR={np.sum(p_te>0)/len(p_te)*100:.1f}% Gross={p_te.sum():+.1f}p")
