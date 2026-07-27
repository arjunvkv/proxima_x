"""Events per day/hour stats."""
import sys, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, r"C:\Trading\Agentic_Trading\proxima_x")
TICK_DIR = Path(r"C:\Trading\Agentic_Trading\proxima_x\data\exness_ticks")

def load(pair, months):
    dfs = []
    for y, m in months:
        p = TICK_DIR / f"{pair}_Raw_Spread_{y}_{m:02d}.zip"
        d = pd.read_csv(p, names=["E","S","Ts","B","A"], skiprows=1, header=None,
                        dtype={"Ts":str,"B":np.float64,"A":np.float64})
        d["Ts"] = pd.to_datetime(d["Ts"].str.replace("Z","",regex=False), format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")
        dfs.append(d.dropna(subset=["Ts"]))
    t = pd.concat(dfs, ignore_index=True).sort_values("Ts").reset_index(drop=True)
    t["ts_s"] = t["Ts"].astype(np.int64) // 10**9
    return t

def detect(ticks, pip_size, thresh_pips=5, thresh_sec=10):
    mid = (ticks["B"] + ticks["A"]).values / 2.0
    ts = ticks["ts_s"].values; n = len(ticks)
    raw = thresh_pips * pip_size; evs = []; i = 0
    skip = max(1, int(n / max(ts[-1]-ts[0], 1)) * 3)
    while i < n:
        end = min(int(np.searchsorted(ts, ts[i] + thresh_sec, side="right")), n)
        if end - i < 2: i += skip; continue
        w = mid[i:end]
        hp = np.max(w) - w[0]; lp = w[0] - np.min(w)
        if max(hp, lp) >= raw:
            ei = i + (np.argmax(w) if hp >= lp else np.argmin(w))
            evs.append({"Ts": ticks["Ts"].iloc[i]})
            i = max(ei, i + 1)
        else: i += skip
    return pd.DataFrame(evs)

pairs_imp = [("EURUSD", 7), ("EURUSD", 5), ("EURJPY", 10)]

for pair, imp in pairs_imp:
    pip = 0.0001 if pair == "EURUSD" else 0.01
    t = load(pair, [(2025,10),(2025,11),(2025,12)])
    ev = detect(t, pip, imp, 10)
    
    ev["hour"] = ev["Ts"].dt.hour
    ev["date"] = ev["Ts"].dt.date
    
    n_total = len(ev)
    trading_days = ev["date"].nunique()
    per_day = n_total / trading_days
    
    print(f"\n{pair} {imp}p/10s:")
    print(f"  Total events: {n_total}")
    print(f"  Trading days: {trading_days}")
    print(f"  Events/day:   {per_day:.1f}")
    print(f"  Hourly distribution:")
    for h in range(24):
        count = (ev["hour"] == h).sum()
        if count > 0:
            bar = "█" * int(count / max(1, n_total / 60))
            print(f"    UTC {h:02d}: {count:>4d} ({count/n_total*100:>4.1f}%) {bar}")
