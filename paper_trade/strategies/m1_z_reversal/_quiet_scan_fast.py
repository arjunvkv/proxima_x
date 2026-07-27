"""Fast quiet hours scan — one pair at a time, fewer configs."""
import sys, numpy as np, pandas as pd
from collections import deque
from pathlib import Path
sys.path.insert(0, r"C:\Trading\Agentic_Trading\proxima_x")
TICK_DIR = Path(r"C:\Trading\Agentic_Trading\proxima_x\data\exness_ticks")

PAIR_INFO = {
    "EURJPY": {"pip": 0.01, "cost": 0.006},
    "GBPJPY": {"pip": 0.01, "cost": 0.007},
}

def load(pair):
    dfs = []
    for y, m in [(2025,10),(2025,11),(2025,12)]:
        p = TICK_DIR / f"{pair}_Raw_Spread_{y}_{m:02d}.zip"
        d = pd.read_csv(p, names=["E","S","Ts","B","A"], skiprows=1, header=None,
                        dtype={"Ts":str,"B":np.float64,"A":np.float64})
        d["Ts"] = pd.to_datetime(d["Ts"].str.replace("Z","",regex=False),
                                 format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")
        dfs.append(d.dropna(subset=["Ts"]))
    df = pd.concat(dfs, ignore_index=True).sort_values("Ts").reset_index(drop=True)
    df["ts_s"] = df["Ts"].astype(np.int64) // 10**9
    df["hour"] = df["Ts"].dt.hour
    return df

def detect(ticks, thresh, win_sec):
    mid = ((ticks["B"].values + ticks["A"].values) / 2.0).astype(np.float64)
    ts = ticks["ts_s"].values; n = len(mid)
    mq = deque(); Mq = deque(); ws = 0; events = []
    for i in range(n):
        v = float(mid[i])
        while mq and mq[-1][0] >= v: mq.pop()
        while Mq and Mq[-1][0] <= v: Mq.pop()
        mq.append((v, i)); Mq.append((v, i))
        while i > ws and ts[i] - ts[ws] > win_sec:
            if mq and mq[0][1] == ws: mq.popleft()
            if Mq and Mq[0][1] == ws: Mq.popleft()
            ws += 1
        if i > ws:
            wp = mid[ws]; hp = float(Mq[0][0] - wp); lp = float(wp - mq[0][0])
            if ts[i] - ts[ws] > win_sec: continue
            if hp >= thresh or lp >= thresh:
                if events and events[-1][0] >= ws: continue
                ext_idx = Mq[0][1] if hp >= lp else mq[0][1]
                events.append((ws, ext_idx, 1 if hp >= lp else -1))
    return events

def sim(events, ticks, pip, cost, hold_s, stop_p):
    bid = ticks["B"].values; ask = ticks["A"].values; ts = ticks["ts_s"].values; n = len(bid)
    pnls = []
    for ws_i, ext_i, ddir in events:
        ed = -ddir
        ei2 = ext_i + 1
        if ei2 >= n - 1: continue
        ep = ask[ei2] if ed == 1 else bid[ei2]
        et = ts[ei2]
        he = int(np.searchsorted(ts, et + hold_s, side="right"))
        if he >= n: continue
        if stop_p > 0:
            stop = ep - stop_p * pip if ed == 1 else ep + stop_p * pip
            si = he
            for j in range(ei2 + 1, he):
                if (ed == 1 and bid[j] <= stop) or (ed == -1 and ask[j] >= stop):
                    si = j; break
            xp = stop if si < he else (bid[he] if ed == 1 else ask[he])
        else:
            xp = bid[he] if ed == 1 else ask[he]
        pnl = (xp - ep) * ed - cost
        pnls.append(pnl / pip)
    return np.array(pnls, dtype=np.float64)

for pair in ["EURJPY", "GBPJPY"]:
    pip = PAIR_INFO[pair]["pip"]; cost = PAIR_INFO[pair]["cost"]
    print(f"\n=== LOADING {pair} ===")
    d = load(pair)
    quiet = d[d["hour"].between(0, 7)].reset_index(drop=True)
    print(f"  Total: {len(d):,} | Quiet: {len(quiet):,}")

    print(f"\n{pair} — QUIET HOURS (00-07 UTC) — ALL CONFIGS:")
    thresholds = [5, 7, 10, 15]
    windows = [20, 60, 120]
    holds = [30, 60, 120]
    best_per_win = {}
    
    for win in windows:
        for tp in thresholds:
            for hold in holds:
                ev = detect(quiet, tp * pip, win)
                if len(ev) < 20: continue
                p = sim(ev, quiet, pip, cost, hold, 0)
                if len(p) == 0: continue
                wr = (p > 0).mean() * 100; avg = p.mean(); gross = p.sum()
                tpd = len(p) / 78
                
                key = f"{tp}p/{win}s"
                if key not in best_per_win or wr > best_per_win[key][0]:
                    best_per_win[key] = (wr, avg, gross, len(p), tpd, hold, tp, win)
                
                if wr >= 55 and avg > 0:
                    print(f"  {tp}p/{win}s hold={hold}s: n={len(p):4d} t/d={tpd:.2f} WR={wr:.1f}% avg={avg:+.2f}p gross={gross:+.1f}p")
    
    print(f"\n{pair} — TOP CONFIGS BY THRESHOLD/WINDOW:")
    for k, v in sorted(best_per_win.items(), key=lambda x: x[1][0], reverse=True):
        wr, avg, gross, n, tpd, hold, tp, win = v
        if wr >= 55:
            print(f"  {k} hold={hold}s: n={n:4d} t/d={tpd:.2f} WR={wr:.1f}% avg={avg:+.2f}p gross={gross:+.1f}p")

print(f"\n{'='*70}")
print("MONTHLY STABILITY — BEST CONFIGS")
for pair in ["EURJPY", "GBPJPY"]:
    pip = PAIR_INFO[pair]["pip"]; cost = PAIR_INFO[pair]["cost"]
    d_full = load(pair)
    d_full["hour"] = d_full["Ts"].dt.hour
    d_full["month"] = d_full["Ts"].dt.month
    
    for tp in [7, 10]:
        for win in [60, 120]:
            for hold in [60, 120]:
                print(f"\n{pair} {tp}p/{win}s hold={hold}s:")
                total_trades = 0; total_pnl = 0; month_wrs = []
                for m_num, m_label in [(10,"Oct"),(11,"Nov"),(12,"Dec")]:
                    md = d_full[(d_full["month"]==m_num) & (d_full["hour"].between(0,7))].reset_index(drop=True)
                    if len(md) < 10000:
                        print(f"  {m_label}: insufficient data ({len(md)} ticks)")
                        continue
                    ev = detect(md, tp * pip, win)
                    p = sim(ev, md, pip, cost, hold, 0)
                    if len(p) == 0:
                        print(f"  {m_label}: 0 trades")
                        continue
                    wr = (p>0).mean()*100; avg = p.mean(); gross = p.sum()
                    total_trades += len(p); total_pnl += gross; month_wrs.append(wr)
                    print(f"  {m_label}: n={len(p):3d} WR={wr:.1f}% avg={avg:+.2f}p gross={gross:+.1f}p")
                if len(month_wrs) >= 2:
                    print(f"  TOTAL: n={total_trades} gross={total_pnl:+.1f}p | WR_min={min(month_wrs):.1f}% WR_max={max(month_wrs):.1f}%")
                    if min(month_wrs) < 45:
                        print(f"  ⚠️  MONTHLY INSTABILITY: {min(month_wrs):.1f}% — at least one losing month")

print("\nDone.")
