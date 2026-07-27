"""Full scan: all 3 pairs on quiet hours (00-07 UTC) — impulse fade."""
import sys, numpy as np, pandas as pd
from collections import deque
from pathlib import Path
sys.path.insert(0, r"C:\Trading\Agentic_Trading\proxima_x")
TICK_DIR = Path(r"C:\Trading\Agentic_Trading\proxima_x\data\exness_ticks")

PAIR_INFO = {
    "EURUSD": {"pip": 0.0001, "cost": 0.00003, "spread_norm": 0.00003},
    "EURJPY": {"pip": 0.01, "cost": 0.006, "spread_norm": 0.006},
    "GBPJPY": {"pip": 0.01, "cost": 0.007, "spread_norm": 0.007},
}

def load(pair, months):
    dfs = []
    for y, m in months:
        p = TICK_DIR / f"{pair}_Raw_Spread_{y}_{m:02d}.zip"
        d = pd.read_csv(p, names=["E","S","Ts","B","A"], skiprows=1, header=None,
                        dtype={"Ts":str,"B":np.float64,"A":np.float64})
        d["Ts"] = pd.to_datetime(d["Ts"].str.replace("Z","",regex=False),
                                 format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")
        dfs.append(d.dropna(subset=["Ts"]))
    return pd.concat(dfs, ignore_index=True).sort_values("Ts").reset_index(drop=True)

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
                if hp >= lp: ext_idx = Mq[0][1]; ddir = 1
                else: ext_idx = mq[0][1]; ddir = -1
                events.append((ws, ext_idx, ddir))
    return events

def sim(events, ticks, pip, cost, hold_s, stop_pips):
    bid = ticks["B"].values; ask = ticks["A"].values; ts = ticks["ts_s"].values; n = len(bid)
    pnls = []
    for ws_i, ext_i, ddir in events:
        ed = -ddir  # FADE
        ei2 = ext_i + 1
        if ei2 >= n - 1: continue
        ep = ask[ei2] if ed == 1 else bid[ei2]
        et = ts[ei2]
        he = int(np.searchsorted(ts, et + hold_s, side="right"))
        if he >= n: continue
        if stop_pips > 0:
            stop = ep - stop_pips * pip if ed == 1 else ep + stop_pips * pip
            stop_hit = False; si = he
            for j in range(ei2 + 1, he):
                if ed == 1:
                    if bid[j] <= stop: stop_hit = True; si = j; break
                else:
                    if ask[j] >= stop: stop_hit = True; si = j; break
            xp = stop if stop_hit else (bid[he] if ed == 1 else ask[he])
        else:
            xp = bid[he] if ed == 1 else ask[he]
        pnl = (xp - ep) * ed - cost
        pnls.append(pnl / pip)
    return np.array(pnls, dtype=np.float64)

print(f"{'='*90}")
print("QUIET HOURS SCAN (00-07 UTC) — All 3 pairs")
print("Testing impulse fade with various thresholds, windows, holds")
print(f"{'='*90}")

all_results = []
for pair, info in PAIR_INFO.items():
    pip = info["pip"]; cost = info["cost"]
    print(f"\n--- Loading {pair} ---")
    d = load(pair, [(2025,10),(2025,11),(2025,12)])
    d["ts_s"] = d["Ts"].astype(np.int64) // 10**9
    d["hour"] = d["Ts"].dt.hour
    
    # Quiet hours: 00-07 UTC
    quiet = d[d["hour"].between(0, 7)].reset_index(drop=True)
    total = d.copy()
    print(f"  Total: {len(d):,} ticks | Quiet hours: {len(quiet):,} ticks")
    
    thresholds = [3, 5, 7, 10]
    windows = [20, 60, 120]
    holds = [30, 60, 120]
    stops = [0, 5, 7]  # stop in pips (0=none)
    
    for tp in thresholds:
        for win in windows:
            for hold in holds:
                for stop in stops:
                    if stop > tp: continue  # stop shouldn't exceed threshold
                    thresh = tp * pip
                    ev = detect(quiet, thresh, win)
                    if len(ev) < 10: continue  # skip configs with <10 trades in 3mo
                    p = sim(ev, quiet, pip, cost, hold, stop * pip if stop > 0 else 0)
                    if len(p) == 0: continue
                    
                    wr = (p > 0).mean() * 100
                    avg = p.mean()
                    gross = p.sum()
                    tpd = len(p) / 78  # 78 trading days
                    
                    all_results.append({
                        "pair": pair, "tp": tp, "win": win, "hold": hold, "stop": stop,
                        "n": len(p), "tpd": tpd, "wr": wr, "avg": avg, "gross": gross
                    })

# Sort by WR descending, filter meaningful
df = pd.DataFrame(all_results)
df = df[df["n"] >= 20]  # min 20 trades

print(f"\n{'='*90}")
print("TOP 20 CONFIGS ON QUIET HOURS (sorted by WR, min 20 trades)")
print(f"{'='*90}")
top = df.sort_values("wr", ascending=False).head(20)
print(f"{'Pair':>8} | {'TP':>3} | {'Win':>4} | {'Hold':>4} | {'Stop':>4} | {'n':>6} | {'t/d':>5} | {'WR':>6} | {'Avg':>6} | {'Gross':>7}")
print("-"*75)
for _, r in top.iterrows():
    print(f"{r['pair']:>8} | {r['tp']:>3} | {r['win']:>4} | {r['hold']:>4} | {r['stop']:>4} | {r['n']:>6} | {r['tpd']:>5.1f} | {r['wr']:>5.1f}% | {r['avg']:>+6.2f}p | {r['gross']:>+7.1f}p")

# Best config per pair
print(f"\n{'='*90}")
print("BEST CONFIG PER PAIR (WR >= 60%, min 20 trades)")
print(f"{'='*90}")
for pair in ["EURUSD", "EURJPY", "GBPJPY"]:
    pdf = df[df["pair"] == pair].sort_values("wr", ascending=False)
    best = pdf[pdf["wr"] >= 60].head(5)
    if len(best) == 0:
        best = pdf.head(3)
    print(f"\n{pair}:")
    for _, r in best.iterrows():
        is_eurusd = pair == "EURUSD"
        pip_val = 10 if is_eurusd else 9.3
        dd_est = r["gross"] * pip_val * -0.3  # rough estimate
        print(f"  {r['tp']}p/{r['win']}s hold={r['hold']}s stop={r['stop']}p: "
              f"n={r['n']} t/d={r['tpd']:.1f} WR={r['wr']:.1f}% "
              f"avg={r['avg']:+.2f}p gross={r['gross']:+.1f}p "
              f"est_dd=${dd_est:.0f}")

# Monthly breakdown for best configs
print(f"\n{'='*90}")
print("MONTHLY BREAKDOWN — Top 3 configs")
print(f"{'='*90}")
for _, r in top.head(3).iterrows():
    pair = r["pair"]; pip = PAIR_INFO[pair]["pip"]; cost = PAIR_INFO[pair]["cost"]
    tp = r["tp"]; win_sec = r["win"]; hold_s = r["hold"]; stop_p = r["stop"]
    
    d = load(pair, [(2025,10),(2025,11),(2025,12)])
    d["ts_s"] = d["Ts"].astype(np.int64) // 10**9
    d["hour"] = d["Ts"].dt.hour
    quiet = d[d["hour"].between(0, 7)].reset_index(drop=True)
    quiet["month"] = quiet["Ts"].dt.month
    
    months_verbose = {10: "Oct", 11: "Nov", 12: "Dec"}
    print(f"\n{pair} {tp}p/{win_sec}s hold={hold_s}s stop={stop_p}p:")
    for m_num, m_label in months_verbose.items():
        m_data = quiet[quiet["month"] == m_num].reset_index(drop=True)
        ev = detect(m_data, tp * pip, win_sec)
        p = sim(ev, m_data, pip, cost, hold_s, stop_p * pip if stop_p > 0 else 0)
        if len(p) > 0:
            print(f"  {m_label}: n={len(p)} WR={(p>0).mean()*100:.1f}% avg={p.mean():+.2f}p gross={p.sum():+.1f}p")
        else:
            print(f"  {m_label}: 0 trades")

print("\nDone.")
