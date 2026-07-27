"""Adversarial validation for best quiet-hour configs.
Tests: walkforward, stops, direction bias, correlation, max DD.
"""
import sys, numpy as np, pandas as pd
from collections import deque
from pathlib import Path
sys.path.insert(0, r"C:\Trading\Agentic_Trading\proxima_x")
TICK_DIR = Path(r"C:\Trading\Agentic_Trading\proxima_x\data\exness_ticks")

PAIR_INFO = {
    "EURJPY": {"pip": 0.01, "cost": 0.006},
    "GBPJPY": {"pip": 0.01, "cost": 0.007},
}

def load(pair, months=None):
    if months is None: months = [(2025,10),(2025,11),(2025,12)]
    dfs = []
    for y, m in months:
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

def max_dd(pnl_array):
    cum = np.cumsum(pnl_array)
    running_max = np.maximum.accumulate(cum)
    dd = cum - running_max
    return dd.min()

# Best configs to test — focused on the best quiet-hour candidates
CONFIGS = [
    ("EURJPY", 10, 60, 30, 0, "10p/60s h=30s"),
    ("EURJPY", 7, 60, 30, 0, "7p/60s h=30s"),
]

print("="*80)
print("ADVERSARIAL VALIDATION — QUIET HOURS (00-07 UTC)")
print("="*80)

for pair, tp, win_s, hold_s, stop_p, label in CONFIGS:
    pip = PAIR_INFO[pair]["pip"]
    cost = PAIR_INFO[pair]["cost"]
    print(f"\n{'='*80}")
    print(f"{pair} | {label} (quiet hours only)")
    print(f"{'='*80}")
    
    d = load(pair)
    quiet = d[d["hour"].between(0, 7)].reset_index(drop=True)
    
    # --- 1. Walkforward: Oct+Nov train, Dec test ---
    quiet["month"] = quiet["Ts"].dt.month
    train = quiet[quiet["month"].isin([10, 11])].reset_index(drop=True)
    test = quiet[quiet["month"] == 12].reset_index(drop=True)
    
    for split_name, split_data in [("Train (Oct+Nov)", train), ("Test (Dec)", test)]:
        ev = detect(split_data, tp * pip, win_s)
        p = sim(ev, split_data, pip, cost, hold_s, stop_p * pip if stop_p > 0 else 0)
        if len(p) > 0:
            wr = (p>0).mean()*100; avg = p.mean(); gross = p.sum(); tpd = len(p) / (60 if "Train" in split_name else 22)
            print(f"  {split_name}: n={len(p):4d} t/d={tpd:.1f} WR={wr:.1f}% avg={avg:+.2f}p gross={gross:+.1f}p")
            if wr < 50: print(f"    ⚠️  FAIL: WR below 50% on {split_name}")
    
    # --- 2. Stop-loss sensitivity ---
    print(f"\n  --- Stop Loss Sensitivity ---")
    for s_p in [0, 3, 5, 7, 10, 15, 20, 30]:
        ev = detect(quiet, tp * pip, win_s)
        p = sim(ev, quiet, pip, cost, hold_s, s_p * pip if s_p > 0 else 0)
        wr = (p>0).mean()*100; avg = p.mean(); gross = p.sum()
        print(f"    stop={s_p}p: n={len(p):4d} WR={wr:.1f}% avg={avg:+.2f}p gross={gross:+.1f}p")
    
    # --- 3. Direction bias ---
    print(f"\n  --- Direction Bias ---")
    ev = detect(quiet, tp * pip, win_s)
    p = sim(ev, quiet, pip, cost, hold_s, stop_p * pip if stop_p > 0 else 0)
    bid = quiet["B"].values; ask = quiet["A"].values; ts = quiet["ts_s"].values; n = len(bid)
    long_pnls = []; short_pnls = []
    for ws_i, ext_i, ddir in ev:
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
        if ed == 1: long_pnls.append(pnl / pip)
        else: short_pnls.append(pnl / pip)
    
    if len(long_pnls) > 10:
        lp_arr = np.array(long_pnls)
        lwr = (lp_arr>0).mean()*100; lavg = lp_arr.mean()
        sp_arr = np.array(short_pnls)
        swr = (sp_arr>0).mean()*100; savg = sp_arr.mean()
        print(f"    LONG : n={len(lp_arr):4d} WR={lwr:.1f}% avg={lavg:+.2f}p")
        print(f"    SHORT: n={len(sp_arr):4d} WR={swr:.1f}% avg={savg:+.2f}p")
        if abs(lwr - swr) > 15:
            print(f"    ⚠️  DIRECTION BIAS: {abs(lwr-swr):.1f}% gap between long/short")
    
    # --- 4. Monthly breakdown ---
    print(f"\n  --- Monthly Breakdown ---")
    months_verbose = {10: "Oct", 11: "Nov", 12: "Dec"}
    m_wrs = []; m_pnls = []
    for m_num, m_label in months_verbose.items():
        md = quiet[quiet["month"] == m_num].reset_index(drop=True)
        ev = detect(md, tp * pip, win_s)
        p = sim(ev, md, pip, cost, hold_s, stop_p * pip if stop_p > 0 else 0)
        if len(p) > 0:
            wr = (p>0).mean()*100; avg = p.mean(); gross = p.sum(); tpd = len(p) / (22 if m_num == 12 else 21)
            m_wrs.append(wr); m_pnls.append(gross)
            print(f"    {m_label}: n={len(p):3d} t/d={tpd:.1f} WR={wr:.1f}% avg={avg:+.2f}p gross={gross:+.1f}p")
    
    if len(m_wrs) >= 2:
        print(f"    WR range: {min(m_wrs):.1f}% to {max(m_wrs):.1f}% | PnL range: {min(m_pnls):+.1f}p to {max(m_pnls):+.1f}p")
        if min(m_wrs) < 50:
            print(f"    ⚠️  MONTHLY INSTABILITY: at least one month below 50% WR")
    
    # --- 5. Max DD ---
    ev = detect(quiet, tp * pip, win_s)
    p = sim(ev, quiet, pip, cost, hold_s, stop_p * pip if stop_p > 0 else 0)
    dd = max_dd(p) if len(p) > 50 else 0
    pip_usd = 14.5 if pair == "EURJPY" else 13.5  # rough
    print(f"\n  Max DD: {dd:.0f}p ≈ ${dd * pip_usd:.0f}")
    
    # --- 6. Correlation check with EURUSD 5p/20s ---
    if pair == "EURJPY":
        print(f"\n  --- EURUSD 5p/20s Overlap Check (quiet hours) ---")
        eurusd = load("EURUSD")
        eurusd_q = eurusd[eurusd["hour"].between(0, 7)].reset_index(drop=True)
        eurusd_q["month"] = eurusd_q["Ts"].dt.month
        eu_ev = detect(eurusd_q, 0.0005, 20)
        print(f"    EURUSD 5p/20s events on quiet hours: {len(eu_ev)} in 3mo ({len(eu_ev)/78:.2f}/day)")
        print(f"    EURJPY coverage: when EURUSD fires ~{len(eu_ev)/78:.2f} trades/day")
        print(f"    EURJPY adds: {len(p)/78:.1f} trades/day on quiet hours")
        print(f"    Total combined trades/day (all hours): ~48 + {max(0, len(p)/78 - len(eu_ev)/78):.0f}")

print("\nDone.")
