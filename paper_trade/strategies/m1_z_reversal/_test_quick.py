"""Quick multi-pair failed extreme test — optimized single-pass detection."""
import sys, time, argparse
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, r"C:\Trading\Agentic_Trading\proxima_x")
TICK_DIR = Path(r"C:\Trading\Agentic_Trading\proxima_x\data\exness_ticks")

SPREAD_COST = {"EURUSD": 0.00003, "EURJPY": 0.006, "GBPJPY": 0.007}
PIP_SIZE = {"EURUSD": 0.0001, "EURJPY": 0.01, "GBPJPY": 0.01}
HOLD_TIMES = [5, 15, 30, 60]

def load_ticks(pair, months=None):
    if months is None: months = [(2025,10),(2025,11),(2025,12)]
    dfs = []
    for y, m in months:
        p = TICK_DIR / f"{pair}_Raw_Spread_{y}_{m:02d}.zip"
        if not p.exists(): continue
        d = pd.read_csv(p, names=["E","S","Ts","B","A"], skiprows=1, header=None,
                        dtype={"Ts":str,"B":np.float64,"A":np.float64})
        d["Ts"] = pd.to_datetime(d["Ts"].str.replace("Z","",regex=False), format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")
        dfs.append(d.dropna(subset=["Ts"]))
    if not dfs: raise FileNotFoundError(f"No data for {pair}")
    t = pd.concat(dfs, ignore_index=True).sort_values("Ts").reset_index(drop=True)
    t["ts_s"] = t["Ts"].astype(np.int64)//10**9
    return t

def find_impulses(ticks, impulse_pips=5, impulse_sec=10, pip_size=0.0001):
    mid = (ticks["B"].values + ticks["A"].values) / 2.0
    ts = ticks["ts_s"].values; n = len(ticks)
    raw_thresh = impulse_pips * pip_size
    events = []; i = 0
    while i < n:
        end = min(int(np.searchsorted(ts, ts[i] + impulse_sec, side="right")), n)
        window = mid[i:end]
        if len(window) < 2: i += 1; continue
        hp = np.max(window) - window[0]
        lp = window[0] - np.min(window)
        if max(hp, lp) >= raw_thresh:
            d = 1 if hp >= lp else -1
            ei = i + (np.argmax(window) if d == 1 else np.argmin(window))
            events.append({"time":ts[i],"extreme_time":ts[ei],
                          "impulse_pips":max(hp,lp)/pip_size,
                          "impulse_sec":ts[ei]-ts[i],"direction":d,
                          "price_start":window[0],"price_extreme":mid[ei],
                          "extreme_idx":ei})
            i = ei
        else: i += 1
    return pd.DataFrame(events)

def simulate_multi(events, ticks, hold_times=(5,15,30,60), cost=0.00003, pip_size=0.0001, retrace=None):
    """Simulate ALL hold times and both directions in ONE pass per event."""
    bids = ticks["B"].values; asks = ticks["A"].values; ts = ticks["ts_s"].values; n = len(ticks)
    trades = []
    for idx, (_, ev) in enumerate(events.iterrows()):
        ei = int(ev["extreme_idx"]) + 1
        if ei >= n: continue
        r = float(retrace[idx]) if retrace is not None and idx < len(retrace) else 0.0
        
        for flip in [False, True]:
            ed = ev["direction"] if flip else -ev["direction"]
            ep = asks[ei] if ed == 1 else bids[ei]
            et = ts[ei]
            dl = "MOM" if flip else "FADE"
            
            for hs in hold_times:
                xi = min(int(np.searchsorted(ts, et + hs, side="right")), n-1)
                if xi <= ei: continue
                xp = bids[xi] if ed == 1 else asks[xi]
                pnl = (xp - ep) * ed - cost
                trades.append({"entry_time":et,"entry_price":ep,"exit_price":xp,
                              "pnl_pips":pnl/pip_size,"pnl":pnl,"direction":ed,
                              "hold_sec":hs,"dir_label":dl,"retrace":r})
    return pd.DataFrame(trades)

def retrace_5s(events, ticks):
    mid = (ticks["B"]+ticks["A"]).values/2.0; ts = ticks["ts_s"].values; n = len(ticks)
    res = np.full(len(events), np.nan)
    for j, (_, ev) in enumerate(events.iterrows()):
        ei = int(ev["extreme_idx"]); ep = ev["price_extreme"]; sp = ev["price_start"]; d = int(ev["direction"])
        imp = abs(ep - sp)
        if imp == 0: continue
        i5 = int(np.searchsorted(ts, ts[ei]+5, side="right"))
        if i5 >= n: continue
        p5 = mid[i5]
        res[j] = ((ep - p5) / imp) if d == 1 else ((p5 - ep) / imp)
    return res

def monthly_label(t):
    t = pd.to_datetime(t, unit="s")
    return f"{t.year}-{t.month:02d}"

def test_pair(pair, months):
    pip = PIP_SIZE.get(pair, 0.0001)
    cost = SPREAD_COST.get(pair, 0.00003)
    cost_p = cost / pip
    ticks = load_ticks(pair, months)
    print(f"\n{pair}: {len(ticks):,} ticks | cost={cost_p:.1f}p", flush=True)

    all_data = []
    configs = [(5,10),(7,10),(10,10),(15,15)] if pair != "EURUSD" else [(5,10),(7,10),(10,10)]
    
    for ip, isec in configs:
        t0 = time.time()
        ev = find_impulses(ticks, ip, isec, pip)
        dt = time.time() - t0
        if len(ev) == 0:
            print(f"  {ip}p/{isec}s: 0 events ({dt:.1f}s)", flush=True); continue
        ret = retrace_5s(ev, ticks)
        print(f"  {ip}p/{isec}s: {len(ev)} events ({dt:.1f}s)", flush=True)
        
        # Batch simulate all holds + directions (retrace passed in for gate)
        tr = simulate_multi(ev, ticks, HOLD_TIMES, cost, pip, retrace=ret)
        if len(tr) == 0: continue
        
        for (hold, dl), grp in tr.groupby(["hold_sec","dir_label"]):
            p = grp["pnl_pips"].values
            n, w = len(p), int(np.sum(p > 0))
            wr = w/n*100 if n > 0 else 0
            gp = float(p.sum()); ap = float(p.mean())
            
            g = grp[grp["retrace"] >= 0.1]["pnl_pips"].values
            gn, gw = len(g), int(np.sum(g > 0))
            gwr = gw/gn*100 if gn > 0 else 0
            ggp = float(g.sum()) if gn > 0 else 0
            
            all_data.append({"pair":pair,"thresh":f"{ip}p/{isec}s","hold":hold,"dir":dl,
                            "n":n,"wr":wr,"gross_p":gp,"avg_p":ap,
                            "gn":gn,"gwr":gwr,"ggross_p":ggp})
            
            if dl == "FADE" and hold in [5,15,30]:
                grp["month"] = grp["entry_time"].apply(monthly_label)
                for m, g2 in grp.groupby("month"):
                    mp = g2["pnl_pips"].values
                    mw = int(np.sum(mp > 0))
                    all_data.append({"pair":pair,"thresh":f"{ip}p/{isec}s","hold":hold,"dir":"FADE",
                                    "month":m,"n":len(mp),"wr":mw/len(mp)*100,
                                    "gross_p":float(mp.sum()),"avg_p":float(mp.mean())})
    
    return all_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pairs", nargs="*", default=["EURUSD","EURJPY","GBPJPY"])
    parser.add_argument("--months", default="10,11,12")
    args = parser.parse_args()
    months = [(2025,int(m)) for m in args.months.split(",")]
    
    all_data = []
    for pair in args.pairs:
        t0 = time.time()
        all_data.extend(test_pair(pair, months))
        print(f"  Done in {time.time()-t0:.0f}s", flush=True)
    
    # Print summary
    df = pd.DataFrame(all_data)
    print(f"\n{'='*70}\nSUMMARY\n{'='*70}", flush=True)
    
    # Best per pair and config
    fade = df[(df["dir"]=="FADE") & (df.get("month",None).isna() if "month" in df.columns else True)]
    for pair in args.pairs:
        sub = df[(df["pair"]==pair)&(df["dir"]=="FADE")&(df["month"].isna())].copy()
        if len(sub)==0: continue
        best = sub.loc[sub["gross_p"].idxmax()]
        print(f"\n{pair} BEST: {best['thresh']} hold={best['hold']}s "
              f"n={best['n']:.0f} WR={best['wr']:.1f}% Gross={best['gross_p']:+.1f}p")
        
        # Compare all configs
        print(f"  All fade configs:")
        for _, r in sub.sort_values("gross_p", ascending=False).iterrows():
            gstr = f"→ gated:{r['gwr']:.1f}%/{r['ggross_p']:+.1f}p" if r['gn'] > 0 else ""
            print(f"    {r['thresh']:>10s} hold={r['hold']:.0f}s "
                  f"n={r['n']:.0f} WR={r['wr']:.1f}% Gross={r['gross_p']:+.1f}p{gstr}")
        
        # Direction check
        for _, row in sub.iterrows():
            mom = df[(df["pair"]==pair)&(df["dir"]=="MOM")&(df["thresh"]==row["thresh"])&(df["hold"]==row["hold"])]
            if len(mom)>0:
                mr = mom.iloc[0]
                gap = row["wr"] - mr["wr"]
                if row["hold"] in [15,30]:
                    print(f"    {row['thresh']} hold={row['hold']}s: FADE WR={row['wr']:.1f}% "
                          f"MOM WR={mr['wr']:.1f}% gap={gap:+.1f}pp")
    
    # Monthly breakdown (best config per pair)
    print(f"\nMonthly:", flush=True)
    for pair in args.pairs:
        sub = df[(df["pair"]==pair)&(df["dir"]=="FADE")&(df["month"].notna())].copy()
        if len(sub)==0: continue
        best_conf = sub.loc[sub["gross_p"].idxmax()]
        print(f"  {pair} {best_conf['thresh']} hold={best_conf['hold']}s: "
              f"{best_conf['month']} n={best_conf['n']:.0f} WR={best_conf['wr']:.1f}% {best_conf['gross_p']:+.1f}p")
    
    # Failure mode checks
    print(f"\nFailure modes:", flush=True)
    for pair in args.pairs:
        sub = df[(df["pair"]==pair)&(df["dir"]=="FADE")&(df["month"].isna())]
        if len(sub)==0: continue
        best = sub.loc[sub["gross_p"].idxmax()]
        
        # Sub-spread check
        cost_p = SPREAD_COST.get(pair,0.00003)/PIP_SIZE.get(pair,0.0001)
        print(f"  {pair}: best avg={best['avg_p']:+.2f}p vs cost={cost_p:.1f}p "
              f"→ {'SURVIVES' if best['avg_p'] > cost_p else 'SUB-SPREAD'}")
        
        # Frequency
        print(f"  {pair}: {best['thresh']} → ~{best['n']/len(months):.0f}/month")
        
        # Direction edge
        mom_wr = df[(df["pair"]==pair)&(df["dir"]=="MOM")&(df["thresh"]==best["thresh"])&(df["hold"]==best["hold"])]
        if len(mom_wr)>0:
            print(f"  {pair}: FADE WR={best['wr']:.1f}% vs MOM WR={mom_wr.iloc[0]['wr']:.1f}% "
                  f"→ {'FADE EDGE' if best['wr'] > mom_wr.iloc[0]['wr']+5 else 'NO EDGE'}")
    
    pg = df[df["dir"]=="FADE"].groupby("pair").agg({"n":"sum","gross_p":"sum","wr":"mean"})
    print(f"\nTotal trades per pair:", flush=True)
    for pair in args.pairs:
        row = pg.loc[pair] if pair in pg.index else None
        if row is not None:
            print(f"  {pair}: {row['n']:.0f}t sum_gross={row['gross_p']:+.1f}p")
