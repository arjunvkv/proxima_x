"""Comprehensive multi-pair failed extreme test — validates all failure modes from RESEARCH_PLAN_v2."""
import sys, time, argparse
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, r"C:\Trading\Agentic_Trading\proxima_x")
TICK_DIR = Path(r"C:\Trading\Agentic_Trading\proxima_x\data\exness_ticks")

SPREAD_COST = {"EURUSD": 0.00003, "EURJPY": 0.00006, "GBPJPY": 0.00007}  # 0.3p, 0.6p, 0.7p
PIP_SIZE = {"EURUSD": 0.0001, "EURJPY": 0.01, "GBPJPY": 0.01}
HOLD_TIMES = [5, 15, 30, 60]
IMPULSE_CONFIGS = [(3, 5), (5, 10), (7, 10), (10, 10), (15, 10)]


def load_ticks(pair, months=None):
    if months is None:
        months = [(2025, 10), (2025, 11), (2025, 12)]
    dfs = []
    for y, m in months:
        p = TICK_DIR / f"{pair}_Raw_Spread_{y}_{m:02d}.zip"
        if not p.exists():
            continue
        d = pd.read_csv(p, names=["E","S","Ts","B","A"], skiprows=1, header=None,
                        dtype={"Ts": str, "B": np.float64, "A": np.float64})
        d["Ts"] = pd.to_datetime(d["Ts"].str.replace("Z","",regex=False),
            format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")
        dfs.append(d.dropna(subset=["Ts"]))
    if not dfs:
        raise FileNotFoundError(f"No data for {pair}")
    t = pd.concat(dfs, ignore_index=True).sort_values("Ts").reset_index(drop=True)
    t["ts_s"] = t["Ts"].astype(np.int64) // 10**9
    return t


def find_impulses(ticks, impulse_pips=5, impulse_sec=10):
    mid = (ticks["B"].values + ticks["A"].values) / 2.0
    ts = ticks["ts_s"].values
    n = len(ticks)
    events = []
    i = 0
    while i < n:
        target_time = ts[i] + impulse_sec
        end = int(np.searchsorted(ts, target_time, side="right"))
        end = min(end, n)
        window = mid[i:end]
        if len(window) < 2:
            i += 1; continue
        start_price = window[0]
        high, low = np.max(window), np.min(window)
        move_up = (high - start_price) * 10000
        move_down = abs((low - start_price) * 10000)
        if max(move_up, move_down) >= impulse_pips:
            direction = 1 if move_up >= move_down else -1
            extreme_idx = i + (np.argmax(window) if direction == 1 else np.argmin(window))
            events.append({
                "time": ts[i], "extreme_time": ts[extreme_idx],
                "impulse_pips": max(move_up, move_down),
                "impulse_sec": ts[extreme_idx] - ts[i],
                "direction": direction,
                "price_start": start_price,
                "price_extreme": mid[extreme_idx],
                "extreme_idx": extreme_idx,
                "ask_start": ticks["A"].values[i],
                "bid_start": ticks["B"].values[i],
            })
            i = extreme_idx
        else:
            i += 1
    return pd.DataFrame(events)


def simulate(events, ticks, hold_sec=15, flip_dir=False, cost=0.00003, pip_size=0.0001):
    bids, asks, ts = ticks["B"].values, ticks["A"].values, ticks["ts_s"].values
    n = len(ticks)
    trades = []
    for _, ev in events.iterrows():
        entry_dir = ev["direction"] if flip_dir else -ev["direction"]
        entry_idx = int(ev["extreme_idx"]) + 1
        if entry_idx >= n: continue
        entry_price = asks[entry_idx] if entry_dir == 1 else bids[entry_idx]
        entry_time = ts[entry_idx]
        exit_time = entry_time + hold_sec
        exit_idx = int(np.searchsorted(ts, exit_time, side="right"))
        if exit_idx >= n: continue
        exit_price = bids[exit_idx] if entry_dir == 1 else asks[exit_idx]
        pnl = (exit_price - entry_price) * entry_dir - cost
        pnl_pips = pnl / pip_size
        trades.append({
            "entry_time": entry_time, "entry_price": entry_price,
            "exit_price": exit_price, "pnl_pips": pnl_pips, "pnl": pnl,
            "direction": entry_dir, "impulse_pips": ev["impulse_pips"],
            "hold_sec": hold_sec, "flip_dir": flip_dir,
        })
    return pd.DataFrame(trades)


def retrace_5s(events, ticks):
    """Compute retracement % at 5s post-extreme. Returns array matching events."""
    mid = (ticks["B"].values + ticks["A"].values) / 2.0
    ts = ticks["ts_s"].values
    n = len(ticks)
    results = []
    for _, ev in events.iterrows():
        ext_idx = int(ev["extreme_idx"])
        ext_t = ts[ext_idx]
        ext_price = ev["price_extreme"]
        start_price = ev["price_start"]
        direction = int(ev["direction"])
        impulse = abs(ext_price - start_price)
        if impulse == 0: results.append(0.0); continue
        t5 = ext_t + 5
        idx5 = int(np.searchsorted(ts, t5, side="right"))
        if idx5 >= n: results.append(0.0); continue
        price5 = mid[idx5]
        if direction == 1:
            retrace = (ext_price - price5) / impulse  # + = price came back down
        else:
            retrace = (price5 - ext_price) / impulse  # + = price came back up
        results.append(retrace)
    return np.array(results)


def monthly_label(entry_time):
    t = pd.to_datetime(entry_time, unit="s")
    return f"{t.year}-{t.month:02d}"


def run_test(pair, months, verbose=True):
    pip = PIP_SIZE.get(pair, 0.0001)
    cost = SPREAD_COST.get(pair, 0.00003)
    cost_pips = cost / pip
    
    print(f"\n{'='*70}")
    print(f"  {pair} | spread_cost={cost_pips:.1f}p | months={months}")
    print(f"{'='*70}")
    
    t0 = time.time()
    ticks = load_ticks(pair, months)
    all_results = []
    
    for ip, isec in IMPULSE_CONFIGS:
        impulses = find_impulses(ticks, ip, isec)
        if len(impulses) == 0:
            if verbose: print(f"  {ip}p/{isec}s: 0 impulses")
            continue

        ret = retrace_5s(impulses, ticks)
        
        for hold in HOLD_TIMES:
            for flip in [False, True]:
                trades = simulate(impulses, ticks, hold, flip, cost, pip)
                if len(trades) == 0: continue
                trades["retrace"] = ret[:len(trades)] if len(ret) >= len(trades) else 0
                
                dir_label = "MOM" if flip else "FADE"
                pnls = trades["pnl_pips"].values
                n = len(pnls)
                w = np.sum(pnls > 0)
                l = np.sum(pnls < 0)
                wr = w / max(w + l, 1) * 100
                gross = pnls.sum()
                
                gate_ok = trades["retrace"].values >= 0.1
                g_pnls = pnls[gate_ok]
                g_n, g_w, g_l = len(g_pnls), np.sum(g_pnls > 0), np.sum(g_pnls < 0)
                g_wr = g_w / max(g_w + g_l, 1) * 100
                g_gross = g_pnls.sum() if g_n > 0 else 0
                
                all_results.append({
                    "pair": pair, "impulse_pips": ip, "impulse_sec": isec,
                    "hold_sec": hold, "direction": dir_label,
                    "n": n, "wr": wr, "gross_pips": gross,
                    "avg_pips": gross / n if n > 0 else 0,
                    "g_n": g_n, "g_wr": g_wr, "g_gross_pips": g_gross,
                })
                
                # Monthly breakdown from same data
                if not flip and hold in [5, 15, 30]:
                    trades["month"] = trades["entry_time"].apply(monthly_label)
                    for month, grp in trades.groupby("month"):
                        p = grp["pnl_pips"].values
                        m_w = np.sum(p > 0)
                        m_l = np.sum(p < 0)
                        all_results.append({
                            "pair": pair, "impulse_pips": ip, "impulse_sec": isec,
                            "hold_sec": hold, "direction": "FADE",
                            "month": month,
                            "n": len(p), "wr": m_w / max(m_w + m_l, 1) * 100,
                            "gross_pips": p.sum(), "avg_pips": p.mean(),
                        })
    
    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.1f}s | {sum(r['n'] for r in all_results if 'month' not in r)} total trade simulations")
    
    return all_results


def print_summary(all_results):
    df = pd.DataFrame(all_results)
    
    # ── Best overall per pair ──
    print(f"\n{'='*70}")
    print("  BEST CONFIGS PER PAIR (fade, un-gated)")
    print(f"{'='*70}")
    fade = df[(df["direction"] == "FADE") & (df["month"].isna())]
    for pair in fade["pair"].unique():
        sub = fade[fade["pair"] == pair]
        best = sub.loc[sub["gross_pips"].idxmax()] if len(sub) > 0 else None
        if best is not None:
            print(f"  {pair}: {best['impulse_pips']:.0f}p/{best['impulse_sec']:.0f}s hold={best['hold_sec']:.0f}s "
                  f"n={best['n']:.0f} WR={best['wr']:.1f}% Gross={best['gross_pips']:+.1f}p")
    
    # ── Retrace gate comparison ──
    print(f"\n{'='*70}")
    print("  RETRACE GATE EFFECT (retrace_5s >= 0.1)")
    print(f"{'='*70}")
    for pair in fade["pair"].unique():
        sub = fade[fade["pair"] == pair].copy()
        sub = sub[sub["g_n"] > 0]
        if len(sub) == 0: continue
        best_idx = sub["gross_pips"].idxmax() if "gross_pips" in sub.columns else None
        if best_idx is not None:
            r = sub.loc[best_idx]
            imp = f"{r['impulse_pips']:.0f}p/{r['impulse_sec']:.0f}s hold={r['hold_sec']:.0f}s"
            impr = (r['g_wr'] - r['wr']) if r['wr'] != 0 else 0
            print(f"  {pair} {imp}: {r['n']:.0f}t → {r['g_n']:.0f}t gated | "
                  f"WR {r['wr']:.1f}% → {r['g_wr']:.1f}% ({impr:+.1f}pp) | "
                  f"Gross {r['gross_pips']:+.1f}p → {r['g_gross_pips']:+.1f}p")
    
    # ── Direction test ──
    print(f"\n{'='*70}")
    print("  DIRECTION TEST (fade vs momentum)")
    print(f"{'='*70}")
    for pair in fade["pair"].unique():
        sub = df[(df["pair"] == pair) & (df["month"].isna())]
        for imp in IMPULSE_CONFIGS:
            ip, isec = imp
            for hold in [15, 30]:
                f = sub[(sub["impulse_pips"] == ip) & (sub["impulse_sec"] == isec) &
                        (sub["hold_sec"] == hold) & (sub["direction"] == "FADE")]
                m = sub[(sub["impulse_pips"] == ip) & (sub["impulse_sec"] == isec) &
                        (sub["hold_sec"] == hold) & (sub["direction"] == "MOM")]
                if len(f) > 0 and len(m) > 0:
                    fr = f.iloc[0]; mr = m.iloc[0]
                    print(f"  {pair} {ip}p/{isec}s hold={hold}s: FADE n={fr['n']:.0f} WR={fr['wr']:.1f}% Gross={fr['gross_pips']:+.1f}p | "
                          f"MOM n={mr['n']:.0f} WR={mr['wr']:.1f}% Gross={mr['gross_pips']:+.1f}p")
    
    # ── Monthly breakdown ──
    print(f"\n{'='*70}")
    print("  MONTHLY BREAKDOWN (fade, no gate)")
    print(f"{'='*70}")
    monthly = df[df["month"].notna()]
    for pair in monthly["pair"].unique():
        sub = monthly[monthly["pair"] == pair]
        for imp in IMPULSE_CONFIGS:
            ip, isec = imp
            for hold in [15, 30]:
                rows = sub[(sub["impulse_pips"] == ip) & (sub["impulse_sec"] == isec) & (sub["hold_sec"] == hold)]
                if len(rows) == 0: continue
                months_str = " | ".join(
                    f"{r['month']}: n={r['n']:.0f} WR={r['wr']:.1f}% {r['gross_pips']:+.1f}p"
                    for _, r in rows.iterrows()
                )
                print(f"  {pair} {ip}p/{isec}s hold={hold}s: {months_str}")
    
    # ── Frequency analysis ──
    print(f"\n{'='*70}")
    print("  EVENT FREQUENCY (trades/month)")
    print(f"{'='*70}")
    for pair in fade["pair"].unique():
        sub = fade[fade["pair"] == pair]
        for imp in IMPULSE_CONFIGS:
            ip, isec = imp
            rows = sub[(sub["impulse_pips"] == ip) & (sub["impulse_sec"] == isec) & (sub["hold_sec"] == 15)]
            if len(rows) == 0: continue
            r = rows.iloc[0]
            print(f"  {pair} {ip}p/{isec}s: ~{r['n']/3:.0f}/month ({r['n']:.0f} total)")
    
    # ── Failure mode checks ──
    print(f"\n{'='*70}")
    print("  FAILURE MODE VALIDATION (from RESEARCH_PLAN_v2.md)")
    print(f"{'='*70}")
    
    # Check sub-spread signal (problem #8): avg PnL > spread cost?
    for pair in fade["pair"].unique():
        sub = fade[fade["pair"] == pair]
        best_avg = sub["avg_pips"].max() if len(sub) > 0 else 0
        cost_p = SPREAD_COST.get(pair, 0.00003) / PIP_SIZE.get(pair, 0.0001)
        print(f"  {pair}: best avg={best_avg:+.2f}p vs spread={cost_p:.1f}p → "
              f"{'SURVIVES' if best_avg > cost_p else 'SUB-SPREAD SIGNAL'}")
    
    # Direction edge (problem #9): both fail?
    for pair in fade["pair"].unique():
        sub = df[(df["pair"] == pair) & (df["month"].isna()) & (df["hold_sec"] == 15)]
        fade_best = sub[sub["direction"] == "FADE"]["wr"].max() if len(sub[sub["direction"] == "FADE"]) > 0 else 0
        mom_best = sub[sub["direction"] == "MOM"]["wr"].max() if len(sub[sub["direction"] == "MOM"]) > 0 else 0
        print(f"  {pair}: direction edge FADE={fade_best:.1f}% MOM={mom_best:.1f}% "
              f"gap={fade_best-mom_best:+.1f}pp → {'FADE EDGE' if fade_best > mom_best + 5 else 'NO DIRECTION EDGE'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pairs", nargs="*", default=["EURUSD", "EURJPY", "GBPJPY"])
    parser.add_argument("--months", default="10,11,12")
    args = parser.parse_args()
    
    months_list = [(2025, int(m)) for m in args.months.split(",")]
    
    all_results = []
    for pair in args.pairs:
        results = run_test(pair, months_list)
        all_results.extend(results)
    
    print_summary(all_results)
    
    # Quick verdict
    print(f"\n{'='*70}")
    print("  VERDICT")
    print(f"{'='*70}")
    df = pd.DataFrame(all_results)
    fade = df[(df["direction"] == "FADE") & (df["month"].isna())]
    for pair in args.pairs:
        sub = fade[fade["pair"] == pair]
        best = sub.loc[sub["gross_pips"].idxmax()] if len(sub) > 0 else None
        cost_p = SPREAD_COST.get(pair, 0.00003) / PIP_SIZE.get(pair, 0.0001)
        if best is not None:
            verdict = "VIABLE" if best["wr"] > 55 and best["n"] > 50 else "UNLIKELY" if best["wr"] > 50 else "FAILS"
            print(f"  {pair}: {verdict} — best WR={best['wr']:.1f}% n={best['n']:.0f} gross={best['gross_pips']:+.1f}p cost={cost_p:.1f}p")
        else:
            print(f"  {pair}: NO DATA")
