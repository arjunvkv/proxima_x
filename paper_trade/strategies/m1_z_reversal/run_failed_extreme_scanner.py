"""Failed Extreme Scanner — fast single-pass over EURUSD ticks.

Detects sharp impulses, measures post-impulse reaction, tests fade trades.
Uses October 2025 EURUSD only for speed (~1M ticks).

Usage:
    python run_failed_extreme_scanner.py
    python run_failed_extreme_scanner.py --impulse 5 5    # 5 pips in 5 seconds
    python run_failed_extreme_scanner.py --all             # Sweep all thresholds
"""
import argparse, time
import numpy as np
import pandas as pd
from pathlib import Path

TICK_DIR = Path(r"C:\Trading\Agentic_Trading\proxima_x\data\exness_ticks")


def load_ticks(pair="EURUSD", months=None):
    if months is None:
        months = [(2025, 10)]
    dfs = []
    for y, m in months:
        p = TICK_DIR / f"{pair}_Raw_Spread_{y}_{m:02d}.zip"
        d = pd.read_csv(p, names=["E","S","Ts","B","A"], skiprows=1, header=None,
                        dtype={"Ts": str, "B": np.float64, "A": np.float64})
        d["Ts"] = pd.to_datetime(d["Ts"].str.replace("Z","",regex=False),
            format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")
        dfs.append(d.dropna(subset=["Ts"]))
    t = pd.concat(dfs, ignore_index=True).sort_values("Ts").reset_index(drop=True)
    t["ts_s"] = t["Ts"].astype(np.int64) // 10**9
    return t


def find_impulses(ticks, impulse_pips=5, impulse_sec=5):
    """Find sharp price moves: X pips in Y seconds.
    
    Uses mid-price for detection. Returns event DataFrame.
    """
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
            i += 1
            continue
        start_price = window[0]
        high = np.max(window)
        low = np.min(window)
        high_pips = (high - start_price) * 10000
        low_pips = (low - start_price) * 10000
        move_up = high_pips
        move_down = abs(low_pips)
        max_move = max(move_up, move_down)
        if max_move >= impulse_pips:
            direction = 1 if move_up >= move_down else -1
            extreme_idx = i + (np.argmax(window) if direction == 1 else np.argmin(window))
            events.append({
                "time": ts[i],
                "extreme_time": ts[extreme_idx],
                "impulse_pips": max_move,
                "impulse_sec": ts[extreme_idx] - ts[i],
                "direction": direction,
                "price_start": start_price,
                "price_extreme": mid[extreme_idx],
                "event_idx": i,
                "extreme_idx": extreme_idx,
            })
            i = extreme_idx
        else:
            i += 1
    
    return pd.DataFrame(events)


def simulate_fade(events, ticks, hold_sec=30, flip_dir=False):
    """Simulate fade trade: enter opposite to impulse, hold fixed time.
    
    If flip_dir=True, enter IN SAME direction as impulse (momentum test).
    """
    mid = (ticks["B"].values + ticks["A"].values) / 2.0
    bids = ticks["B"].values
    asks = ticks["A"].values
    ts = ticks["ts_s"].values
    n = len(ticks)
    
    trades = []
    for _, ev in events.iterrows():
        entry_dir = ev["direction"] if flip_dir else -ev["direction"]  # fade or momentum
        entry_idx = int(ev["extreme_idx"]) + 1  
        if entry_idx >= n:
            continue
        
        entry_price = asks[entry_idx] if entry_dir == 1 else bids[entry_idx]
        entry_time = ts[entry_idx]
        
        exit_time = entry_time + hold_sec
        exit_idx = int(np.searchsorted(ts, exit_time, side="right"))
        if exit_idx >= n:
            continue
        
        exit_price = bids[exit_idx] if entry_dir == 1 else asks[exit_idx]
        
        pnl = (exit_price - entry_price) * entry_dir
        
        trades.append({
            "entry_time": entry_time,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl_pips": pnl * 10000,
            "direction": entry_dir,
            "impulse_pips": ev["impulse_pips"],
            "hold_sec": exit_time - entry_time,
        })
    
    return pd.DataFrame(trades)


def extract_features(events, ticks, reaction_window=15):
    """Compute RQ1 reaction features for each impulse event.
    
    Features computed in the `reaction_window` seconds after the extreme tick:
    - retracement_pct: % of impulse retraced at 5s, 10s, reaction_window
    - tick_imbalance: bid vs ask tick count ratio
    - failed_extremes: count of ticks near extreme (within 10% of impulse) that don't break it
    - spread_pctl: spread at detection compared to rolling median
    - vol_decay: volatility (tick-to-tick range) in 0-5s vs 10-15s
    """
    mid = (ticks["B"].values + ticks["A"].values) / 2.0
    spreads = (ticks["A"].values - ticks["B"].values) * 10000  # in pips
    ts = ticks["ts_s"].values
    n = len(ticks)
    
    features = []
    for _, ev in events.iterrows():
        extreme_idx = int(ev["extreme_idx"])
        extreme_time = ts[extreme_idx]
        extreme_price = ev["price_extreme"]
        start_price = ev["price_start"]
        impulse = ev["impulse_pips"] / 10000  # raw price units
        direction = int(ev["direction"])
        
        # Window after extreme
        window_end = extreme_time + reaction_window
        w_end = int(np.searchsorted(ts, window_end, side="right"))
        w_end = min(w_end, n)
        
        window_mid = mid[extreme_idx:w_end]
        window_ts = ts[extreme_idx:w_end]
        window_spread = spreads[extreme_idx:w_end]
        
        if len(window_mid) < 2:
            features.append(None)
            continue
        
        # 1. Retracement at multiple points
        retrace_5s = _retracement(window_mid, window_ts, 5, extreme_time, start_price, extreme_price, direction)
        retrace_10s = _retracement(window_mid, window_ts, 10, extreme_time, start_price, extreme_price, direction)
        retrace_15s = _retracement(window_mid, window_ts, 15, extreme_time, start_price, extreme_price, direction)
        
        # 2. Tick imbalance (bid vs ask in reaction window)
        bid_ticks = (ticks["B"].values[extreme_idx:w_end] + ticks["A"].values[extreme_idx:w_end]) / 2
        bid_count = 0
        ask_count = 0
        for j in range(extreme_idx, w_end):
            price_change = mid[j] - (mid[j-1] if j > extreme_idx else extreme_price)
            if price_change < 0:
                bid_count += 1  # price went down, selling pressure
            elif price_change > 0:
                ask_count += 1  # price went up, buying pressure
        tick_imb = (ask_count - bid_count) / max(bid_count + ask_count, 1)
        
        # 3. Failed extreme attempts
        failed = 0
        impulse_threshold = abs(extreme_price - start_price) * 0.1
        for j in range(extreme_idx, w_end):
            if direction == 1:
                dist_from_extreme = extreme_price - mid[j]
            else:
                dist_from_extreme = mid[j] - extreme_price
            if 0 < dist_from_extreme < impulse_threshold:
                failed += 1
        
        # 4. Spread percentile at detection
        pre_spreads = spreads[max(0, extreme_idx-200):extreme_idx]
        rolling_median = np.median(pre_spreads) if len(pre_spreads) > 0 else 0.03
        spread_at = window_spread[0] if len(window_spread) > 0 else 0.03
        spread_ratio = spread_at / max(rolling_median, 0.001)
        
        # 5. Volatility decay (0-5s vs 10-15s)
        vol_early = _volatility(window_mid, window_ts, 0, 5, extreme_time)
        vol_late = _volatility(window_mid, window_ts, 10, 15, extreme_time)
        vol_decay = vol_late / max(vol_early, 0.0001) if vol_early > 0 else 1.0
        
        features.append({
            "retrace_5s": retrace_5s,
            "retrace_10s": retrace_10s,
            "retrace_15s": retrace_15s,
            "tick_imbalance": tick_imb,
            "failed_extremes": failed,
            "spread_ratio": spread_ratio,
            "vol_decay": vol_decay,
            "extreme_idx": extreme_idx,
        })
    
    return features


def _retracement(window_mid, window_ts, lookback, extreme_time, start_price, extreme_price, direction):
    """Compute % retracement of impulse at `lookback` seconds after extreme."""
    target = extreme_time + lookback
    idx = np.searchsorted(window_ts, target, side="right")
    if idx >= len(window_mid):
        idx = -1
    current = window_mid[idx]
    impulse = extreme_price - start_price
    if abs(impulse) < 1e-10:
        return 0.0
    if direction == 1:
        retrace = (extreme_price - current) / impulse
    else:
        retrace = (current - extreme_price) / abs(impulse)
    return float(np.clip(retrace, -2, 2))


def _volatility(window_mid, window_ts, t0, t1, extreme_time):
    """Volatility as mean absolute tick-to-tick change in [t0, t1] window."""
    start_t = extreme_time + t0
    end_t = extreme_time + t1
    s = np.searchsorted(window_ts, start_t, side="right")
    e = np.searchsorted(window_ts, end_t, side="right")
    if e - s < 2:
        return 0.0001
    changes = np.abs(np.diff(window_mid[s:e]))
    return float(np.mean(changes)) if len(changes) > 0 else 0.0001


def report_classifier(features_list, trades_df, label):
    """Print feature comparison for winners vs losers."""
    if trades_df is None or len(trades_df) < 5:
        return
    winners = trades_df[trades_df["pnl_pips"] > 0]
    losers = trades_df[trades_df["pnl_pips"] <= 0]
    
    feat_keys = ["retrace_5s", "retrace_10s", "retrace_15s", "tick_imbalance",
                  "failed_extremes", "spread_ratio", "vol_decay"]
    
    print(f"\n  {label} — Feature Comparison (winners vs losers)")
    print(f"  {'Feature':<18} {'Winners':>10} {'Losers':>10} {'Delta':>10} {'Direction':>10}")
    print(f"  {'-'*58}")
    
    for key in feat_keys:
        w_vals = [f[key] for f, (_, tr) in zip(features_list, trades_df.iterrows())
                  if f is not None and tr["pnl_pips"] > 0]
        l_vals = [f[key] for f, (_, tr) in zip(features_list, trades_df.iterrows())
                  if f is not None and tr["pnl_pips"] <= 0]
        if not w_vals or not l_vals:
            continue
        w_avg = np.mean(w_vals)
        l_avg = np.mean(l_vals)
        delta = w_avg - l_avg
        # Direction: which side is better (+ means higher is better for fade)
        direction_marker = "+" if (key.startswith("retrace") or key == "failed_extremes" or key == "vol_decay") else "-"
        print(f"  {key:<18} {w_avg:>+10.3f} {l_avg:>+10.3f} {delta:>+10.3f}")


def report(trades, label):
    if len(trades) == 0:
        print(f"  {label}: 0 trades")
        return
    wins = (trades["pnl_pips"] > 0).sum()
    total = len(trades)
    wr = wins / total
    avg_pnl = trades["pnl_pips"].mean()
    gross = trades["pnl_pips"].sum()
    print(f"  {label}: n={total} WR={wr:.1%} avg={avg_pnl:+.2f}p gross={gross:+.1f}p")


def run_single(impulse_pips, impulse_sec, hold_sec=30):
    print(f"\n  Impulse: {impulse_pips}p/{impulse_sec}s  Hold: {hold_sec}s")
    events = find_impulses(ticks, impulse_pips, impulse_sec)
    if len(events) == 0:
        print(f"    0 events detected")
        return
    trades = simulate_fade(events, ticks, hold_sec)
    report(trades, "Fade")


def sweep_all():
    print("=" * 60)
    print("FAILED EXTREME SCANNER — Threshold Sweep")
    print(f"Data: EURUSD Oct 2025 ({len(ticks):,} ticks)")
    print("=" * 60)
    
    impulses = [
        (3, 3), (3, 5), (5, 3), (5, 5), 
        (5, 10), (7, 5), (7, 10), (10, 5), (10, 10), (10, 15),
        (15, 10), (15, 15), (20, 15), (20, 30),
    ]
    holds = [15, 30, 60, 120]
    
    print(f"\n{'Impulse':>12} {'Hold':>6} {'n':>6} {'WR':>8} {'AvgP':>8} {'Gross':>8}")
    print("-" * 52)
    results = []
    for ip, isec in impulses:
        events = find_impulses(ticks, ip, isec)
        for hold in holds:
            trades = simulate_fade(events, ticks, hold)
            if len(trades) < 5:
                continue
            wr = (trades["pnl_pips"] > 0).mean()
            avg = trades["pnl_pips"].mean()
            gross = trades["pnl_pips"].sum()
            results.append((ip, isec, hold, len(trades), wr, avg, gross))
    
    results.sort(key=lambda r: -r[4])  # sort by WR desc
    for r in results:
        ip, isec, hold, n, wr, avg, gross = r
        flag = " <<<" if wr > 0.55 and n >= 20 else ""
        print(f"  {ip:>3}p/{isec:<2}s {hold:>5}s {n:>5} {wr:>7.1%} {avg:>+7.2f}p {gross:>+7.1f}p{flag}")
    
    print(f"\nBEST by WR (min 20 trades):")
    best_wr = [r for r in results if r[3] >= 20]
    if best_wr:
        best_wr.sort(key=lambda r: -r[4])
        top = best_wr[0]
        print(f"  {top[0]}p/{top[1]}s hold={top[2]}s: n={top[3]} WR={top[4]:.1%} avg={top[5]:+.2f}p gross={top[6]:+.1f}p")
        print(f"\nFeature analysis for best config ({top[0]}p/{top[1]}s hold={top[2]}s):")
        _analyze_config(ticks, top[0], top[1], top[2])
        
        # Also analyze the best trade-off config
        print(f"\nFeature analysis for trade-off config (5p/10s hold=15s):")
        _analyze_config(ticks, 5, 10, 15)


def _analyze_config(ticks, impulse_pips, impulse_sec, hold_sec):
    ev = find_impulses(ticks, impulse_pips, impulse_sec)
    tr = simulate_fade(ev, ticks, hold_sec)
    feats = extract_features(ev, ticks)
    valid = [(f, tr.iloc[i]) for i, f in enumerate(feats) if f is not None and i < len(tr)]
    if not valid:
        return
    f_list, tr_view = zip(*valid)
    report_classifier(f_list, pd.DataFrame(tr_view), f"{impulse_pips}p/{impulse_sec}s")
    
    # Test retracement gate: only enter if retrace_5s > threshold
    print(f"\n  Retracement gate test (enter if retrace_5s > threshold):")
    for thresh in [0.0, 0.1, 0.2, 0.3, 0.5]:
        gated_trades = []
        for f, trade in zip(feats, [dict(zip(tr.columns, r)) for _, r in tr.iterrows()]):
            if f is None:
                continue
            if f["retrace_5s"] >= thresh:
                gated_trades.append(trade)
        if len(gated_trades) >= 5:
            gated_df = pd.DataFrame(gated_trades)
            g_wr = (gated_df["pnl_pips"] > 0).mean()
            g_n = len(gated_df)
            g_avg = gated_df["pnl_pips"].mean()
            print(f"    retrace >= {thresh:.1f}: n={g_n} WR={g_wr:.1%} avg={g_avg:+.2f}p")
        else:
            print(f"    retrace >= {thresh:.1f}: {len(gated_trades)} trades (too few)")


def run_test(test_fn, label, ticks_cache):
    """Cache-aware test runner: stores loaded ticks by (pair, months_tuple)."""
    pass  # placeholder


def adversity_tests():
    """Comprehensive adversity testing against every problem from RESEARCH_PLAN_v2.md."""
    print("=" * 70)
    print("ADVERSITY TEST SUITE — Against all RESEARCH_PLAN_v2.md failures")
    print("=" * 70)
    
    configs = [(5, 10, 15), (7, 10, 15), (5, 5, 15), (3, 5, 15)]
    pairs = ["EURUSD", "EURJPY", "GBPJPY"]
    months = [(2025, 10), (2025, 11), (2025, 12)]
    
    # Preload all data: data[(pair, month_tuple)] = ticks_df
    print("Preloading tick data across all pairs and months...")
    data = {}
    for pair in pairs:
        t_all = load_ticks(pair, months)
        data[(pair, "all")] = t_all
        print(f"  {pair} full: {len(t_all):,} ticks")
        for m in months:
            t_m = load_ticks(pair, [m])
            data[(pair, m)] = t_m
            print(f"  {pair} {m[1]}: {len(t_m):,} ticks")
        t_train = load_ticks(pair, [(2025, 10), (2025, 11)])
        data[(pair, "train")] = t_train
        print(f"  {pair} train: {len(t_train):,} ticks")
    print()
    
    # ── 1. Monthly breakdown per pair ──
    print(f"{'─'*70}")
    print("TEST 1: Monthly breakdown — does the signal hold every month?")
    print(f"{'─'*70}")
    for pair in pairs:
        print(f"\n  {pair}:")
        for m in months:
            t = data[(pair, m)]
            for ip, isec, hold in configs:
                ev = find_impulses(t, ip, isec)
                tr = simulate_fade(ev, t, hold)
                if len(tr) < 5:
                    continue
                wr = (tr["pnl_pips"] > 0).mean()
                avg = tr["pnl_pips"].mean()
                gross = tr["pnl_pips"].sum()
                flag = ""
                if wr < 0.50:
                    flag = " !! FAILS"
                elif wr >= 0.60:
                    flag = " ✓"
                print(f"    {m[1]:>2d}mo {ip}p/{isec}s h{hold}s: n={len(tr):>4d} WR={wr:.1%} avg={avg:+.2f}p gross={gross:+.1f}p{flag}")
    
    # ── 2. Retrace gate stability across months ──
    print(f"\n{'-'*70}")
    print("TEST 2: Retrace gate (>=0.1) stability across months")
    print( '-' * 70)
    for pair in pairs:
        print(f"\n  {pair}:")
        for m in months:
            t = data[(pair, m)]
            for ip, isec, hold in [(5, 10, 15), (7, 10, 15)]:
                ev = find_impulses(t, ip, isec)
                tr = simulate_fade(ev, t, hold)
                feats = extract_features(ev, t)
                gated = []
                for f, (_, trade) in zip(feats, tr.iterrows()):
                    if f is not None and f["retrace_5s"] >= 0.1:
                        gated.append(trade.to_dict() if hasattr(trade, 'to_dict') else trade)
                if len(gated) >= 5:
                    g_df = pd.DataFrame(gated)
                    g_wr = (g_df["pnl_pips"] > 0).mean()
                    g_avg = g_df["pnl_pips"].mean()
                    print(f"    {m[1]:>2d}mo {ip}p/{isec}s gated: n={len(g_df):>4d} WR={g_wr:.1%} avg={g_avg:+.2f}p")
                else:
                    print(f"    {m[1]:>2d}mo {ip}p/{isec}s gated: {len(gated)} trades (too few)")
    
    # ── 3. Direction reversal test (momentum) ──
    print(f"\n{'-'*70}")
    print("TEST 3: Direction reversal — does momentum ALSO work? (if yes, signal is random)")
    print('-' * 70)
    for pair in pairs:
        t = data[(pair, "all")]
        print(f"\n  {pair}:")
        for ip, isec, hold in [(5, 10, 15), (7, 10, 15)]:
            ev = find_impulses(t, ip, isec)
            tr_fade = simulate_fade(ev, t, hold)
            tr_mom = simulate_fade(ev, t, hold, flip_dir=True)
            wr_fade = (tr_fade["pnl_pips"] > 0).mean() if len(tr_fade) > 0 else 0
            wr_mom = (tr_mom["pnl_pips"] > 0).mean() if len(tr_mom) > 0 else 0
            avg_fade = tr_fade["pnl_pips"].mean() if len(tr_fade) > 0 else 0
            avg_mom = tr_mom["pnl_pips"].mean() if len(tr_mom) > 0 else 0
            gap = wr_fade - wr_mom
            verdict = "✓ directional edge" if gap > 0.15 else "?? weak directionality" if gap > 0.05 else "!! NO DIRECTIONAL EDGE"
            print(f"    {ip}p/{isec}s: Fade WR={wr_fade:.1%}({avg_fade:+.2f}p) Mom WR={wr_mom:.1%}({avg_mom:+.2f}p) gap={gap:+.0%} {verdict}")
    
    # ── 4. Random entry baseline ──
    print(f"\n{'-'*70}")
    print("TEST 4: Random entry baseline — is edge real or microstructure noise?")
    print('-' * 70)
    for pair in pairs[:1]:
        t = data[(pair, "all")]
        print(f"\n  {pair}:")
        for ip, isec, hold in [(5, 10, 15), (7, 10, 15)]:
            ev = find_impulses(t, ip, isec)
            tr = simulate_fade(ev, t, hold)
            if len(tr) < 20:
                continue
            rng = np.random.RandomState(42)
            rand_pnls = tr["pnl_pips"].values.copy()
            for i in range(len(rand_pnls)):
                rand_pnls[i] *= (1 if rng.rand() > 0.5 else -1)
            wr_rand = (rand_pnls > 0).mean()
            wr_real = (tr["pnl_pips"] > 0).mean()
            print(f"    {ip}p/{isec}s: Real WR={wr_real:.1%} Random WR={wr_rand:.1%} edge={wr_real-wr_rand:+.0%} {'✓' if wr_real - wr_rand > 0.08 else '??'}")
    
    # ── 5. Walk-forward: train Oct+Nov, test Dec ──
    print(f"\n{'-'*70}")
    print("TEST 5: Walk-forward — train Oct+Nov, test Dec (holiday regime)")
    print('-' * 70)
    for pair in pairs:
        train = data[(pair, "train")]
        test = data[(pair, (2025, 12))]
        print(f"\n  {pair}: train={len(train):,} test={len(test):,}")
        
        best_wr, best_cfg = 0, None
        for ip, isec, hold in configs:
            ev = find_impulses(train, ip, isec)
            tr = simulate_fade(ev, train, hold)
            if len(tr) < 20:
                continue
            wr = (tr["pnl_pips"] > 0).mean()
            if wr > best_wr:
                best_wr, best_cfg = wr, (ip, isec, hold)
        
        if best_cfg:
            ip, isec, hold = best_cfg
            print(f"    Best on train: {ip}p/{isec}s hold={hold}s WR={best_wr:.1%}")
            ev_test = find_impulses(test, ip, isec)
            tr_test = simulate_fade(ev_test, test, hold)
            if len(tr_test) > 0:
                wr_test = (tr_test["pnl_pips"] > 0).mean()
                avg_test = tr_test["pnl_pips"].mean()
                gross_test = tr_test["pnl_pips"].sum()
                verdict = "✓ HOLDS" if wr_test > 0.55 else "!! DEGRADES" if wr_test < 0.50 else "? marginal"
                print(f"    Test Dec: n={len(tr_test)} WR={wr_test:.1%} avg={avg_test:+.2f}p gross={gross_test:+.1f}p {verdict}")
                feats_test = extract_features(ev_test, test)
                gated = []
                for f, (_, trade) in zip(feats_test, tr_test.iterrows()):
                    if f is not None and f["retrace_5s"] >= 0.1:
                        gated.append(trade.to_dict() if hasattr(trade, 'to_dict') else trade)
                if len(gated) >= 5:
                    g_df = pd.DataFrame(gated)
                    g_wr = (g_df["pnl_pips"] > 0).mean()
                    print(f"    Test Dec gated: n={len(g_df)} WR={g_wr:.1%} avg={g_df['pnl_pips'].mean():+.2f}p")
    
    # ── 6. Full period summary ──
    print(f"\n{'-'*70}")
    print("TEST 6: Full period (Oct-Dec) performance summary")
    print('-' * 70)
    for pair in pairs:
        t = data[(pair, "all")]
        print(f"\n  {pair} ({len(t):,} ticks):")
        for ip, isec, hold in configs:
            ev = find_impulses(t, ip, isec)
            tr = simulate_fade(ev, t, hold)
            if len(tr) < 10:
                continue
            wr = (tr["pnl_pips"] > 0).mean()
            avg = tr["pnl_pips"].mean()
            gross = tr["pnl_pips"].sum()
            print(f"    {ip}p/{isec}s hold={hold}s: n={len(tr):>4d} WR={wr:.1%} avg={avg:+.2f}p gross={gross:+.1f}p")
            
            feats = extract_features(ev, t)
            gated = []
            for f, (_, trade) in zip(feats, tr.iterrows()):
                if f is not None and f["retrace_5s"] >= 0.1:
                    gated.append(trade.to_dict() if hasattr(trade, 'to_dict') else trade)
            if len(gated) >= 10:
                g_df = pd.DataFrame(gated)
                g_wr = (g_df["pnl_pips"] > 0).mean()
                g_avg = g_df["pnl_pips"].mean()
                g_gross = g_df["pnl_pips"].sum()
                print(f"           gated(ret>=0.1): n={len(g_df):>4d} WR={g_wr:.1%} avg={g_avg:+.2f}p gross={g_gross:+.1f}p")


def walkforward_validate(train_months, test_months):
    """Train config on train_months, validate on test_months."""
    print(f"\n{'='*60}")
    print(f"WALK-FORWARD VALIDATION")
    print(f"Train: EURUSD {train_months[0]}")
    print(f"Test:  EURUSD {test_months[0]}")
    print('='*60)
    
    train_ticks = load_ticks("EURUSD", train_months)
    test_ticks = load_ticks("EURUSD", test_months)
    print(f"Train: {len(train_ticks):,} ticks, Test: {len(test_ticks):,} ticks")
    
    # Find best config on train data
    print(f"\n--- Training on train data ---")
    impulses = [(5, 10), (7, 10), (5, 5), (3, 5), (7, 5)]
    best_wr, best_cfg = 0, None
    for ip, isec in impulses:
        ev = find_impulses(train_ticks, ip, isec)
        tr = simulate_fade(ev, train_ticks, 15)
        if len(tr) < 20:
            continue
        wr = (tr["pnl_pips"] > 0).mean()
        if wr > best_wr:
            best_wr = wr
            best_cfg = (ip, isec, 15)
        print(f"  Train {ip}p/{isec}s hold=15s: n={len(tr)} WR={wr:.1%} avg={tr['pnl_pips'].mean():+.2f}p")
    
    if best_cfg is None:
        print("No good config found on train data")
        return
    
    # Also test the 5p/10s config (more trades, slightly lower WR)
    print(f"\n--- Additional config: 5p/10s hold=15s ---")
    ev_5 = find_impulses(test_ticks, 5, 10)
    tr_5 = simulate_fade(ev_5, test_ticks, 15)
    if len(tr_5) > 0:
        wr_5 = (tr_5["pnl_pips"] > 0).mean()
        print(f"  Test 5p/10s hold=15s: n={len(tr_5)} WR={wr_5:.1%} avg={tr_5['pnl_pips'].mean():+.2f}p gross={tr_5['pnl_pips'].sum():+.1f}p")
        feats_5 = extract_features(ev_5, test_ticks)
        g5 = []
        for f, (_, trade) in zip(feats_5, tr_5.iterrows()):
            if f is not None and f["retrace_5s"] >= 0.1:
                g5.append(trade.to_dict() if hasattr(trade, 'to_dict') else trade)
        if g5:
            g5_df = pd.DataFrame(g5)
            gw5 = (g5_df["pnl_pips"] > 0).mean()
            print(f"  Test 5p/10s gated(retrace>=0.1): n={len(g5_df)} WR={gw5:.1%} avg={g5_df['pnl_pips'].mean():+.2f}p gross={g5_df['pnl_pips'].sum():+.1f}p")
    
    ip, isec, hold = best_cfg
    print(f"\nBest config: {ip}p/{isec}s hold={hold}s (WR={best_wr:.1%} on train)")
    print(f"\n--- Testing on test data (no retrace gate) ---")
    ev_test = find_impulses(test_ticks, ip, isec)
    tr_test = simulate_fade(ev_test, test_ticks, hold)
    if len(tr_test) > 0:
        wr = (tr_test["pnl_pips"] > 0).mean()
        print(f"  Test {ip}p/{isec}s hold={hold}s: n={len(tr_test)} WR={wr:.1%} avg={tr_test['pnl_pips'].mean():+.2f}p gross={tr_test['pnl_pips'].sum():+.1f}p")
    
    # Test with retrace gate
    print(f"\n--- Testing on test data (with retrace_5s >= 0.1 gate) ---")
    feats_test = extract_features(ev_test, test_ticks)
    gated_trades = []
    for f, (_, trade) in zip(feats_test, tr_test.iterrows()):
        if f is not None and f["retrace_5s"] >= 0.1:
            gated_trades.append(trade.to_dict() if hasattr(trade, 'to_dict') else trade)
    if gated_trades:
        g_df = pd.DataFrame(gated_trades)
        g_wr = (g_df["pnl_pips"] > 0).mean()
        print(f"  Test gated: n={len(g_df)} WR={g_wr:.1%} avg={g_df['pnl_pips'].mean():+.2f}p gross={g_df['pnl_pips'].sum():+.1f}p")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Failed Extreme Scanner")
    parser.add_argument("--impulse", nargs=2, type=float, default=None,
                        help="Impulse threshold: pips seconds")
    parser.add_argument("--hold", type=int, default=30, help="Hold time in seconds")
    parser.add_argument("--all", action="store_true", help="Sweep all thresholds")
    parser.add_argument("--validate", action="store_true", help="Walk-forward validation (train Oct, test Nov)")
    parser.add_argument("--adversity", action="store_true", help="Run full adversity test suite against all known failure modes")
    args = parser.parse_args()
    
    if args.adversity:
        adversity_tests()
        raise SystemExit(0)
    
    if args.validate:
        walkforward_validate([(2025, 10)], [(2025, 11)])
        raise SystemExit(0)
    
    print("Loading EURUSD Oct 2025 ticks...")
    ticks = load_ticks("EURUSD", [(2025, 10)])
    print(f"Loaded {len(ticks):,} ticks")
    
    if args.all:
        sweep_all()
    elif args.impulse:
        run_single(args.impulse[0], args.impulse[1], args.hold)
    else:
        # Default: quick multi-impulse test
        run_single(3, 5, 30)
        run_single(5, 5, 30)
        run_single(5, 10, 60)
        run_single(10, 10, 60)
        run_single(10, 5, 30)
