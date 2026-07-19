"""Session Transition Stress Index (STSI) — replace clock with state.

Tests:
  1. Does Tokyo edge correlate with STSI state (not just hour)?
  2. Neighbor robustness: parameter plateau (vary lookback/hold ±50%)
  3. OOS: STSI-based vs clock-based performance
"""

import sys, os, numpy as np
from pathlib import Path
from datetime import datetime, timedelta, timezone
from collections import deque

project_root = str(Path(__file__).resolve().parents[2])
os.chdir(project_root)
cd_root = os.path.join(project_root, "currency_decomposition")
sys.path.insert(0, cd_root)

import MetaTrader5 as mt5
if not mt5.initialize():
    raise RuntimeError("MT5 init failed")

from config.settings import BASE_CURRENCY_MAP
ALL_PAIRS = list(BASE_CURRENCY_MAP.keys())[:15]

def load_data():
    end = datetime.now()
    start = end - timedelta(days=120)
    all_data = {}
    for pair in ALL_PAIRS:
        rates = mt5.copy_rates_range(pair, mt5.TIMEFRAME_M5, start, end)
        if rates is not None and len(rates) > 0:
            all_data[pair] = rates
    return all_data, [p for p in ALL_PAIRS if p in all_data]

def stats(trades):
    if not trades or len(trades) < 5: return None
    pnls = [t["pnl"] for t in trades]
    mu, s = float(np.mean(pnls)), float(np.std(pnls))
    wr = sum(1 for t in trades if t["won"]) / len(trades) * 100
    t_stat = mu / (s / np.sqrt(len(trades))) if s > 0 else 0
    return {"n":len(trades),"wr":wr,"mean_bp":mu,"mean_usd":mu*10,"t_stat":t_stat}

# ── FAST STSI ──

def compute_stsi_fast(all_data, avail, idx, lookback=12):
    """Fast STSI — no pair-wise correlation (too slow)."""
    if idx < lookback + 5: return None

    dt = datetime.fromtimestamp(float(all_data[avail[0]][idx]["time"]), tz=timezone.utc)
    hour, minute = dt.hour, dt.minute

    # 1. Cross-sectional dispersion of lookback returns
    returns = []
    for p in avail:
        cur = float(all_data[p][idx]["close"])
        bf = float(all_data[p][idx - lookback]["close"])
        returns.append((cur / bf - 1) if bf > 0 else 0)
    dispersion = float(np.std(returns)) * 10000

    # 2. Volatility shock
    trs = []
    for p in avail:
        hi = float(all_data[p][idx]["high"])
        lo = float(all_data[p][idx]["low"])
        pc = float(all_data[p][idx - 1]["close"])
        tr = max(hi - lo, abs(hi - pc), abs(lo - pc))
        trs.append(tr / float(all_data[p][idx]["close"]))
    atr_now = float(np.mean(trs)) * 10000

    # Rolling ATR mean (288 bars = ~24h)
    atr_hist = []
    for j in range(max(0, idx - 288), idx, 12):  # sample every 12 bars for speed
        trs_j = []
        for p in avail:
            hi = float(all_data[p][j]["high"])
            lo = float(all_data[p][j]["low"])
            pc = float(all_data[p][j - 1]["close"])
            tr = max(hi - lo, abs(hi - pc), abs(lo - pc))
            trs_j.append(tr / float(all_data[p][j]["close"]))
        atr_hist.append(float(np.mean(trs_j)) * 10000)
    atr_mean = float(np.mean(atr_hist)) if atr_hist else atr_now
    vol_shock = atr_now / atr_mean if atr_mean > 0 else 1.0

    # 3. Overnight displacement (return since 00:00 UTC)
    today_start = dt.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    odisp = 0
    found = False
    for j in range(idx - 288, idx):
        if float(all_data[avail[0]][j]["time"]) >= today_start:
            count = 0
            for p in avail:
                cur = float(all_data[p][idx]["close"])
                bf = float(all_data[p][j]["close"])
                ret = abs((cur / bf - 1)) if bf > 0 else 0
                odisp += ret
                count += 1
            odisp = (odisp / count) * 10000 if count > 0 else 0
            found = True
            break
    if not found:
        odisp = 0

    # 4. Session proximity
    time_mins = hour * 60 + minute
    boundaries = [0, 420, 960, 1380]
    dist = min(abs(time_mins - b) for b in boundaries)
    session_prox = max(0, 1 - dist / 120)

    # 5. Return extremity (max |z-score|)
    ret_mean = float(np.mean(returns))
    ret_std = float(np.std(returns)) if np.std(returns) > 0 else 0.0001
    max_z = max(abs((r - ret_mean) / ret_std) for r in returns)

    # 6. Mean absolute return (overall cross-sectional tension)
    mean_abs_ret = float(np.mean([abs(r) for r in returns])) * 10000

    return {
        "dispersion": dispersion, "vol_shock": vol_shock, "odisp_bp": odisp,
        "session_prox": session_prox, "max_z": max_z, "mean_abs_ret": mean_abs_ret,
        "hour": hour, "minute": minute,
    }

def build_stsi_norms(all_data, avail, max_samples=3000):
    N = min(len(v) for v in all_data.values())
    samples = []
    step = max(1, N // max_samples)
    for idx in range(100, N - 10, step):
        s = compute_stsi_fast(all_data, avail, idx)
        if s: samples.append(s)
    norms = {}
    for key in ["dispersion", "vol_shock", "odisp_bp", "max_z", "mean_abs_ret"]:
        vals = sorted([s[key] for s in samples])
        norms[key] = vals
    return norms

def stsi_score(stsi, norms):
    """0-1 composite STSI."""
    score = 0
    for key, w in [("dispersion", 0.20), ("vol_shock", 0.15), ("odisp_bp", 0.25),
                   ("session_prox", 0.15), ("max_z", 0.10), ("mean_abs_ret", 0.15)]:
        if key == "session_prox":
            v = stsi["session_prox"]
        else:
            vals = norms[key]
            v = sum(1 for x in vals if x < stsi[key]) / len(vals) if vals else 0.5
        score += v * w
    return score

def compute_all_stsi(all_data, avail, start, end, norms):
    """Pre-compute STSI for all bars in range."""
    result = {}
    step = 1
    for idx in range(start, end, step):
        s = compute_stsi_fast(all_data, avail, idx)
        if s:
            result[idx] = {"raw": s, "score": stsi_score(s, norms)}
    return result

# ── BACKTEST FUNCTIONS (records idx) ──

def bt_hour0(all_data, avail, idx_start, idx_end, hold=3, lookback=3, top_n=3,
             max_pos=3, costs_bp=0.3, vol_filter=True):
    """Tokyo Hour 0 strategy — records trade idx."""
    N = min(len(v) for v in all_data.values() if v is not None)
    atr_window = deque(maxlen=288)
    positions = {}
    trades = []

    for idx in range(max(lookback, idx_start), min(N - hold, idx_end)):
        dt = datetime.fromtimestamp(float(all_data[avail[0]][idx]["time"]), tz=timezone.utc)
        hour = dt.hour
        if hour != 0: continue

        if vol_filter:
            atr = 0.0
            for p in avail:
                hi = float(all_data[p][idx]["high"])
                lo = float(all_data[p][idx]["low"])
                pc = float(all_data[p][idx - 1]["close"])
                tr = max(hi - lo, abs(hi - pc), abs(lo - pc))
                atr += tr / float(all_data[p][idx]["close"])
            atr /= len(avail)
            atr_window.append(atr)
            if len(atr_window) >= 30:
                if atr <= sorted(atr_window)[2 * len(atr_window) // 3]:
                    continue

        for p in list(positions.keys()):
            if idx >= positions[p]: del positions[p]
        if len(positions) >= max_pos: continue

        pair_moves = []
        for p in avail:
            if p in positions: continue
            cur = float(all_data[p][idx]["close"])
            bf = float(all_data[p][idx - lookback]["close"])
            ret = (cur / bf - 1) if bf > 0 else 0
            pair_moves.append((p, abs(ret), ret))
        pair_moves.sort(key=lambda x: x[1], reverse=True)

        for p, mag, ret in pair_moves[:top_n]:
            if p in positions or len(positions) >= max_pos: break
            if ret > 0: continue
            if idx + 1 + hold >= N: continue
            spread_cost = costs_bp / 10000
            entry = float(all_data[p][idx + 1]["open"])
            exit_ = float(all_data[p][idx + hold]["close"])
            gross = (exit_ / entry - 1) if entry > 0 else 0
            net_pnl = gross - spread_cost
            trades.append({"pnl": net_pnl * 10000, "won": net_pnl > 0, "idx": idx, "pair": p})
            positions[p] = idx + hold
    return trades

def bt_stsi(all_data, avail, idxs_with_stsi, stsi_data, hold=3, lookback=3, top_n=3,
            direction="both", max_pos=3, costs_bp=0.3, min_stsi=0):
    """STSI-based: trade at high-stress bars regardless of hour."""
    N = min(len(v) for v in all_data.values() if v is not None)
    positions = {}
    trades = []

    for idx in sorted(idxs_with_stsi):
        if stsi_data[idx]["score"] < min_stsi: continue
        if idx < lookback or idx + hold >= N: continue

        for p in list(positions.keys()):
            if idx >= positions[p]: del positions[p]
        if len(positions) >= max_pos: continue

        pair_moves = []
        for p in avail:
            if p in positions: continue
            cur = float(all_data[p][idx]["close"])
            bf = float(all_data[p][idx - lookback]["close"])
            ret = (cur / bf - 1) if bf > 0 else 0
            pair_moves.append((p, abs(ret), ret))
        pair_moves.sort(key=lambda x: x[1], reverse=True)

        for p, mag, ret in pair_moves[:top_n]:
            if p in positions or len(positions) >= max_pos: break
            if direction == "long" and ret > 0: continue
            if direction == "short" and ret < 0: continue
            dir_signal = 1 if ret < 0 else -1
            if idx + 1 + hold >= N: continue
            spread_cost = costs_bp / 10000
            entry = float(all_data[p][idx + 1]["open"])
            exit_ = float(all_data[p][idx + hold]["close"])
            gross = dir_signal * (exit_ / entry - 1) if entry > 0 else 0
            net_pnl = gross - spread_cost
            trades.append({"pnl": net_pnl * 10000, "won": net_pnl > 0, "idx": idx, "pair": p})
            positions[p] = idx + hold
    return trades

def main():
    all_data, avail = load_data()
    N = min(len(v) for v in all_data.values())
    n_days = N / 288
    split = int(N * 0.7)
    print(f"Data: {len(avail)} pairs, {N} bars, {n_days:.0f} days")
    print(f"Train: 0-{split} ({split/288:.0f}d), Test: {split}-{N} ({(N-split)/288:.0f}d)")

    # ── BUILD STSI ──
    print(f"\n{'='*70}")
    print("BUILDING STSI NORMALIZER (training set)...")
    print("=" * 70)
    norms = build_stsi_norms(all_data, avail, max_samples=3000)

    comps = ["dispersion", "vol_shock", "odisp_bp", "max_z", "mean_abs_ret"]
    print(f"\n  {'Component':>15s}  {'Mean':>8s}  {'P50':>8s}  {'P90':>8s}  {'P99':>8s}")
    print(f"  {'-'*50}")
    for key in comps:
        vals = sorted(norms[key])
        print(f"  {key:>15s}:  {float(np.mean(vals)):>7.2f}  {vals[len(vals)//2]:>7.2f}  "
              f"{vals[int(len(vals)*0.9)]:>7.2f}  {vals[int(len(vals)*0.99)]:>7.2f}")

    # ── TEST 1: STSI STRATIFICATION ──
    print(f"\n{'='*70}")
    print("TEST 1: STSI STRATIFICATION — trade outcomes by STSI level")
    print("=" * 70)

    train_stsi = compute_all_stsi(all_data, avail, 100, split, norms)
    print(f"  Computed STSI for {len(train_stsi)} training bars")

    # Get Hour 0 trades with their STSI
    train_trades = bt_hour0(all_data, avail, 100, split)
    trade_stsis = []
    for t in train_trades:
        idx = t["idx"]
        if idx in train_stsi:
            trade_stsis.append((train_stsi[idx]["score"], t["pnl"], t["won"]))

    if trade_stsis:
        scores = [ts[0] for ts in trade_stsis]
        pnls = [ts[1] for ts in trade_stsis]
        corr = np.corrcoef(scores, pnls)[0, 1] if len(scores) > 2 else 0
        print(f"\n  Hour 0 trades matched to STSI: {len(trade_stsis)}")
        print(f"  STSI-PnL correlation: {corr:+.3f}")

        # Stratify by STSI quartile
        sorted_scores = sorted(scores)
        n = len(sorted_scores)
        quartiles = [sorted_scores[min(int(n * q), n - 1)] for q in [0, 0.25, 0.5, 0.75, 1.0]]
        print(f"\n  {'STSI Quartile':>15s}  {'n':>5s}  {'WR':>6s}  {'Mean(bp)':>10s}")
        print(f"  {'-'*40}")
        for q in range(4):
            lo, hi = quartiles[q], quartiles[q+1]
            subset = [ts for ts in trade_stsis if lo <= ts[0] < hi]
            if subset:
                mu = float(np.mean([ts[1] for ts in subset]))
                wr = sum(1 for ts in subset if ts[2]) / len(subset) * 100
                print(f"  {lo:.2f}-{hi:.2f}    {len(subset):4d}  {wr:5.1f}%  {mu:>+9.2f}bp")

        # High STSI vs Low STSI
        med = float(np.median(scores))
        high = [ts for ts in trade_stsis if ts[0] >= med]
        low = [ts for ts in trade_stsis if ts[0] < med]
        for label, subset in [("Above median STSI", high), ("Below median STSI", low)]:
            if subset:
                mu = float(np.mean([ts[1] for ts in subset]))
                wr = sum(1 for ts in subset if ts[2]) / len(subset) * 100
                t = stats([{"pnl": ts[1], "won": ts[2]} for ts in subset])
                t_str = f"t={t['t_stat']:.2f}" if t else ""
                print(f"  {label:>20s}:  n={len(subset):4d}  wr={wr:5.1f}%  mean={mu:>+7.2f}bp  {t_str}")

    # ── TEST 2: PLATEAU ANALYSIS ──
    print(f"\n{'='*70}")
    print("TEST 2: PARAMETER PLATEAU (vary lookback/hold/top N)")
    print("=" * 70)

    print(f"\n  {'Lookback':>8s}  {'Hold':>5s}  {'TopN':>5s}  {'WR':>6s}  {'Mean(bp)':>10s}  {'n':>5s}  {'t':>7s}")
    print(f"  {'-'*50}")

    results = []
    for L in [1, 2, 3, 5, 8]:
        for H in [1, 2, 3, 5, 8]:
            for T in [1, 2, 3, 5]:
                trades = bt_hour0(all_data, avail, 100, split,
                                  hold=H, lookback=L, top_n=T)
                s = stats(trades)
                if s and s["n"] >= 15:
                    results.append((s["wr"], s["mean_bp"], s["n"], s["t_stat"], L, H, T))
                    if L in [2, 3, 5] and H in [2, 3, 5] and T in [1, 2, 3]:
                        print(f"  L={L:>2d}     H={H:>2d}     T={T:>2d}:   {s['wr']:5.1f}%  {s['mean_bp']:>+9.2f}bp  {s['n']:4d}  {s['t_stat']:>+6.2f}")

    if results:
        strong = [r for r in results if r[0] >= 70 and r[2] >= 15]
        good = [r for r in results if r[0] >= 65 and r[2] >= 15]
        best = max(results, key=lambda x: x[0])
        print(f"\n  Configs with WR >= 70%: {len(strong)}/{len(results)} ({len(strong)/len(results)*100:.0f}%)")
        print(f"  Configs with WR >= 65%: {len(good)}/{len(results)} ({len(good)/len(results)*100:.0f}%)")
        print(f"  Best:  L{best[4]} H{best[5]} T{best[6]} → {best[0]:.1f}% WR")
        if len(strong) >= 5:
            print(f"  → PLATEAU (multiple configs work)")
        elif len(strong) >= 2:
            print(f"  → WEAK PLATEAU (some configs work)")
        else:
            print(f"  → SPIKE (overfit risk)")

    # ── TEST 3: OOS STSI-based vs Clock-based ──
    print(f"\n{'='*70}")
    print("TEST 3: OUT-OF-SAMPLE COMPARISON (last 30% of data)")
    print("=" * 70)

    # Clock baseline (Hour 0, H3 L3 T3 long VF)
    base_test = bt_hour0(all_data, avail, split, N)
    s_base = stats(base_test)
    if s_base:
        print(f"\n  Clock-based (Hour 0, H3 L3 T3 long VF):")
        print(f"    n={s_base['n']:4d}  wr={s_base['wr']:5.1f}%  mean={s_base['mean_bp']:>+.2f}bp  t={s_base['t_stat']:+.2f}")

    # STSI-based on test set
    test_stsi = compute_all_stsi(all_data, avail, split, N - 10, norms)
    print(f"  Computed STSI for {len(test_stsi)} test bars")

    thresh_range = [0.7, 0.8, 0.85, 0.9, 0.95]
    stsi_vals = sorted([d["score"] for d in test_stsi.values()])

    for pct in thresh_range:
        thresh = stsi_vals[min(int(len(stsi_vals) * pct), len(stsi_vals) - 1)] if stsi_vals else 0.9
        high_stsi_idxs = {idx for idx, d in test_stsi.items() if d["score"] >= thresh}

        if len(high_stsi_idxs) < 10: continue

        # STSI strategy: both directions, no vol filter, any hour
        stsi_trades = bt_stsi(all_data, avail, high_stsi_idxs, test_stsi,
                              hold=3, lookback=3, top_n=3, direction="both",
                              costs_bp=0.3, min_stsi=0)
        s_stsi = stats(stsi_trades)
        if s_stsi:
            pct_str = f"STSI >{pct*100:.0f}%ile"
            print(f"\n  {pct_str:>20s} (n_signals={len(high_stsi_idxs):4d}):")
            print(f"    Trades={s_stsi['n']:4d}  wr={s_stsi['wr']:5.1f}%  mean={s_stsi['mean_bp']:>+.2f}bp  t={s_stsi['t_stat']:+.2f}")

    # Also test: STSI at Hour 0 only — does filtering by STSI improve?
    hour0_test_idxs = {idx for idx in test_stsi
                       if test_stsi[idx]["raw"]["hour"] == 0}
    if hour0_test_idxs:
        for pct in [0.5, 0.7, 0.9]:
            thresh = stsi_vals[min(int(len(stsi_vals) * pct), len(stsi_vals) - 1)]
            filtered = {idx for idx in hour0_test_idxs if test_stsi[idx]["score"] >= thresh}
            if len(filtered) < 5: continue
            ft = bt_stsi(all_data, avail, filtered, test_stsi,
                          hold=3, lookback=3, top_n=3, direction="long",
                          costs_bp=0.3, min_stsi=0)
            s_ft = stats(ft)
            if s_ft:
                print(f"\n  Hour 0 + STSI >{pct*100:.0f}%ile:")
                print(f"    n={s_ft['n']:4d}  wr={s_ft['wr']:5.1f}%  mean={s_ft['mean_bp']:>+.2f}bp  t={s_ft['t_stat']:+.2f}")

    # ── BONUS: STSI distribution by hour ──
    print(f"\n{'='*70}")
    print("BONUS: Mean STSI by hour of day (full dataset)")
    print("=" * 70)
    all_stsi = compute_all_stsi(all_data, avail, 100, N - 10, norms)
    hour_stsi = {h: [] for h in range(24)}
    for idx, d in all_stsi.items():
        h = d["raw"]["hour"]
        hour_stsi[h].append(d["score"])
    print(f"  {'Hour':>6s}  {'Mean STSI':>10s}  {'P90 STSI':>10s}  {'n_bars':>7s}")
    print(f"  {'-'*37}")
    for h in range(24):
        v = hour_stsi[h]
        if v:
            print(f"  {h:>2d}:00   {float(np.mean(v)):>8.3f}     {sorted(v)[int(len(v)*0.9)]:>8.3f}     {len(v):>5d}")

    mt5.shutdown()

if __name__ == "__main__":
    main()
