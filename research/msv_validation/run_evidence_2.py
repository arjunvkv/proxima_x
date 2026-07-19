"""Additional evidence: cost-adjusted, currency exposure, regime durability, independence.

Tests:
  A. Realistic cost model (spread+commission+slippage)
  B. Currency exposure per batch — are 3 positions truly independent?
  C. Monthly durability — does every month contribute?
  D. USD trend regime — does edge survive in USD up vs USD down?
  E. Rolling 10-trade WR stability
  F. Trade-level PnL distribution
  G. Survival test: remove best month
  H. Hourly performance profile
"""

import sys, os, numpy as np
from pathlib import Path
from datetime import datetime, timedelta, timezone
from collections import deque, Counter

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
    available = [p for p in ALL_PAIRS if p in all_data]
    return all_data, available

def stats(trades):
    if not trades or len(trades) < 3:
        return {"n":0,"wr":0,"mean_bp":0,"mean_usd":0,"t_stat":0}
    pnls = [t["pnl"] for t in trades]
    mu, s = float(np.mean(pnls)), float(np.std(pnls))
    wr = sum(1 for t in trades if t["won"]) / len(trades) * 100
    t = mu / (s / np.sqrt(len(trades))) if s > 0 else 0
    return {"n":len(trades),"wr":wr,"mean_bp":mu,"mean_usd":mu*10,"t_stat":t}

def backtest_asia(all_data, avail_pairs, costs_bp=0.5, hold=3, top_n=3, max_pos=3):
    """Asia session T3 H3 with per-trade cost."""
    N = min(len(v) for v in all_data.values() if v is not None)
    atr_window = deque(maxlen=288)
    positions = {}
    trades = []

    for idx in range(3, N - hold):
        dt = datetime.fromtimestamp(float(all_data[avail_pairs[0]][idx]["time"]), tz=timezone.utc)
        hour = dt.hour
        if hour >= 7:
            continue

        atr = 0.0
        for p in avail_pairs:
            hi = float(all_data[p][idx]["high"])
            lo = float(all_data[p][idx]["low"])
            pc = float(all_data[p][idx - 1]["close"])
            tr = max(hi - lo, abs(hi - pc), abs(lo - pc))
            atr += tr / float(all_data[p][idx]["close"])
        atr /= len(avail_pairs)
        atr_window.append(atr)
        if len(atr_window) >= 30:
            thresh = sorted(atr_window)[2 * len(atr_window) // 3]
            if atr <= thresh:
                continue

        for p in list(positions.keys()):
            if idx >= positions[p]:
                del positions[p]
        if len(positions) >= max_pos:
            continue

        pair_moves = []
        for p in avail_pairs:
            if p in positions:
                continue
            cur = float(all_data[p][idx]["close"])
            bf = float(all_data[p][idx - 3]["close"])
            ret = (cur / bf - 1) if bf > 0 else 0
            pair_moves.append((p, abs(ret), ret))

        pair_moves.sort(key=lambda x: x[1], reverse=True)

        for p, mag, ret in pair_moves[:top_n]:
            if p in positions:
                continue
            if len(positions) >= max_pos:
                break
            if ret > 0:
                continue
            direction = 1

            if idx + 1 + hold >= N:
                continue
            spread_cost = costs_bp / 10000
            entry = float(all_data[p][idx + 1]["open"])
            exit_ = float(all_data[p][idx + hold]["close"])
            gross = direction * (exit_ / entry - 1) if entry > 0 else 0
            net_pnl = gross - spread_cost
            won = net_pnl > 0

            trades.append({
                "pnl": net_pnl * 10000, "won": won, "hour": hour,
                "pair": p, "gross_pnl": gross * 10000, "dt": dt,
                "idx": idx,
            })
            positions[p] = idx + hold
    return trades

def main():
    all_data, avail_pairs = load_data()
    N = min(len(v) for v in all_data.values())
    n_days = N / 288
    print(f"Data: {len(avail_pairs)} pairs, {N} bars, {n_days:.0f} days\n")

    # ── A: COST MODEL ──
    print("=" * 70)
    print("A. REALISTIC COST MODEL (spread+commission+slippage)")
    print("=" * 70)
    print(f"  {'Cost(bp)':>10s}  {'n':>5s}  {'WR':>6s}  {'Mean(bp)':>10s}  {'$/trade':>8s}  {'t':>7s}  {'W/L':>8s}")
    print(f"  {'-'*58}")

    baseline = None
    for cost in [0, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]:
        trades = backtest_asia(all_data, avail_pairs, costs_bp=cost)
        s = stats(trades)
        w = sum(1 for t in trades if t["won"])
        l = len(trades) - w
        print(f"  {cost:>7.1f}bp  {s['n']:5d}  {s['wr']:5.1f}%  {s['mean_bp']:>+10.2f}  ${s['mean_usd']:>+6.1f}  {s['t_stat']:>+6.2f}  {w}/{l:>3d}")
        if cost == 0:
            baseline = trades
        if cost == 0.5:
            base_trades = trades

    # ── H: HOURLY PROFILE ──
    print(f"\n{'='*70}")
    print("H. HOURLY PERFORMANCE PROFILE")
    print("=" * 70)
    print(f"  {'Hour':>6s}  {'n':>5s}  {'WR':>6s}  {'Mean(bp)':>10s}")
    print(f"  {'-'*32}")
    by_hour = {}
    for t in base_trades:
        h = t["hour"]
        if h not in by_hour:
            by_hour[h] = []
        by_hour[h].append(t["pnl"])
    for h in sorted(by_hour.keys()):
        v = by_hour[h]
        mu = float(np.mean(v))
        wr = sum(1 for x in v if x > 0) / len(v) * 100
        print(f"  {h:>2d}:00   {len(v):4d}  {wr:5.1f}%  {mu:>+8.2f}bp")

    # ── C: MONTHLY DURABILITY ──
    print(f"\n{'='*70}")
    print("C. MONTHLY DURABILITY (0.5bp cost)")
    print("=" * 70)
    by_month = {}
    for t in base_trades:
        m = t["dt"].strftime("%Y-%m")
        if m not in by_month:
            by_month[m] = []
        by_month[m].append(t["pnl"])

    print(f"  {'Month':>8s}  {'n':>5s}  {'WR':>6s}  {'Mean(bp)':>10s}  {'$/trade':>8s}")
    print(f"  {'-'*42}")
    monthly_wrs = []
    for m in sorted(by_month.keys()):
        v = by_month[m]
        mu = float(np.mean(v))
        wr = sum(1 for x in v if x > 0) / len(v) * 100
        monthly_wrs.append(wr)
        print(f"  {m:>8s}:  {len(v):4d}  {wr:5.1f}%  {mu:>+10.2f}bp  ${mu*10:>+7.1f}")

    print(f"\n  Monthly WR range: {min(monthly_wrs):.1f}% - {max(monthly_wrs):.1f}%")
    print(f"  All months WR > 50%: {all(w > 50 for w in monthly_wrs)}")
    print(f"  All months WR > 55%: {all(w > 55 for w in monthly_wrs)}")

    # ── D: USD TREND REGIME ──
    print(f"\n{'='*70}")
    print("D. USD TREND REGIME (EURUSD 5d proxy)")
    print("=" * 70)
    lookback_bars = 1440  # ~5 days

    usd_up, usd_down, usd_flat = [], [], []
    eurusd = all_data["EURUSD"]

    for t in base_trades:
        idx = t["idx"]
        if idx < lookback_bars:
            usd_flat.append(t)
            continue
        eurusd_now = float(eurusd[idx]["close"])
        eurusd_before = float(eurusd[idx - lookback_bars]["close"])
        eurusd_ret = (eurusd_now / eurusd_before - 1) if eurusd_before > 0 else 0
        usd_ret = -eurusd_ret  # USD index moves opposite to EURUSD
        if usd_ret > 0.01:   # 1% USD strengthening
            usd_up.append(t)
        elif usd_ret < -0.01:  # 1% USD weakening
            usd_down.append(t)
        else:
            usd_flat.append(t)

    for label, subset in [("USD strengthening (>1%)", usd_up),
                          ("USD weakening (>1%)", usd_down),
                          ("USD flat (<1% move)", usd_flat)]:
        if subset:
            pnls = [t["pnl"] for t in subset]
            mu = float(np.mean(pnls))
            wr = sum(1 for t in subset if t["won"]) / len(subset) * 100
            nd = float(np.std(pnls))
            tstat = mu / (nd / np.sqrt(len(pnls))) if nd > 0 else 0
            print(f"  {label:>25s}:  n={len(subset):4d}  wr={wr:5.1f}%  mean={mu:>+7.2f}bp  t={tstat:+5.2f}")

    # ── E: HOURLY VOLATILITY REGIME ──
    print(f"\n{'='*70}")
    print("E. HOURLY VOLATILITY REGIME (ATR decile within Asia)")
    print("=" * 70)
    # Group by ATR decile
    # Re-run to get ATR at trade time
    N = min(len(v) for v in all_data.values())
    atr_window = deque(maxlen=288)
    trade_atrs = []
    base_idx = 0
    positions = {}
    for idx in range(3, N - 3):
        dt = datetime.fromtimestamp(float(all_data[avail_pairs[0]][idx]["time"]), tz=timezone.utc)
        hour = dt.hour
        if hour >= 7:
            continue

        atr = 0.0
        for p in avail_pairs:
            hi = float(all_data[p][idx]["high"])
            lo = float(all_data[p][idx]["low"])
            pc = float(all_data[p][idx - 1]["close"])
            tr = max(hi - lo, abs(hi - pc), abs(lo - pc))
            atr += tr / float(all_data[p][idx]["close"])
        atr /= len(avail_pairs)
        atr_window.append(atr)

        # Skip low vol check, just track
        for p in list(positions.keys()):
            if idx >= positions[p]:
                del positions[p]
        if len(positions) >= 3:
            continue

        pair_moves = []
        for p in avail_pairs:
            if p in positions:
                continue
            cur = float(all_data[p][idx]["close"])
            bf = float(all_data[p][idx - 3]["close"])
            ret = (cur / bf - 1) if bf > 0 else 0
            pair_moves.append((p, abs(ret), ret))

        pair_moves.sort(key=lambda x: x[1], reverse=True)

        for p, mag, ret in pair_moves[:3]:
            if p in positions:
                continue
            if len(positions) >= 3:
                break
            if ret > 0:
                continue
            if len(atr_window) >= 10:
                trade_atrs.append((atr, sum(atr_window) / len(atr_window), idx))
            positions[p] = idx + 3

    if trade_atrs:
        atr_vals = [t[0] for t in trade_atrs]
        rank_atr = sorted(atr_vals)
        decile_trades = {i: [] for i in range(10)}
        for t_val, avg_atr, idx in trade_atrs:
            rank = sum(1 for x in rank_atr if x < t_val) / len(rank_atr)
            decile = min(int(rank * 10), 9)
            # Find pnl for this trade
            pnl = 0
            for bt in base_trades:
                if bt["idx"] == idx:
                    pnl = bt["pnl"]
                    break
            decile_trades[decile].append(pnl)

        print(f"  {'ATR Decile':>12s}  {'n':>5s}  {'WR':>6s}  {'Mean(bp)':>10s}")
        print(f"  {'-'*35}")
        for d in range(10):
            v = decile_trades[d]
            if v:
                mu = float(np.mean(v))
                wr = sum(1 for x in v if x > 0) / len(v) * 100
                print(f"  {d*10:>3d}-{(d+1)*10:>3d}%ile    {len(v):4d}  {wr:5.1f}%  {mu:>+8.2f}bp")

    # ── F: TRADE DISTRIBUTION ──
    print(f"\n{'='*70}")
    print("F. TRADE-LEVEL DISTRIBUTION (0.5bp cost)")
    print("=" * 70)
    pnls = [t["pnl"] for t in base_trades]
    if pnls:
        pnls_sorted = sorted(pnls)
        print(f"  {'Percentile':>12s}  {'PnL(bp)':>10s}")
        print(f"  {'-'*24}")
        for pct in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
            val = pnls_sorted[int(len(pnls_sorted) * pct / 100)]
            print(f"  {pct:>3d}th          {val:>+9.2f}bp")

        winners = [x for x in pnls if x > 0]
        losers = [x for x in pnls if x <= 0]
        if winners and losers:
            avg_w = float(np.mean(winners))
            avg_l = float(np.mean(losers))
            pf = abs(avg_w / avg_l) if avg_l != 0 else float('inf')
            print(f"\n  Avg winner:  {avg_w:+.2f}bp")
            print(f"  Avg loser:   {avg_l:+.2f}bp")
            print(f"  Profit factor: {pf:.2f}")
            print(f"  Hit rate: {len(winners)/len(pnls)*100:.1f}%")
            print(f"  Expected value: {float(np.mean(pnls)):+.2f}bp ({float(np.mean(pnls))*10:+.2f}$)")

    # ── G: SURVIVAL TEST ──
    print(f"\n{'='*70}")
    print("G. SURVIVAL TESTS")
    print("=" * 70)

    # G1: Remove best month
    if by_month:
        month_means = {m: float(np.mean(v)) for m, v in by_month.items()}
        best_m = max(month_means, key=month_means.get)
        worst_m = min(month_means, key=month_means.get)
        print(f"  Best month:  {best_m} ({month_means[best_m]:+.2f}bp, n={len(by_month[best_m])})")
        print(f"  Worst month: {worst_m} ({month_means[worst_m]:+.2f}bp, n={len(by_month[worst_m])})")

        # Without best month
        all_ex_best = [v for m, vs in by_month.items() if m != best_m for v in vs]
        mu_best = float(np.mean(all_ex_best))
        wr_best = sum(1 for x in all_ex_best if x > 0) / len(all_ex_best) * 100
        tstat_best = mu_best / (float(np.std(all_ex_best)) / np.sqrt(len(all_ex_best))) if np.std(all_ex_best) > 0 else 0
        print(f"\n  Without best month:  n={len(all_ex_best):4d}  wr={wr_best:5.1f}%  mean={mu_best:>+7.2f}bp  t={tstat_best:+5.2f}")

        # Without best AND worst
        all_ex_both = [v for m, vs in by_month.items() if m not in (best_m, worst_m) for v in vs]
        if all_ex_both:
            mu_both = float(np.mean(all_ex_both))
            wr_both = sum(1 for x in all_ex_both if x > 0) / len(all_ex_both) * 100
            tstat_both = mu_both / (float(np.std(all_ex_both)) / np.sqrt(len(all_ex_both))) if np.std(all_ex_both) > 0 else 0
            print(f"  Without best+worst:  n={len(all_ex_both):4d}  wr={wr_both:5.1f}%  mean={mu_both:>+7.2f}bp  t={tstat_both:+5.2f}")

        # Jackknife: remove each month, report min WR
        jackknife_wrs = []
        for excluded in by_month:
            remaining = [v for m, vs in by_month.items() if m != excluded for v in vs]
            wr_j = sum(1 for x in remaining if x > 0) / len(remaining) * 100
            jackknife_wrs.append(wr_j)
        print(f"\n  Jackknife WR range: {min(jackknife_wrs):.1f}% - {max(jackknife_wrs):.1f}%")
        print(f"  Minimum jackknife WR: {min(jackknife_wrs):.1f}%")
        print(f"  WR degrades by at most: {max(jackknife_wrs) - min(jackknife_wrs):.1f}pp")

    # ── B: CURRENCY EXPOSURE ──
    print(f"\n{'='*70}")
    print("B. CURRENCY EXPOSURE (3 concurrent positions analysis)")
    print("=" * 70)
    # Re-run tracking concurrent positions
    N = min(len(v) for v in all_data.values())
    atr_window = deque(maxlen=288)
    positions = {}
    batch_snapshots = []
    full_capacity_batches = []

    for idx in range(3, N - 3):
        dt = datetime.fromtimestamp(float(all_data[avail_pairs[0]][idx]["time"]), tz=timezone.utc)
        hour = dt.hour
        if hour >= 7:
            continue
        atr = 0.0
        for p in avail_pairs:
            hi = float(all_data[p][idx]["high"])
            lo = float(all_data[p][idx]["low"])
            pc = float(all_data[p][idx - 1]["close"])
            tr = max(hi - lo, abs(hi - pc), abs(lo - pc))
            atr += tr / float(all_data[p][idx]["close"])
        atr /= len(avail_pairs)
        atr_window.append(atr)
        if len(atr_window) >= 30:
            if atr <= sorted(atr_window)[2 * len(atr_window) // 3]:
                continue
        for p in list(positions.keys()):
            if idx >= positions[p]:
                del positions[p]

        # Snapshot current currency exposure
        if len(positions) >= 2:
            currencies = {}
            for pair in positions:
                base, quote = pair[:3], pair[3:]
                currencies[base] = currencies.get(base, 0) + 1
                currencies[quote] = currencies.get(quote, 0) - 1
            net = {c: v for c, v in currencies.items() if v != 0}
            if net:
                batch_snapshots.append(net)
            if len(positions) >= 3:
                full_capacity_batches.append(net)

        if len(positions) >= 3:
            continue
        pair_moves = []
        for p in avail_pairs:
            if p in positions:
                continue
            cur = float(all_data[p][idx]["close"])
            bf = float(all_data[p][idx - 3]["close"])
            ret = (cur / bf - 1) if bf > 0 else 0
            pair_moves.append((p, abs(ret), ret))
        pair_moves.sort(key=lambda x: x[1], reverse=True)
        for p, mag, ret in pair_moves[:3]:
            if p in positions:
                continue
            if len(positions) >= 3:
                break
            if ret > 0:
                continue
            positions[p] = idx + 3

    if batch_snapshots:
        # Count all currency exposures
        all_curr_pairs = []
        directional_clusters = 0
        for exp in batch_snapshots:
            items = [(c, v) for c, v in exp.items()]
            for c, v in items:
                label = f"{c}{'+' if v > 0 else '-'}{abs(v)}"
                all_curr_pairs.append(label)

        common = Counter(all_curr_pairs).most_common(10)
        print(f"  Batches with 2+ concurrent positions: {len(batch_snapshots)}")
        print(f"  Batches at full 3-position capacity: {len(full_capacity_batches)}")
        print(f"\n  Most common net currency exposures:")
        for pair, cnt in common:
            print(f"    {pair:>10s}: {cnt:4d} ({cnt/len(batch_snapshots)*100:.1f}%)")

        # USD concentration
        usd_batches = [(i, e) for i, e in enumerate(batch_snapshots) if "USD" in e]
        only_usd = [(i, e) for i, e in enumerate(batch_snapshots) if len(e) == 1 and "USD" in e]
        single_ccy_bets = [(i, e) for i, e in enumerate(batch_snapshots) if len(e) == 1]
        print(f"\n  Batches with USD exposure: {len(usd_batches)}/{len(batch_snapshots)} ({len(usd_batches)/len(batch_snapshots)*100:.1f}%)")
        print(f"  Batches with ONLY USD exposure: {len(only_usd)}/{len(batch_snapshots)} ({len(only_usd)/len(batch_snapshots)*100:.1f}%)")
        print(f"  Batches with single-currency net bet: {len(single_ccy_bets)}/{len(batch_snapshots)} ({len(single_ccy_bets)/len(batch_snapshots)*100:.1f}%)")

    # ── I: PAIR CONCENTRATION ──
    print(f"\n{'='*70}")
    print("I. PAIR SELECTION DISTRIBUTION")
    print("=" * 70)
    pair_counts = Counter(t["pair"] for t in base_trades)
    pair_pnls = {}
    for t in base_trades:
        p = t["pair"]
        if p not in pair_pnls:
            pair_pnls[p] = []
        pair_pnls[p].append(t["pnl"])

    print(f"  {'Pair':>8s}  {'n':>5s}  {'WR':>6s}  {'Mean(bp)':>10s}  {'$/trade':>8s}")
    print(f"  {'-'*42}")
    for p, cnt in pair_counts.most_common():
        v = pair_pnls[p]
        mu = float(np.mean(v))
        wr = sum(1 for x in v if x > 0) / len(v) * 100
        print(f"  {p:>8s}:  {cnt:4d}  {wr:5.1f}%  {mu:>+10.2f}bp  ${mu*10:>+7.1f}")

    # ── J: ROLLING 20-TRADE WR ──
    print(f"\n{'='*70}")
    print("J. ROLLING 20-TRADE WR STABILITY")
    print("=" * 70)
    pnls = [t["pnl"] for t in base_trades]
    window = 20
    rolling_wr = []
    for i in range(len(pnls) - window):
        chunk = pnls[i:i+window]
        rolling_wr.append(sum(1 for x in chunk if x > 0) / window * 100)

    if rolling_wr:
        print(f"  Rolling {window}-trade WR (sampled every 5th point to reduce noise):")
        sampled = rolling_wr[::5]
        print(f"    Mean:       {np.mean(rolling_wr):.1f}%")
        print(f"    Median:     {np.median(rolling_wr):.1f}%")
        print(f"    Min:        {min(rolling_wr):.1f}%")
        print(f"    Max:        {max(rolling_wr):.1f}%")
        print(f"    Std:        {np.std(rolling_wr):.1f}%")
        print(f"    % below 50%: {sum(1 for x in rolling_wr if x < 50)/len(rolling_wr)*100:.1f}%")
        print(f"    % below 40%: {sum(1 for x in rolling_wr if x < 40)/len(rolling_wr)*100:.1f}%")
        print(f"    % above 60%: {sum(1 for x in rolling_wr if x >= 60)/len(rolling_wr)*100:.1f}%")
        print(f"    % above 70%: {sum(1 for x in rolling_wr if x >= 70)/len(rolling_wr)*100:.1f}%")

    # ── SUMMARY ──
    print(f"\n{'='*70}")
    print("EVIDENCE SUMMARY")
    print("=" * 70)

    s = stats(base_trades)
    monthly_all_pos = all(w > 50 for w in monthly_wrs) if monthly_wrs else False
    survival = "PASS" if monthly_all_pos else "WARN"
    cost_survival = "PASS" if s["mean_bp"] > 0 else "FAIL"

    # USD regime check
    usd_robust = True
    for subset, label in [(usd_up, "USD up"), (usd_down, "USD down"), (usd_flat, "USD flat")]:
        if len(subset) >= 10:
            mu = float(np.mean([t["pnl"] for t in subset]))
            if mu <= 0:
                usd_robust = False
    usd_status = "PASS" if usd_robust else "WARN"

    rolling_ok = "PASS" if rolling_wr and np.mean(rolling_wr) > 50 else "WARN"

    print(f"""
  {'Metric':<30s} {'Value'}
  {'-'*55}
  Win rate (0.5bp cost)          {s['wr']:.1f}%
  Mean per trade                 {s['mean_bp']:+.2f}bp  (${s['mean_usd']:+.1f})
  Trades/day                     {s['n']/n_days:.1f}
  t-stat                         {s['t_stat']:+.2f}

  {'Survival & Robustness':<30s}
  Monthly consistency            {survival:<8s}  {'All months WR>50%' if monthly_all_pos else 'At least one month WR<50%'}
  Cost survival (1.0bp)          {cost_survival:<8s}
  USD regime survival            {usd_status:<8s}
  Rolling WR stability           {rolling_ok:<8s}  Mean rolling WR: {np.mean(rolling_wr):.1f}%

  {'Risk':<30s}
  Monthly WR range               {min(monthly_wrs):.1f}% - {max(monthly_wrs):.1f}%
  Max jackknife WR degradation   {max(jackknife_wrs) - min(jackknife_wrs):.1f}pp
  Rolling WR min                  {min(rolling_wr):.1f}%
  Rolling WR below 50%           {sum(1 for x in rolling_wr if x < 50)/len(rolling_wr)*100:.1f}% of windows
  Full-currency-bet batches      {len(single_ccy_bets)}/{len(batch_snapshots)} ({len(single_ccy_bets)/len(batch_snapshots)*100:.1f}%)
  Profit factor                  {abs(np.mean(winners)/np.mean(losers)):.2f}

  {'Verdict':<30s}
  Shadow readiness:              {'YES' if survival == 'PASS' and cost_survival == 'PASS' else 'MORE DATA NEEDED'}
  Capital deployment:            NO — need 12+ months data
""")

    mt5.shutdown()

if __name__ == "__main__":
    main()
