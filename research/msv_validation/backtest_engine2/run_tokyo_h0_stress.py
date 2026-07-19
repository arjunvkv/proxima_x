"""
Tokyo Hour 0 Deep Stress Test.
Replicates bt_hour0() and tests: replication, pair breakdown, rolling WR,
month split, day-of-week, cost sensitivity, drawdown, spread sensitivity.
"""
import sys, os, time, numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime, timedelta, timezone
from collections import deque, Counter
import warnings
warnings.filterwarnings('ignore')

TICK_DIR = Path(__file__).resolve().parents[3] / 'data' / 'exness_ticks'
TICK_PAIRS = ['EURJPY', 'GBPJPY', 'EURUSD']
MONTHS = [(2025, 12), (2025, 11), (2025, 10)]
SCALE = {'EURJPY': 100, 'GBPJPY': 100, 'EURUSD': 10000}
HOLD = 3
LOOKBACK = 3
TOP_N = 3
MAX_POS = 3
COSTS_BP = 0.3

np.random.seed(42)

# ====================================================================
# MT5 DATA LOADING (120 days, 15 pairs)
# ====================================================================
def load_mt5_data():
    print("=" * 70)
    print("MT5 DATA LOADING (120 days, 15 pairs)")
    print("=" * 70)
    t0 = time.time()
    try:
        import MetaTrader5 as mt5
        if not mt5.initialize():
            print("  MT5 init FAILED — skipping MT5 tests")
            return None, []
    except Exception as e:
        print(f"  MT5 import failed: {e}")
        return None, []

    project_root = str(Path(__file__).resolve().parents[3])
    cd_root = os.path.join(project_root, "currency_decomposition")
    sys.path.insert(0, cd_root)
    from config.settings import BASE_CURRENCY_MAP
    all_pairs = list(BASE_CURRENCY_MAP.keys())[:15]

    end = datetime.now()
    start = end - timedelta(days=120)
    all_data = {}
    for pair in all_pairs:
        rates = mt5.copy_rates_range(pair, mt5.TIMEFRAME_M5, start, end)
        if rates is not None and len(rates) > 0:
            all_data[pair] = rates
    available = [p for p in all_pairs if p in all_data]
    N = min(len(v) for v in all_data.values()) if all_data else 0
    print(f"  Pairs: {len(available)}, Bars: {N:,d} ({N/288:.0f} days, {time.time()-t0:.1f}s)")
    mt5.shutdown()
    return all_data, available


# ====================================================================
# EXNESS TICK DATA → M5 BARS
# ====================================================================
def load_exness_m5():
    print("\n" + "=" * 70)
    print("EXNESS TICK → M5 BUILDING (3 pairs, Oct-Dec 2025)")
    print("=" * 70)
    t0 = time.time()
    tick_data = {}
    for pair in TICK_PAIRS:
        dfs = []
        for year, month in MONTHS:
            fn = TICK_DIR / f'{pair}_Raw_Spread_{year}_{month:02d}.zip'
            if not fn.exists(): continue
            df = pd.read_csv(fn, compression='zip',
                names=['Exness', 'Symbol', 'Timestamp', 'Bid', 'Ask'],
                skiprows=1, header=None, on_bad_lines='skip',
                dtype={'Exness': str, 'Symbol': str, 'Timestamp': str,
                       'Bid': np.float64, 'Ask': np.float64})
            df['Timestamp'] = df['Timestamp'].str.replace('Z', '', regex=False)
            df['Timestamp'] = pd.to_datetime(df['Timestamp'],
                format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
            df = df.dropna(subset=['Timestamp'])
            dfs.append(df)
        tick_data[pair] = pd.concat(dfs, ignore_index=True)
        tick_data[pair] = tick_data[pair].sort_values('Timestamp').reset_index(drop=True)

    m5_dict = {}
    for pair in TICK_PAIRS:
        s = SCALE[pair]
        df = tick_data[pair].copy()
        df['Mid'] = (df['Bid'] + df['Ask']) / 2
        df['Spread_pips'] = (df['Ask'] - df['Bid']) * s
        df = df.set_index('Timestamp')
        ohlc = df['Mid'].resample('5min').ohlc()
        tick_count = df['Mid'].resample('5min').count()
        med_spread = df['Spread_pips'].resample('5min').median()
        max_spread = df['Spread_pips'].resample('5min').max()
        bars = pd.DataFrame({
            'open': ohlc['open'], 'high': ohlc['high'],
            'low': ohlc['low'], 'close': ohlc['close'],
            'tick_count': tick_count, 'med_spread': med_spread,
            'max_spread': max_spread,
        })
        bars = bars.dropna(subset=['open', 'close', 'med_spread'])
        bars['hour'] = bars.index.hour
        bars['minute'] = bars.index.minute
        bars['dow'] = bars.index.dayofweek
        bars['date'] = bars.index.date
        bars['ret'] = (bars['close'] - bars['open']) * s
        bars['roll_spread_med'] = bars['med_spread'].rolling(12).median()
        bars['roll_vol'] = bars['ret'].rolling(12).std()
        bars['spread_z'] = ((bars['med_spread'] - bars['roll_spread_med']) /
                            bars['roll_spread_med'].clip(1e-8))
        m5_dict[pair] = bars

    N = min(len(v) for v in m5_dict.values())
    print(f"  Bars per pair: ~{N:,d} ({N/288:.0f} days, {time.time()-t0:.1f}s)")
    for p in TICK_PAIRS:
        print(f"    {p}: {len(m5_dict[p]):,d} bars, "
              f"spread_med={m5_dict[p]['med_spread'].median():.2f}p, "
              f"ticks/bar={m5_dict[p]['tick_count'].median():.0f}")
    return m5_dict


# ====================================================================
# CORE STRATEGY — exact bt_hour0() replication
# ====================================================================
def bt_hour0(all_data, avail_pairs, costs_bp=COSTS_BP, hold=HOLD,
             lookback=LOOKBACK, top_n=TOP_N, max_pos=MAX_POS, vol_filter=False):
    N = min(len(v) for v in all_data.values() if v is not None)
    positions = {}
    trades = []

    for idx in range(max(lookback, 0), min(N - hold, len(next(iter(all_data.values()))))):
        dt = datetime.fromtimestamp(float(all_data[avail_pairs[0]][idx]["time"]), tz=timezone.utc)
        if dt.hour != 0 or dt.minute != 0:
            continue

        if vol_filter:
            atr_window = deque(maxlen=288)
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
        if len(positions) >= max_pos:
            continue

        pair_moves = []
        for p in avail_pairs:
            if p in positions:
                continue
            cur = float(all_data[p][idx]["close"])
            bf = float(all_data[p][idx - lookback]["close"])
            ret = (cur / bf - 1) if bf > 0 else 0
            pair_moves.append((p, abs(ret), ret))
        pair_moves.sort(key=lambda x: x[1], reverse=True)

        for p, mag, ret in pair_moves[:top_n]:
            if p in positions or len(positions) >= max_pos:
                break
            if ret > 0:
                continue
            if idx + 1 + hold >= N:
                continue
            spread_cost = costs_bp / 10000
            entry = float(all_data[p][idx + 1]["open"])
            exit_ = float(all_data[p][idx + hold]["close"])
            gross = (exit_ / entry - 1) if entry > 0 else 0
            net_pnl = gross - spread_cost
            trades.append({
                "pnl": net_pnl * 10000, "won": net_pnl > 0,
                "idx": idx, "pair": p, "dt": dt,
                "gross_bp": gross * 10000,
            })
            positions[p] = idx + hold
    return trades


# ====================================================================
# EXNESS ADAPTED STRATEGY (uses DataFrame instead of mt5 rates)
# ====================================================================
def bt_hour0_exness(m5_dict, costs_bp=COSTS_BP, hold=HOLD,
                    lookback=LOOKBACK, top_n=TOP_N, max_pos=MAX_POS, vol_filter=False):
    avail_pairs = list(m5_dict.keys())
    N = min(len(v) for v in m5_dict.values())
    positions = {}
    trades = []

    for i in range(max(lookback, 0), N - hold):
        ref_pair = avail_pairs[0]
        bar = m5_dict[ref_pair].iloc[i]
        dt = bar.name.to_pydatetime().replace(tzinfo=timezone.utc)
        if bar['hour'] != 0 or bar['minute'] != 0:
            continue

        if vol_filter:
            atr_window = deque(maxlen=288)
            atr_vals = []
            for p in avail_pairs:
                row = m5_dict[p].iloc[i]
                hi = row['high']
                lo = row['low']
                pc = m5_dict[p].iloc[i - 1]['close']
                tr = max(hi - lo, abs(hi - pc), abs(lo - pc))
                atr_vals.append(tr / row['close'])
            atr = np.mean(atr_vals) if atr_vals else 0
            atr_window.append(atr)
            if len(atr_window) >= 30:
                if atr <= sorted(atr_window)[2 * len(atr_window) // 3]:
                    continue

        for p in list(positions.keys()):
            if i >= positions[p]:
                del positions[p]
        if len(positions) >= max_pos:
            continue

        pair_moves = []
        for p in avail_pairs:
            if p in positions:
                continue
            cur = m5_dict[p].iloc[i]['close']
            bf = m5_dict[p].iloc[i - lookback]['close']
            ret = (cur / bf - 1) if bf > 0 else 0
            pair_moves.append((p, abs(ret), ret))
        pair_moves.sort(key=lambda x: x[1], reverse=True)

        for p, mag, ret in pair_moves[:top_n]:
            if p in positions or len(positions) >= max_pos:
                break
            if ret > 0:
                continue
            if i + 1 + hold >= N:
                continue
            spread_cost = costs_bp / 10000
            entry = m5_dict[p].iloc[i + 1]['open']
            exit_ = m5_dict[p].iloc[i + hold]['close']
            gross = (exit_ / entry - 1) if entry > 0 else 0
            net_pnl = gross - spread_cost

            entry_bar = m5_dict[p].iloc[i + 1]
            sp = entry_bar['med_spread'] if not np.isnan(entry_bar['med_spread']) else 0

            trades.append({
                "pnl": net_pnl * 10000, "won": net_pnl > 0,
                "idx": i, "pair": p, "dt": dt,
                "gross_bp": gross * 10000,
                "entry_spread": sp,
                "entry_tick_count": entry_bar['tick_count'],
                "entry_vol": m5_dict[p].iloc[i]['roll_vol'],
                "spread_z": m5_dict[p].iloc[i]['spread_z'],
            })
            positions[p] = i + hold
    return trades


# ====================================================================
# STRESS TESTS
# ====================================================================

def pair_breakdown(trades):
    print(f"\n{'='*70}")
    print("STRESS TEST A: PAIR BREAKDOWN")
    print("=" * 70)
    by_pair = {}
    for t in trades:
        by_pair.setdefault(t["pair"], []).append(t)
    print(f"  {'Pair':<10s} {'n':>5s} {'WR':>7s} {'Mean(bp)':>10s} {'Gross(bp)':>10s}")
    print(f"  {'-'*45}")
    for pair in sorted(by_pair.keys()):
        v = by_pair[pair]
        pnls = [t["pnl"] for t in v]
        gross = [t["gross_bp"] for t in v]
        mu = float(np.mean(pnls))
        wr = sum(1 for x in pnls if x > 0) / len(pnls) * 100
        print(f"  {pair:<10s} {len(v):>5d}  {wr:>6.1f}%  {mu:>+9.2f}  {np.mean(gross):>+9.2f}")
    return by_pair


def rolling_wr(trades, window=100):
    print(f"\n{'='*70}")
    print(f"STRESS TEST B: ROLLING {window}-TRADE WR")
    print("=" * 70)
    pnls = [t["pnl"] for t in trades]
    if len(pnls) < window + 10:
        print(f"  Too few trades ({len(pnls)}), skipping")
        return
    wrs = []
    for i in range(len(pnls) - window + 1):
        wrs.append(sum(1 for x in pnls[i:i+window] if x > 0) / window * 100)
    min_wr, max_wr = min(wrs), max(wrs)
    below_60 = sum(1 for w in wrs if w < 60) / len(wrs) * 100
    below_50 = sum(1 for w in wrs if w < 50) / len(wrs) * 100
    print(f"  N={len(trades)}, Window={window}")
    print(f"  Rolling WR: {min_wr:.1f}% – {max_wr:.1f}% (mean={np.mean(wrs):.1f}%)")
    print(f"  % windows below 60%: {below_60:.1f}%")
    print(f"  % windows below 50%: {below_50:.1f}%")
    return wrs


def month_breakdown(trades):
    print(f"\n{'='*70}")
    print("STRESS TEST C: MONTHLY DURABILITY")
    print("=" * 70)
    by_month = {}
    for t in trades:
        m = t["dt"].strftime("%Y-%m")
        by_month.setdefault(m, []).append(t)
    months_sorted = sorted(by_month.keys())
    print(f"  {'Month':>8s}  {'n':>4s}  {'WR':>6s}  {'Mean(bp)':>10s}")
    print(f"  {'-'*34}")
    wrs = []
    for m in months_sorted:
        v = by_month[m]
        pnls = [t["pnl"] for t in v]
        mu = float(np.mean(pnls))
        wr = sum(1 for x in pnls if x > 0) / len(pnls) * 100
        wrs.append(wr)
        print(f"  {m:>8s}:  {len(v):>4d}  {wr:>5.1f}%  {mu:>+9.2f}bp")
    if wrs:
        print(f"\n  Range: {min(wrs):.1f}% – {max(wrs):.1f}%")
        print(f"  All > 50%: {all(w > 50 for w in wrs)}")
        print(f"  All > 60%: {all(w > 60 for w in wrs)}")
    return by_month


def day_of_week(trades):
    print(f"\n{'='*70}")
    print("STRESS TEST D: DAY-OF-WEEK")
    print("=" * 70)
    dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    by_dow = {}
    for t in trades:
        d = t["dt"].weekday()
        by_dow.setdefault(d, []).append(t)
    print(f"  {'Day':<12s} {'n':>4s}  {'WR':>6s}  {'Mean(bp)':>10s}")
    print(f"  {'-'*36}")
    for d in sorted(by_dow.keys()):
        v = by_dow[d]
        pnls = [t["pnl"] for t in v]
        mu = float(np.mean(pnls))
        wr = sum(1 for x in pnls if x > 0) / len(pnls) * 100
        print(f"  {dow_names[d]:<12s} {len(v):>4d}  {wr:>5.1f}%  {mu:>+9.2f}bp")
    return by_dow


def cost_sensitivity(all_data, avail_pairs):
    print(f"\n{'='*70}")
    print("STRESS TEST E: COST SENSITIVITY")
    print("=" * 70)
    print(f"  {'Cost(bp)':>10s}  {'n':>5s}  {'WR':>7s}  {'Mean(bp)':>10s}  {'Gross(bp)':>10s}")
    print(f"  {'-'*47}")
    for cost in [0, 0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]:
        trades = bt_hour0(all_data, avail_pairs, costs_bp=cost)
        if not trades:
            continue
        pnls = [t["pnl"] for t in trades]
        gross = [t["gross_bp"] for t in trades]
        mu = float(np.mean(pnls))
        wr = sum(1 for x in pnls if x > 0) / len(pnls) * 100
        print(f"  {cost:>7.1f}bp  {len(trades):>5d}  {wr:>6.1f}%  {mu:>+9.2f}bp  {np.mean(gross):>+9.2f}bp")
    return {}


def max_drawdown(trades):
    print(f"\n{'='*70}")
    print("STRESS TEST F: MAX DRAWDOWN & CONSECUTIVE LOSSES")
    print("=" * 70)
    pnls = [t["pnl"] for t in trades]
    eq = np.cumsum(pnls)
    running_max = np.maximum.accumulate(eq)
    dd = eq - running_max
    max_dd = float(np.min(dd))
    max_dd_idx = int(np.argmin(dd))
    max_consec_loss = 0
    cur_loss = 0
    for p in pnls:
        if p <= 0:
            cur_loss += 1
            max_consec_loss = max(max_consec_loss, cur_loss)
        else:
            cur_loss = 0
    max_consec_win = 0
    cur_win = 0
    for p in pnls:
        if p > 0:
            cur_win += 1
            max_consec_win = max(max_consec_win, cur_win)
        else:
            cur_win = 0
    print(f"  Max drawdown: {max_dd:.2f}bp (at trade {max_dd_idx})")
    print(f"  Max consec losses: {max_consec_loss}")
    print(f"  Max consec wins: {max_consec_win}")
    return {"max_dd": max_dd, "max_consec_loss": max_consec_loss, "max_consec_win": max_consec_win}


def pair_removal_test(all_data, avail_pairs, base_trades):
    print(f"\n{'='*70}")
    print("STRESS TEST G: PAIR REMOVAL (leave-one-out)")
    print("=" * 70)
    by_pair = {}
    for t in base_trades:
        by_pair.setdefault(t["pair"], []).append(t["pnl"])
    pair_wrs = {p: (sum(1 for x in v if x > 0) / len(v) * 100, np.mean(v))
                for p, v in by_pair.items()}
    best_pair = max(pair_wrs, key=lambda p: pair_wrs[p][0])
    worst_pair = min(pair_wrs, key=lambda p: pair_wrs[p][0])
    print(f"  Base ({len(avail_pairs)} pairs): n={len(base_trades)}, "
          f"WR={sum(1 for t in base_trades if t['won'])/len(base_trades)*100:.1f}%")
    print(f"  Best pair: {best_pair} ({pair_wrs[best_pair][0]:.1f}%, "
          f"{pair_wrs[best_pair][1]:+.2f}bp)")
    print(f"  Worst pair: {worst_pair} ({pair_wrs[worst_pair][0]:.1f}%, "
          f"{pair_wrs[worst_pair][1]:+.2f}bp)")
    for remove_pair in [best_pair, worst_pair]:
        reduced_pairs = [p for p in avail_pairs if p != remove_pair]
        trades = bt_hour0(all_data, reduced_pairs, costs_bp=COSTS_BP)
        if not trades:
            continue
        pnls = [t["pnl"] for t in trades]
        wr = sum(1 for x in pnls if x > 0) / len(pnls) * 100
        print(f"  Remove {remove_pair:<8s}: n={len(trades):>4d}, WR={wr:>5.1f}%, "
              f"Mean={np.mean(pnls):+.2f}bp")
    return {}


def spread_sensitivity(trades):
    print(f"\n{'='*70}")
    print("STRESS TEST H: SPREAD SENSITIVITY (Exness tick data only)")
    print("=" * 70)
    if not trades:
        print("  No trades, skipping")
        return
    if "entry_spread" not in trades[0]:
        print("  No spread data, skipping")
        return

    sp = np.array([t["entry_spread"] for t in trades])
    sp_z = np.array([t["spread_z"] for t in trades])
    pnls = np.array([t["pnl"] for t in trades])

    median_sp = np.median(sp)

    print(f"  Median entry spread: {median_sp:.2f}p")
    for thresh, label in [(median_sp, f'Above median ({median_sp:.1f}p)'),
                          (median_sp * 2, f'Above 2× median ({2*median_sp:.1f}p)'),
                          (median_sp * 3, f'Above 3× median ({3*median_sp:.1f}p)')]:
        mask = sp >= thresh
        n = int(mask.sum())
        if n < 5:
            print(f"  {label:<40s}: n={n}, too few")
            continue
        wr = np.mean(pnls[mask] > 0) * 100
        mu = np.mean(pnls[mask])
        print(f"  {label:<40s}: n={n:>4d}, WR={wr:>5.1f}%, Mean={mu:+.2f}bp")

    print(f"\n  By spread_z quartile:")
    for q, label in [(0.25, 'Lowest 25%'), (0.5, '25-50%'), (0.75, '50-75%'), (1.0, 'Top 25%')]:
        if q == 0.25:
            mask = sp_z <= np.percentile(sp_z[~np.isnan(sp_z)], 25)
        elif q == 0.5:
            mask = ((sp_z > np.percentile(sp_z[~np.isnan(sp_z)], 25)) &
                    (sp_z <= np.percentile(sp_z[~np.isnan(sp_z)], 50)))
        elif q == 0.75:
            mask = ((sp_z > np.percentile(sp_z[~np.isnan(sp_z)], 50)) &
                    (sp_z <= np.percentile(sp_z[~np.isnan(sp_z)], 75)))
        else:
            mask = sp_z > np.percentile(sp_z[~np.isnan(sp_z)], 75)
        n = int(mask.sum())
        if n < 5: continue
        wr = np.mean(pnls[mask] > 0) * 100
        mu = np.mean(pnls[mask])
        print(f"    {label:<20s}: n={n:>4d}, WR={wr:>5.1f}%, Mean={mu:+.2f}bp")
    return {}


def vol_regime_sensitivity(trades):
    print(f"\n{'='*70}")
    print("STRESS TEST I: VOL REGIME SENSITIVITY (Exness only)")
    print("=" * 70)
    if not trades or "entry_vol" not in trades[0]:
        print("  No vol data, skipping")
        return
    vols = np.array([t["entry_vol"] for t in trades])
    pnls = np.array([t["pnl"] for t in trades])
    valid = ~np.isnan(vols)
    if valid.sum() < 20:
        print(f"  Too few valid vol entries ({valid.sum()}), skipping")
        return
    vols = vols[valid]
    pnls = pnls[valid]
    thresh_low = np.percentile(vols, 33)
    thresh_high = np.percentile(vols, 67)
    for label, mask in [("Low vol (bottom 33%)", vols <= thresh_low),
                        ("Mid vol (33-67%)", (vols > thresh_low) & (vols <= thresh_high)),
                        ("High vol (top 33%)", vols > thresh_high)]:
        n = int(mask.sum())
        if n < 5: continue
        wr = np.mean(pnls[mask] > 0) * 100
        mu = np.mean(pnls[mask])
        print(f"  {label:<30s}: n={n:>4d}, WR={wr:>5.1f}%, Mean={mu:+.2f}bp")
    return {}


def basic_stats(trades, label):
    if not trades:
        print(f"  {label}: 0 trades")
        return
    pnls = [t["pnl"] for t in trades]
    n = len(pnls)
    wr = sum(1 for x in pnls if x > 0) / n * 100
    mu = float(np.mean(pnls))
    med = float(np.median(pnls))
    std = float(np.std(pnls))
    sharpe = mu / std if std > 0 else 0
    t_stat = mu / (std / np.sqrt(n)) if std > 0 else 0
    gross = np.mean([t["gross_bp"] for t in trades])
    print(f"  {label}: n={n:,d}, WR={wr:.1f}%, Mean={mu:+.2f}bp, "
          f"Med={med:+.2f}bp, Sharpe={sharpe:.2f}, t={t_stat:.2f}, Gross={gross:+.2f}bp")


# ====================================================================
# MAIN
# ====================================================================
def main():
    t_main = time.time()

    # ── MT5 DATA ──
    all_data, avail_pairs = load_mt5_data()
    mt5_trades = []
    if all_data:
        print("\n" + "=" * 70)
        print("MT5 TOKYO H0 STRATEGY (120 days, 15 pairs)")
        print("=" * 70)
        mt5_trades = bt_hour0(all_data, avail_pairs, costs_bp=COSTS_BP)
        basic_stats(mt5_trades, "Tokyo H0 (MT5)")
        if mt5_trades:
            pair_breakdown(mt5_trades)
            rolling_wr(mt5_trades)
            month_breakdown(mt5_trades)
            day_of_week(mt5_trades)
            cost_sensitivity(all_data, avail_pairs)
            max_drawdown(mt5_trades)
            pair_removal_test(all_data, avail_pairs, mt5_trades)

    # ── EXNESS DATA ──
    ex5 = load_exness_m5()
    print("\n" + "=" * 70)
    print("EXNESS TOKYO H0 STRATEGY (3 pairs, Oct-Dec 2025)")
    print("=" * 70)
    ex_trades = bt_hour0_exness(ex5, costs_bp=COSTS_BP)
    basic_stats(ex_trades, "Tokyo H0 (Exness)")
    if ex_trades:
        pair_breakdown(ex_trades)
        rolling_wr(ex_trades)
        month_breakdown(ex_trades)
        day_of_week(ex_trades)
        max_drawdown(ex_trades)
        spread_sensitivity(ex_trades)
        vol_regime_sensitivity(ex_trades)

    print(f"\n{'='*70}")
    print(f"TOKYO H0 STRESS TEST COMPLETE")
    print(f"{'='*70}")
    print(f"  Total time: {time.time() - t_main:.1f}s")
    if mt5_trades:
        print(f"  MT5: {len(mt5_trades)} trades")
    if ex_trades:
        print(f"  Exness: {len(ex_trades)} trades")


if __name__ == '__main__':
    main()
