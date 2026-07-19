"""
Tokyo H0 Vol Filter Comparison — Full vs No Filter Across Multiple Samples.
"""
import sys, os, time, numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime, timedelta, timezone
from collections import deque

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


def load_mt5_data():
    try:
        import MetaTrader5 as mt5
        if not mt5.initialize():
            return None, []
    except:
        return None, []

    project_root = str(Path(__file__).resolve().parents[3])
    cd_root = os.path.join(project_root, "currency_decomposition")
    sys.path.insert(0, cd_root)
    from config.settings import BASE_CURRENCY_MAP
    all_pairs = list(BASE_CURRENCY_MAP.keys())[:15]

    end = datetime.now()
    start = end - timedelta(days=180)
    all_data = {}
    for pair in all_pairs:
        rates = mt5.copy_rates_range(pair, mt5.TIMEFRAME_M5, start, end)
        if rates is not None and len(rates) > 0:
            all_data[pair] = rates
    available = [p for p in all_pairs if p in all_data]
    mt5.shutdown()
    return all_data, available


def load_exness_m5():
    tick_data = {}
    for pair in TICK_PAIRS:
        dfs = []
        for year, month in MONTHS:
            fn = TICK_DIR / f'{pair}_Raw_Spread_{year}_{month:02d}.zip'
            if not fn.exists():
                continue
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
        bars = pd.DataFrame({
            'open': ohlc['open'], 'high': ohlc['high'],
            'low': ohlc['low'], 'close': ohlc['close'],
        })
        bars = bars.dropna(subset=['open', 'close'])
        bars['hour'] = bars.index.hour
        bars['minute'] = bars.index.minute
        bars['dow'] = bars.index.dayofweek
        bars['date'] = bars.index.date
        bars['ret'] = (bars['close'] - bars['open']) * s
        m5_dict[pair] = bars
    return m5_dict


def bt_hour0(all_data, avail_pairs, costs_bp=COSTS_BP, hold=HOLD,
             lookback=LOOKBACK, top_n=TOP_N, max_pos=MAX_POS, vol_filter=True):
    N = min(len(v) for v in all_data.values() if v is not None)
    atr_window = deque(maxlen=288)
    positions = {}
    trades = []

    for idx in range(max(lookback, 0), min(N - hold, len(next(iter(all_data.values()))))):
        dt = datetime.fromtimestamp(float(all_data[avail_pairs[0]][idx]["time"]), tz=timezone.utc)
        if dt.hour != 0 or dt.minute != 0:
            continue

        if vol_filter:
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


def bt_hour0_exness(m5_dict, costs_bp=COSTS_BP, hold=HOLD,
                    lookback=LOOKBACK, top_n=TOP_N, max_pos=MAX_POS, vol_filter=True):
    avail_pairs = list(m5_dict.keys())
    N = min(len(v) for v in m5_dict.values())
    atr_window = deque(maxlen=288)
    positions = {}
    trades = []

    for i in range(max(lookback, 0), N - hold):
        ref_pair = avail_pairs[0]
        bar = m5_dict[ref_pair].iloc[i]
        dt = bar.name.to_pydatetime().replace(tzinfo=timezone.utc)
        if bar['hour'] != 0 or bar['minute'] != 0:
            continue

        if vol_filter:
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
            trades.append({
                "pnl": net_pnl * 10000, "won": net_pnl > 0,
                "idx": i, "pair": p, "dt": dt,
                "gross_bp": gross * 10000,
            })
            positions[p] = i + hold
    return trades


def stats(trades, label=""):
    if not trades:
        return {"n": 0, "wr": 0, "mean_bp": 0, "total_bp": 0,
                "sharpe": 0, "trades_per_day": 0, "label": label}
    pnls = [t["pnl"] for t in trades]
    n = len(pnls)
    wr = sum(1 for x in pnls if x > 0) / n * 100
    mu = float(np.mean(pnls))
    total = float(np.sum(pnls))
    std = float(np.std(pnls))
    sharpe = mu / std if std > 0 else 0

    days = set()
    for t in trades:
        days.add(t["dt"].date())
    num_days = max(len(days), 1)
    tpd = n / num_days

    return {"n": n, "wr": wr, "mean_bp": mu, "total_bp": total,
            "sharpe": sharpe, "trades_per_day": tpd, "label": label}


def print_result(r):
    print(f"  {r['label']:<40s} n={r['n']:>4d}  WR={r['wr']:>5.1f}%  "
          f"Mean={r['mean_bp']:>+7.2f}bp  Total={r['total_bp']:>+7.0f}bp  "
          f"t/d={r['trades_per_day']:.1f}  Sharpe={r['sharpe']:.2f}")


def print_header(s):
    print(f"\n{'='*80}")
    print(s)
    print('=' * 80)


def time_split(trades, n_splits=3):
    """Split trades into n chronological chunks."""
    if len(trades) < n_splits:
        return [trades] if trades else []
    sorted_trades = sorted(trades, key=lambda t: t["dt"])
    chunk_size = max(1, len(sorted_trades) // n_splits)
    chunks = []
    for i in range(n_splits):
        start = i * chunk_size
        end = start + chunk_size if i < n_splits - 1 else len(sorted_trades)
        chunks.append(sorted_trades[start:end])
    return chunks


def main():
    t0 = time.time()

    # ── LOAD DATA ──
    all_data, avail_pairs = load_mt5_data()
    ex5 = load_exness_m5()

    # ── RUN ALL COMBINATIONS ──
    results = []

    # === MT5 DATA ===
    if all_data:
        print_header("MT5 DATA (15 pairs, ~180 days) — FULL COMPARISON")
        vf_trades = bt_hour0(all_data, avail_pairs, vol_filter=True)
        nf_trades = bt_hour0(all_data, avail_pairs, vol_filter=False)

        for label, trades in [("Full (vol filter)", vf_trades),
                               ("Full (no vol filter)", nf_trades)]:
            r = stats(trades, label)
            results.append(r)
            print_result(r)

        # Ratio
        vf_total = sum(t["pnl"] for t in vf_trades)
        nf_total = sum(t["pnl"] for t in nf_trades)
        ratio = nf_total / vf_total if vf_total != 0 else 0
        print(f"\n  >>> No-filter / Filter total PnL ratio: {ratio:.2f}x")
        print(f"  >>> No-filter adds {nf_total - vf_total:+.0f}bp total PnL")

        # ── Monthly splits ──
        print_header("MT5 — MONTHLY BREAKDOWN")
        for month_label, vf_m, nf_m in [
            ("March 2026", [t for t in vf_trades if t["dt"].month == 3],
                           [t for t in nf_trades if t["dt"].month == 3]),
            ("April 2026", [t for t in vf_trades if t["dt"].month == 4],
                           [t for t in nf_trades if t["dt"].month == 4]),
            ("May 2026", [t for t in vf_trades if t["dt"].month == 5],
                          [t for t in nf_trades if t["dt"].month == 5]),
            ("June 2026", [t for t in vf_trades if t["dt"].month == 6],
                           [t for t in nf_trades if t["dt"].month == 6]),
            ("July 2026", [t for t in vf_trades if t["dt"].month == 7],
                           [t for t in nf_trades if t["dt"].month == 7]),
        ]:
            r_vf = stats(vf_m, f"  Vol filter — {month_label}")
            r_nf = stats(nf_m, f"  No filter  — {month_label}")
            results.append(r_vf)
            results.append(r_nf)
            if r_vf['n'] > 0 or r_nf['n'] > 0:
                print_result(r_vf)
                print_result(r_nf)
                vf_m_total = sum(t["pnl"] for t in vf_m)
                nf_m_total = sum(t["pnl"] for t in nf_m)
                if r_vf['n'] > 0 and r_nf['n'] > 0:
                    print(f"    >>> No-filter / Filter = {nf_m_total/vf_m_total:.2f}x, "
                          f"No-filter delta: {nf_m_total - vf_m_total:+.0f}bp")
                elif r_nf['n'] > 0:
                    print(f"    >>> No-filter trades: {r_nf['n']} (filter blocked all)")
                print()

        # ── Time-chunk splits ──
        print_header("MT5 — CHRONOLOGICAL THIRDS")
        vf_chunks = time_split(vf_trades, 3)
        nf_chunks = time_split(nf_trades, 3)
        for i, (vf_c, nf_c) in enumerate(zip(vf_chunks, nf_chunks)):
            chunk_dates = ""
            if vf_c:
                chunk_dates = f"{vf_c[0]['dt'].date()} to {vf_c[-1]['dt'].date()}"
            elif nf_c:
                chunk_dates = f"{nf_c[0]['dt'].date()} to {nf_c[-1]['dt'].date()}"
            r_vf = stats(vf_c, f"  Vol filter — Third {i+1} ({chunk_dates})")
            r_nf = stats(nf_c, f"  No filter  — Third {i+1} ({chunk_dates})")
            results.append(r_vf)
            results.append(r_nf)
            print_result(r_vf)
            print_result(r_nf)
            vf_c_total = sum(t["pnl"] for t in vf_c)
            nf_c_total = sum(t["pnl"] for t in nf_c)
            if r_vf['n'] > 0 and r_nf['n'] > 0:
                print(f"    >>> No-filter / Filter = {nf_c_total/vf_c_total:.2f}x")
            print()

        # ── Day-of-week ──
        print_header("MT5 — DAY-OF-WEEK")
        dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        for d_idx in range(5):
            vf_d = [t for t in vf_trades if t["dt"].weekday() == d_idx]
            nf_d = [t for t in nf_trades if t["dt"].weekday() == d_idx]
            if not vf_d and not nf_d:
                continue
            r_vf = stats(vf_d, f"  Vol filter — {dow_names[d_idx]}")
            r_nf = stats(nf_d, f"  No filter  — {dow_names[d_idx]}")
            results.append(r_vf)
            results.append(r_nf)
            if r_vf['n'] > 0 or r_nf['n'] > 0:
                print_result(r_vf)
                print_result(r_nf)
                if r_vf['n'] > 0 and r_nf['n'] > 0:
                    vf_d_total = sum(t["pnl"] for t in vf_d)
                    nf_d_total = sum(t["pnl"] for t in nf_d)
                    print(f"    >>> No-filter / Filter = {nf_d_total/vf_d_total:.2f}x")
                print()

        # ── Pair-subset splits ──
        print_header("MT5 — PAIR SUBSETS (leave-group-out)")
        major_template = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'USDCAD']
        major_pairs = [p for p in major_template if p in avail_pairs]
        cross_pairs = [p for p in avail_pairs if p not in major_pairs]

        for subset_name, subset in [("Majors only (EUR, GBP, USD pairs)", major_pairs),
                                     ("Crosses only (non-USD)", cross_pairs)]:
            vf_s = bt_hour0(all_data, subset, vol_filter=True)
            nf_s = bt_hour0(all_data, subset, vol_filter=False)
            r_vf = stats(vf_s, f"  Vol filter — {subset_name}")
            r_nf = stats(nf_s, f"  No filter  — {subset_name}")
            results.append(r_vf)
            results.append(r_nf)
            print_result(r_vf)
            print_result(r_nf)
            vf_s_total = sum(t["pnl"] for t in vf_s)
            nf_s_total = sum(t["pnl"] for t in nf_s)
            if r_vf['n'] > 0 and r_nf['n'] > 0:
                print(f"    >>> No-filter / Filter = {nf_s_total/vf_s_total:.2f}x")
            print()

    # === EXNESS DATA ===
    if ex5:
        print_header("EXNESS TICK DATA (3 pairs, Oct-Dec 2025) — FULL COMPARISON")
        evf_trades = bt_hour0_exness(ex5, vol_filter=True)
        enf_trades = bt_hour0_exness(ex5, vol_filter=False)

        for label, trades in [("Full (vol filter)", evf_trades),
                               ("Full (no vol filter)", enf_trades)]:
            r = stats(trades, label)
            results.append(r)
            print_result(r)

        evf_total = sum(t["pnl"] for t in evf_trades)
        enf_total = sum(t["pnl"] for t in enf_trades)
        ratio = enf_total / evf_total if evf_total != 0 else 0
        print(f"\n  >>> No-filter / Filter total PnL ratio: {ratio:.2f}x")
        print(f"  >>> No-filter adds {enf_total - evf_total:+.0f}bp total PnL")

        # Monthly
        print_header("EXNESS — MONTHLY BREAKDOWN")
        for m_label, month_num in [("October 2025", 10), ("November 2025", 11), ("December 2025", 12)]:
            evf_m = [t for t in evf_trades if t["dt"].month == month_num]
            enf_m = [t for t in enf_trades if t["dt"].month == month_num]
            r_vf = stats(evf_m, f"  Vol filter — {m_label}")
            r_nf = stats(enf_m, f"  No filter  — {m_label}")
            results.append(r_vf)
            results.append(r_nf)
            print_result(r_vf)
            print_result(r_nf)
            if r_vf['n'] > 0 and r_nf['n'] > 0:
                evf_m_total = sum(t["pnl"] for t in evf_m)
                enf_m_total = sum(t["pnl"] for t in enf_m)
                print(f"    >>> No-filter / Filter = {enf_m_total/evf_m_total:.2f}x")
            print()

        # Day of week
        print_header("EXNESS — DAY-OF-WEEK")
        for d_idx in range(5):
            evf_d = [t for t in evf_trades if t["dt"].weekday() == d_idx]
            enf_d = [t for t in enf_trades if t["dt"].weekday() == d_idx]
            if not evf_d and not enf_d:
                continue
            r_vf = stats(evf_d, f"  Vol filter — {dow_names[d_idx]}")
            r_nf = stats(enf_d, f"  No filter  — {dow_names[d_idx]}")
            results.append(r_vf)
            results.append(r_nf)
            print_result(r_vf)
            print_result(r_nf)
            print()

    # ── SUMMARY ──
    print_header("FINAL SUMMARY — Vol Filter vs No Filter Across All Samples")

    summary_rows = []
    for r in results:
        if "Vol filter" in r["label"] or "No filter" in r["label"]:
            summary_rows.append(r)

    print(f"  {'Sample':<50s} {'n':>4s} {'WR':>6s} {'Mean(bp)':>10s} {'Total(bp)':>10s} {'t/d':>5s}")
    print(f"  {'-'*90}")
    for r in summary_rows:
        short = r["label"][:50]
        print(f"  {short:<50s} {r['n']:>4d} {r['wr']:>5.1f}% {r['mean_bp']:>+9.2f} {r['total_bp']:>+9.0f} {r['trades_per_day']:>4.1f}")

    # Consistency check
    print_header("CONSISTENCY VERDICT")
    filter_results = [r for r in summary_rows if "Vol filter" in r["label"]]
    nofilter_results = [r for r in summary_rows if "No filter" in r["label"]]

    consistent_better = 0
    consistent_worse = 0
    ambiguous = 0
    total_comparisons = 0

    for fr, nr in zip(filter_results, nofilter_results):
        total_comparisons += 1
        if nr["total_bp"] > fr["total_bp"]:
            consistent_better += 1
            verdict = "NO FILTER BETTER"
        elif nr["total_bp"] < fr["total_bp"]:
            consistent_worse += 1
            verdict = "VOL FILTER BETTER"
        else:
            ambiguous += 1
            verdict = "TIE"

        sample = fr["label"].replace("Vol filter", "").strip(" —")
        print(f"  {sample:<55s}: {verdict}")

    print(f"\n  No-filter better: {consistent_better}/{total_comparisons} samples")
    print(f"  Vol-filter better: {consistent_worse}/{total_comparisons} samples")
    print(f"  Tie: {ambiguous}/{total_comparisons} samples")

    if consistent_better > consistent_worse:
        print(f"\n  >>> VERDICT: No vol filter consistently outperforms across samples")
        print(f"  >>> Confirmed: Live Tokyo H0 should run without vol filter ✓")
    elif consistent_worse > consistent_better:
        print(f"\n  >>> VERDICT: Vol filter consistently outperforms — contradicts user claim")
    else:
        print(f"\n  >>> VERDICT: Mixed results — no clear winner across samples")

    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
