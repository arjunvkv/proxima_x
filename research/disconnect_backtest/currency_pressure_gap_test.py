"""Currency Pressure — MT5 disconnect gap backtest.
Matches the ORIGINAL validated backtest methodology (in-sample Z).
Tests if gap detection (skipping corrupted returns after reconnect) helps or hurts.
"""
import numpy as np
import pandas as pd
import os, glob, time

np.random.seed(42)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'phase_dislocation', 'dukascopy_data')
CURRENCIES = ['USD', 'EUR', 'JPY', 'GBP', 'AUD', 'NZD', 'CAD', 'CHF']
Z_THRESH = 2.0
HOLD_MIN = 5
GAP_THRESHOLD_SEC = 90
FOCUS_CURRENCIES = ['NZD', 'CHF', 'JPY', 'USD', 'GBP']

SPREADS_BPS = {
    'EURUSD':1.5,'GBPUSD':2.0,'USDJPY':1.8,'AUDUSD':1.5,'NZDUSD':2.0,
    'USDCAD':2.0,'USDCHF':2.0,'EURJPY':2.5,'GBPJPY':4.0,'EURGBP':2.0,
    'EURAUD':2.5,'EURCHF':2.0,'EURCAD':2.0,'EURNZD':2.5,
    'GBPAUD':2.5,'GBPCAD':2.5,'GBPCHF':3.0,'GBPNZD':3.0,
    'AUDJPY':2.5,'AUDCAD':2.5,'AUDCHF':2.5,'AUDNZD':2.5,
    'NZDJPY':3.0,'NZDCAD':2.5,'NZDCHF':2.5,'CADJPY':2.5,'CADCHF':2.5,'CHFJPY':3.0,
}
DEFAULT_SPREAD = 2.0

BEST_PAIR = {
    "USD": "AUDUSD", "EUR": "EURUSD", "JPY": "NZDJPY",
    "GBP": "GBPUSD", "AUD": "AUDUSD", "NZD": "NZDUSD",
    "CAD": "NZDCAD", "CHF": "USDCHF",
}


def load_and_map():
    """Load data and build currency pair maps."""
    all_dfs = {}
    for fpath in sorted(glob.glob(os.path.join(DATA_DIR, '*.parquet'))):
        pair = os.path.basename(fpath).replace('.parquet', '').upper()
        df = pd.read_parquet(fpath)[['timestamp', 'close']].rename(columns={'close': pair})
        all_dfs[pair] = df

    merged = None
    for pair, df in all_dfs.items():
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on='timestamp', how='outer')
    merged = merged.sort_values('timestamp').dropna().reset_index(drop=True)
    pairs_list = [c for c in merged.columns if c != 'timestamp']
    prices = merged[pairs_list].values.astype(np.float64)
    ts = merged['timestamp'].astype(np.int64).values // 10**9

    # Currency pair mapping
    curr_pairs = {c: [] for c in CURRENCIES}
    for j, pair in enumerate(pairs_list):
        for c in ['AUD', 'CAD', 'CHF', 'EUR', 'GBP', 'JPY', 'NZD', 'USD']:
            if pair.startswith(c):
                quote = pair[len(c):]
                if c in curr_pairs:
                    curr_pairs[c].append((j, 1.0, pair))
                if quote in curr_pairs:
                    curr_pairs[quote].append((j, -1.0, pair))
                break

    # Pre-compute constant vol weights (full-sample)
    lr_full = np.diff(np.log(prices + 1e-15), axis=0)
    pair_vols = {}
    for j, p in enumerate(pairs_list):
        pair_vols[p] = np.std(lr_full[:, j]) + 1e-10

    curr_wm = {}
    for c in CURRENCIES:
        cp = curr_pairs.get(c, [])
        if len(cp) < 2:
            continue
        cols = np.array([p[0] for p in cp])
        signs = np.array([p[1] for p in cp])
        pvs = np.array([pair_vols[p[2]] for p in cp])
        inv = 1.0 / pvs
        w = inv / np.sum(inv)
        curr_wm[c] = (cols, signs, w)

    # Best pair index
    pair_idx = {p: j for j, p in enumerate(pairs_list)}
    best_j = {}
    for c in CURRENCIES:
        bp = BEST_PAIR.get(c)
        if bp and bp in pair_idx:
            best_j[c] = pair_idx[bp]

    return {
        'prices': prices, 'ts': ts, 'pairs_list': pairs_list,
        'pair_idx': pair_idx, 'best_j': best_j,
        'curr_wm': curr_wm, 'lr_full': lr_full,
        'month_labels': pd.DatetimeIndex(pd.to_datetime(ts, unit='s')),
    }


def gap_mask(prices, ts, drop_rate, gap_min, rng):
    """Return modified price/ts with gaps injected."""
    n = len(prices)
    mask = np.ones(n, dtype=bool)
    if drop_rate > 0 and n > gap_min + 2:
        nd = max(1, int(n * drop_rate))
        ms = n - gap_min
        ng = max(1, min(nd // gap_min, ms))
        if ng > 0 and ms > 0:
            starts = rng.choice(ms, size=min(ng, ms), replace=False)
            for s in starts:
                mask[s:s+gap_min] = False
    return prices[mask], ts[mask]


def compute_curr_rets(lr, curr_wm):
    """Compute all currency returns from pair returns."""
    cr = {}
    for c in curr_wm:
        cols, signs, w = curr_wm[c]
        cr[c] = np.sum(lr[:, cols] * signs * w, axis=1)
    return cr


def compute_z_in_sample(rets):
    """Full-sample Z-score (matches original validated backtest)."""
    m = np.mean(rets)
    s = np.std(rets)
    if s < 1e-12:
        return np.zeros_like(rets)
    return (rets - m) / s


def run_month(prices, ts, data, dr, gd, rng, use_gap_detection):
    """Run a single month with given gap params."""
    mp, mt = gap_mask(prices, ts, dr, gd, rng)
    lr = np.diff(np.log(mp + 1e-15), axis=0)
    n = len(lr)

    # Detect gap-corrupted returns
    gap_pos = np.zeros(n, dtype=bool)
    if use_gap_detection:
        for i in range(1, n):
            if mt[i] - mt[i-1] > GAP_THRESHOLD_SEC:
                gap_pos[i] = True

    # Currency returns
    cr = compute_curr_rets(lr, data['curr_wm'])

    # In-sample Z
    z = {}
    for c in cr:
        # For dirty: use all returns
        # For clean: exclude gap-corrupted returns from Z computation
        if use_gap_detection:
            valid = ~gap_pos
            c_clean = cr[c][valid]
            if len(c_clean) < 100:
                z[c] = np.full(n, np.nan)
                continue
            m = np.mean(c_clean)
            s = np.std(c_clean)
            z[c] = (cr[c] - m) / s if s > 1e-12 else np.zeros(n)
            z[c][gap_pos] = np.nan
        else:
            z[c] = compute_z_in_sample(cr[c])

    # Generate trades (vectorized per currency)
    trades = []
    for c in sorted(cr.keys()):
        zv = z[c]
        if zv is None or np.all(np.isnan(zv)):
            continue
        j = data['best_j'].get(c)
        if j is None:
            continue

        # Find positions where Z exceeds threshold
        sig = ~np.isnan(zv) & (np.abs(zv) >= Z_THRESH) & ~gap_pos
        sig_indices = np.where(sig)[0]
        sig_indices = sig_indices[sig_indices < n - HOLD_MIN]

        if len(sig_indices) == 0:
            continue

        directions = np.where(zv[sig_indices] > 0, 1, -1)

        # Compute forward returns
        fwd_list = []
        for idx in sig_indices:
            fwd_list.append(np.sum(lr[idx+1:idx+1+HOLD_MIN, j]))
        fwd_arr = np.array(fwd_list)

        pnls = fwd_arr * directions
        corrects = (np.sign(fwd_arr) == directions).astype(int)

        for k, idx in enumerate(sig_indices):
            trades.append({
                'currency': c, 'pair': BEST_PAIR[c],
                'pnl': pnls[k],
                'correct': corrects[k],
            })

    return trades


def summarize(trades, fold, dr, gd, rep, mode):
    if not trades:
        return None
    df = pd.DataFrame(trades)
    wr = df['correct'].mean() * 100
    avg = df['pnl'].mean() * 10000
    sp = SPREADS_BPS.get(df['pair'].iloc[0], DEFAULT_SPREAD)
    return {
        'fold': fold, 'dr': dr, 'gd': gd, 'rep': rep, 'mode': mode,
        'n': len(trades), 'wr': wr, 'avg_bps': avg, 'net_bps': avg - sp,
    }


def main():
    print("=" * 90)
    print("CURRENCY PRESSURE — MT5 DISCONNECT GAP BACKTEST")
    print("Method: In-sample Z (matches original validated backtest)")
    print("=" * 90)
    t0 = time.time()

    print("\nLoading...")
    data = load_and_map()
    print("  %d pairs, %d bars" % (len(data['pairs_list']), len(data['prices'])))

    months = sorted(set(m.strftime('%Y-%m') for m in data['month_labels']))
    print("  Months: %s" % months)

    # Reduced grid for speed
    drop_rates = [0.00, 0.05, 0.10]
    gap_durs = [1, 3, 5]
    n_repeats = 10

    all_rows = []

    for month in months:
        mask = data['month_labels'].strftime('%Y-%m') == month
        mp, mt = data['prices'][mask], data['ts'][mask]
        if len(mp) < 500:
            continue

        print("\n[%s] %d bars..." % (month, len(mp)), end=' ', flush=True)
        t1 = time.time()
        cnt = 0

        for dr in drop_rates:
            for gd in gap_durs:
                for rep in range(n_repeats):
                    rng = np.random.RandomState(hash((month, dr, gd, rep)) % 2**31)
                    for mode, ug in [('dirty', False), ('gap_detect', True)]:
                        trades = run_month(mp, mt, data, dr, gd, rng, ug)
                        row = summarize(trades, month, dr, gd, rep, mode)
                        if row:
                            all_rows.append(row)
                            cnt += 1

        print("%d rows in %.1fs" % (cnt, time.time() - t1), flush=True)

    # Walk-forward
    for i in range(len(months) - 1):
        train_m, test_m = months[i], months[i+1]
        ti = data['month_labels'].strftime('%Y-%m') == train_m
        te = data['month_labels'].strftime('%Y-%m') == test_m
        train_p, test_p = data['prices'][ti], data['prices'][te]
        train_t, test_t = data['ts'][ti], data['ts'][te]
        if len(train_p) < 200 or len(test_p) < 200:
            continue

        fold = "wf_%s->%s" % (train_m, test_m)
        print("\n[%s] train=%d test=%d..." % (fold, len(train_p), len(test_p)), end=' ', flush=True)
        t1 = time.time()
        cnt = 0

        # Learn vol weights from training
        lr_tr = np.diff(np.log(train_p + 1e-15), axis=0)
        wf_wm = {}
        for c in data['curr_wm']:
            cols, signs, _ = data['curr_wm'][c]
            vols = np.std(lr_tr[:, cols] * signs, axis=0) + 1e-10
            inv = 1.0 / vols
            w = inv / np.sum(inv)
            wf_wm[c] = (cols, signs, w)

        orig_wm = data['curr_wm']
        data['curr_wm'] = wf_wm

        for dr in drop_rates:
            for gd in gap_durs:
                for rep in range(n_repeats):
                    rng = np.random.RandomState(hash((fold, dr, gd, rep)) % 2**31)
                    for mode, ug in [('dirty', False), ('gap_detect', True)]:
                        trades = run_month(test_p, test_t, data, dr, gd, rng, ug)
                        row = summarize(trades, fold, dr, gd, rep, mode)
                        if row:
                            all_rows.append(row)
                            cnt += 1

        data['curr_wm'] = orig_wm
        print("%d rows in %.1fs" % (cnt, time.time() - t1), flush=True)

    # === RESULTS ===
    df = pd.DataFrame(all_rows)
    if df.empty:
        print("\nNo results.")
        return

    print("\n" + "=" * 90)
    print("RESULTS (%d sims)" % len(df))
    print("=" * 90)

    for mode, label in [('dirty', 'DIRTY (current)'), ('gap_detect', 'GAP DETECT')]:
        s = df[df['mode'] == mode]
        print("\n[%s]" % label)
        print("  %5s %3s %5s %5s %6s %6s %8s" % ("Drp%", "Gm", "Flds", "Sims", "nAvg", "WR%", "Net"))
        for dr in sorted(s['dr'].unique()):
            for gd in sorted(s['gd'].unique()):
                ss = s[(s['dr'] == dr) & (s['gd'] == gd)]
                if len(ss) < 3:
                    continue
                print("  %5.0f%% %3d %5d %5d %6d %6.1f %+8.2f" % (
                    dr*100, gd, len(ss['fold'].unique()), len(ss),
                    int(ss['n'].mean()), ss['wr'].mean(), ss['net_bps'].mean()))

    # Comparison
    print("\n" + "=" * 90)
    print("DIRTY vs GAP_DETECT")
    print("%5s %3s %6s %6s %6s %8s %8s %8s" % ("Drp%", "Gm", "D_WR", "C_WR", "dWR", "D_Net", "C_Net", "dNet"))
    for dr in sorted(df['dr'].unique()):
        for gd in sorted(df['gd'].unique()):
            d = df[(df['mode'] == 'dirty') & (df['dr'] == dr) & (df['gd'] == gd)]
            c = df[(df['mode'] == 'gap_detect') & (df['dr'] == dr) & (df['gd'] == gd)]
            if len(d) < 3 or len(c) < 3:
                continue
            print("%5.0f%% %3d %6.1f %6.1f %+6.1f %+8.2f %+8.2f %+8.2f" % (
                dr*100, gd, d['wr'].mean(), c['wr'].mean(),
                c['wr'].mean() - d['wr'].mean(),
                d['net_bps'].mean(), c['net_bps'].mean(),
                c['net_bps'].mean() - d['net_bps'].mean()))

    # VERDICT
    print("\n" + "=" * 90)
    print("VERDICT")
    for dr in sorted(df['dr'].unique()):
        for gd in sorted(df['gd'].unique()):
            d = df[(df['mode'] == 'dirty') & (df['dr'] == dr) & (df['gd'] == gd)]
            c = df[(df['mode'] == 'gap_detect') & (df['dr'] == dr) & (df['gd'] == gd)]
            if len(d) < 3 or len(c) < 3:
                continue
            dw, cw = d['wr'].mean(), c['wr'].mean()
            dn, cn = d['net_bps'].mean(), c['net_bps'].mean()
            delta_wr = cw - dw
            delta_net = cn - dn
            icon = "+" if delta_net > 0 else ("0" if abs(delta_net) < 0.1 else "-")
            print("  dr=%d%% gd=%dm: dirty WR=%.1f%% net=%+.2f | detect WR=%.1f%% net=%+.2f | %sWR=%+.1f%% Net=%+.2f" % (
                dr*100, gd, dw, dn, cw, cn, icon, delta_wr, delta_net))

    print("\nTime: %.1fs" % (time.time() - t0))


if __name__ == '__main__':
    main()
