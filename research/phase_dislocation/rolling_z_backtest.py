"""
Rolling-window Z-score backtest for Currency Inventory Pressure.
Matches paper trade's real-time logic: rolling mean/std, no lookahead.
Sweeps window sizes to find optimal min_hist and z_window.
"""
import numpy as np
import pandas as pd
import os, glob, json

# --- Config ---
DATADIR = 'research/phase_dislocation/dukascopy_data'
Z_THRESH = 2.0
FWD_MINUTES = 5
LOT_SIZE = 0.5
CONTRACT_SIZE = 100000
WINDOWS = [100, 500, 1000, 2000, 4000]
CURRENCIES = ['USD', 'EUR', 'JPY', 'GBP', 'AUD', 'NZD', 'CAD', 'CHF']

# Spreads (bps) from original backtest
SPREADS = {
    'EURUSD':1.5,'GBPUSD':2.0,'USDJPY':1.8,'AUDUSD':1.5,'NZDUSD':2.0,
    'USDCAD':2.0,'USDCHF':2.0,'EURJPY':2.5,'GBPJPY':4.0,'EURGBP':2.0,
    'EURAUD':2.5,'EURCHF':2.0,'EURCAD':2.0,'EURNZD':2.5,
    'GBPAUD':2.5,'GBPCAD':2.5,'GBPCHF':3.0,'GBPNZD':3.0,
    'AUDJPY':2.5,'AUDCAD':2.5,'AUDCHF':2.5,'AUDNZD':2.5,
    'NZDJPY':3.0,'NZDCAD':2.5,'NZDCHF':2.5,'CADJPY':2.5,'CADCHF':2.5,'CHFJPY':3.0,
}

def base_quote(pair):
    for c in ['AUD','CAD','CHF','EUR','GBP','JPY','NZD','USD']:
        if pair.startswith(c):
            return c, pair[len(c):]
    return None, None

# --- Load data ---
files = sorted(glob.glob(os.path.join(DATADIR, '*.parquet')))
print(f"Loading {len(files)} pairs...")
all_data = {}
for fpath in files:
    pair = os.path.basename(fpath).replace('.parquet', '').upper()
    df = pd.read_parquet(fpath).sort_values('timestamp')
    all_data[pair] = df

pairs_list = sorted(all_data.keys())
print(f"Pairs: {len(pairs_list)}")

# Align to common timestamps
ref = all_data[pairs_list[0]]
timestamps = ref['timestamp'].values
n = len(timestamps)
price_matrix = np.full((n, len(pairs_list)), np.nan)
for j, pair in enumerate(pairs_list):
    df = all_data[pair]
    times = df['timestamp'].values
    prices = df['close'].values
    for i, t in enumerate(timestamps):
        match = np.where(times == t)[0]
        if len(match) > 0:
            price_matrix[i, j] = prices[match[0]]
        elif i > 0:
            price_matrix[i, j] = price_matrix[i-1, j]

first_valid = np.where(~np.isnan(price_matrix).any(axis=1))[0]
if len(first_valid) > 0:
    price_matrix = price_matrix[first_valid[0]:]
    timestamps = timestamps[first_valid[0]:]

n = len(price_matrix)
lr = np.diff(np.log(price_matrix), axis=0)
print(f"Aligned bars: {n}, log returns: {lr.shape[0]}")

# Build currency->pairs mapping
curr_pairs = {c: [] for c in CURRENCIES}
for j, pair in enumerate(pairs_list):
    base, quote = base_quote(pair)
    if base and quote:
        if base in curr_pairs:
            curr_pairs[base].append((j, 1.0, pair))
        if quote in curr_pairs:
            curr_pairs[quote].append((j, -1.0, pair))

# Best pair per currency (from backtest: highest net bps)
BEST_PAIR = {
    "USD": "AUDUSD", "EUR": "EURUSD", "JPY": "NZDJPY",
    "GBP": "GBPUSD", "AUD": "AUDUSD", "NZD": "NZDUSD",
    "CAD": "NZDCAD", "CHF": "USDCHF",
}

print(f"\n{'='*80}")
print(f"{'ROLLING Z-SCORE BACKTEST':^80}")
print(f"{'Z threshold='+str(Z_THRESH)+' fwd='+str(FWD_MINUTES)+'min lot='+str(LOT_SIZE):^80}")
print(f"{'='*80}")

results = {}

for W in WINDOWS:
    print(f"\n{'─'*80}")
    print(f"WINDOW SIZE: {W} bars (~{W//60}h{W%60:02d}m)")
    print(f"{'─'*80}")

    all_trades = []
    nlr = lr.shape[0]

    # Compute rolling currency returns (same method as paper trade)
    # First pre-compute per-bar currency returns
    curr_rets = np.zeros((nlr, len(CURRENCIES)))
    for ci, c in enumerate(CURRENCIES):
        pairs = curr_pairs[c]
        if len(pairs) == 0:
            continue
        cols = [p[0] for p in pairs]
        signs = [p[1] for p in pairs]
        raw_rets = lr[:, cols] * np.array(signs)
        vols = np.nanstd(raw_rets[:min(200, nlr)], axis=0) + 1e-10
        weights = (1.0 / vols) / np.sum(1.0 / vols)
        curr_rets[:, ci] = np.nansum(raw_rets * weights, axis=1)

    # Rolling Z-score (paper trade logic: at each bar i, use hist[i-W:i])
    # Track per-currency history for proper rolling mean/std
    history = {c: [] for c in CURRENCIES}

    for i in range(1, nlr - FWD_MINUTES):
        updated = False
        for ci, c in enumerate(CURRENCIES):
            ret = curr_rets[i, ci]
            if np.isnan(ret):
                continue
            history[c].append(ret)
            if not updated:
                updated = True

        if not updated:
            continue

        # Compute Z for each currency using rolling window
        for ci, c in enumerate(CURRENCIES):
            hist = history[c]
            if len(hist) < 5:  # min_hist
                continue

            # Use last W bars for mean/std, but cap at available history
            lookback = hist[-W:] if len(hist) >= W else hist
            arr = np.array(lookback)
            mean = np.mean(arr)
            std = np.std(arr)
            if std < 1e-12:
                continue

            ret = curr_rets[i, ci]
            if np.isnan(ret):
                continue
            z = (ret - mean) / std

            if abs(z) < Z_THRESH:
                continue

            pair = BEST_PAIR.get(c)
            if pair is None or pair not in pairs_list:
                continue

            # Find pair index
            try:
                j = pairs_list.index(pair)
            except ValueError:
                continue

            # Sign adjustment
            sign = 1.0
            for idx, s, pn in curr_pairs[c]:
                if pn == pair:
                    sign = s
                    break

            direction = 1 if z > 0 else -1
            expected_dir = direction * sign
            fwd_ret = np.sum(lr[i+1:i+1+FWD_MINUTES, j])
            pnl = fwd_ret * expected_dir
            correct = np.sign(fwd_ret) == expected_dir

            # Dollar PnL
            entry_price = price_matrix[i+1, j]
            exit_price = price_matrix[i+1+FWD_MINUTES, j]
            gross_dollars = LOT_SIZE * CONTRACT_SIZE * (exit_price - entry_price) * direction
            # Correct for sign: if direction=+1 (BUY), pnl = lot * (exit-entry)
            # if direction=-1 (SELL), pnl = lot * (entry-exit)

            sprd = SPREADS.get(pair, 2.0)
            # Net bps = avg_bps - spread_bps (spread is a cost in bps)
            # Divide by entry price to convert price diff to bps
            pnl_bps = (exit_price - entry_price) / entry_price * 10000 * direction

            all_trades.append({
                'currency': c,
                'pair': pair,
                'z': z,
                'pnl_bps': pnl_bps,
                'pnl_dollars': gross_dollars,
                'correct': correct,
                'entry_time': timestamps[i+1],
                'hist_len': len(hist),
            })

    if not all_trades:
        print(f"  No trades generated")
        results[W] = None
        continue

    tdf = pd.DataFrame(all_trades)
    n_trades = len(tdf)
    wr = np.mean(tdf['correct']) * 100

    # Per-pair stats
    print(f"\n  Total trades: {n_trades}")
    print(f"  Portfolio WR: {wr:.1f}%")
    print(f"\n  Per-pair breakdown:")

    pair_data = []
    for pair in tdf['pair'].unique():
        psub = tdf[tdf['pair'] == pair]
        n = len(psub)
        if n < 5:
            continue
        pwr = np.mean(psub['correct']) * 100
        mean_bps = np.mean(psub['pnl_bps'])
        mean_dollars = np.mean(psub['pnl_dollars'])
        net_bps = mean_bps - SPREADS.get(pair, 2.0)
        pair_data.append({
            'pair': pair, 'n': n, 'wr': pwr,
            'mean_bps': mean_bps, 'net_bps': net_bps,
            'mean_dollars': mean_dollars,
        })

    pair_data.sort(key=lambda x: x['net_bps'], reverse=True)
    for pd_ in pair_data:
        print(f"    {pd_['pair']:>7s}: n={pd_['n']:>4d} WR={pd_['wr']:5.1f}% mean={pd_['mean_bps']:+.2f}bps net={pd_['net_bps']:+.2f}bps ${pd_['mean_dollars']:+.2f}")

    # Per-currency stats
    print(f"\n  Per-currency breakdown:")
    for c in CURRENCIES:
        csub = tdf[tdf['currency'] == c]
        if len(csub) < 5:
            continue
        cwr = np.mean(csub['correct']) * 100
        cmean = np.mean(csub['pnl_bps'])
        cdollars = np.mean(csub['pnl_dollars'])
        print(f"    {c:>4s}: n={len(csub):>4d} WR={cwr:5.1f}% mean={cmean:+.2f}bps ${cdollars:+.2f}")

    # Hist length analysis (to determine min_hist)
    print(f"\n  Hist length distribution:")
    hist_lens = tdf['hist_len']
    print(f"    Min: {hist_lens.min()}, Median: {hist_lens.median():.0f}, Mean: {hist_lens.mean():.0f}, Max: {hist_lens.max()}")
    for pct in [5, 10, 25, 50]:
        val = np.percentile(hist_lens, pct)
        print(f"    P{pct}: {val:.0f}")

    # Dollar stats
    dollars = tdf['pnl_dollars']
    print(f"\n  Dollar PnL per trade:")
    print(f"    Mean: ${dollars.mean():+.2f}")
    print(f"    Median: ${dollars.median():+.2f}")
    print(f"    Std: ${dollars.std():+.2f}")
    print(f"    Min: ${dollars.min():+.2f}")
    print(f"    Max: ${dollars.max():+.2f}")
    print(f"    P5: ${np.percentile(dollars, 5):+.2f}")
    print(f"    P95: ${np.percentile(dollars, 95):+.2f}")

    results[W] = {
        'n_trades': n_trades,
        'wr': wr,
        'mean_bps': np.mean(tdf['pnl_bps']),
        'mean_dollars': np.mean(tdf['pnl_dollars']),
        'min_hist_observed': int(hist_lens.min()),
    }

print(f"\n{'='*80}")
print(f"{'SUMMARY':^80}")
print(f"{'='*80}")
print(f"{'Window':>6s} | {'Trades':>7s} | {'WR':>5s} | {'Mean(bps)':>10s} | {'Mean($)':>10s} | {'MinHist':>7s}")
print(f"{'-'*6}-+-{'-'*7}-+-{'-'*5}-+-{'-'*10}-+-{'-'*10}-+-{'-'*7}")
for W in WINDOWS:
    r = results.get(W)
    if r:
        print(f"{W:>6d} | {r['n_trades']:>7d} | {r['wr']:>5.1f}% | {r['mean_bps']:>+9.2f} | {r['mean_dollars']:>+9.2f} | {r['min_hist_observed']:>7d}")
    else:
        print(f"{W:>6d} | {'NO DATA':>7}")

# Save results
out = {str(k): v for k, v in results.items()}
with open('research/phase_dislocation/rolling_z_results.json', 'w') as f:
    json.dump(out, f, indent=2, default=str)
print(f"\nResults saved to rolling_z_results.json")
print("Done.")
