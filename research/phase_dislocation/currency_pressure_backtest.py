"""Full Currency Inventory Pressure backtest — 3 months, 26 pairs."""
import numpy as np
import pandas as pd
import os, glob

datadir = 'research/phase_dislocation/dukascopy_data'
parquet_files = sorted(glob.glob(os.path.join(datadir, '*.parquet')))
print(f"Parquet files: {len(parquet_files)}")

# Load all pairs
all_data = {}
for fpath in parquet_files:
    pair = os.path.basename(fpath).replace('.parquet', '')
    df = pd.read_parquet(fpath).sort_values('timestamp')
    all_data[pair] = df

pairs_list = sorted(all_data.keys())
print(f"Pairs: {len(pairs_list)}")

# Align to common timestamps
ref = all_data[pairs_list[0]]
timestamps = ref['timestamp'].values
n = len(timestamps)
print(f"Reference bars: {n}")

price_matrix = np.full((n, len(pairs_list)), np.nan)
for j, pair in enumerate(pairs_list):
    df = all_data[pair]
    times = df['timestamp'].values
    prices = df['close'].values
    for i, t in enumerate(timestamps):
        match = np.where(times == t)[0]
        if len(match) > 0:
            price_matrix[i, j] = prices[match[0]]
        else:
            # Try previous bar
            if i > 0:
                price_matrix[i, j] = price_matrix[i-1, j]

# Remove leading NaN rows
first_valid = np.where(~np.isnan(price_matrix).any(axis=1))[0]
if len(first_valid) > 0:
    price_matrix = price_matrix[first_valid[0]:]
    timestamps = timestamps[first_valid[0]:]

n = len(price_matrix)
print(f"Aligned bars: {n}")

lr = np.diff(np.log(price_matrix), axis=0)
timestamps = timestamps[1:]
n_lr = len(lr)
print(f"Return bars: {n_lr}")

# Currency mapping
currencies = ['USD', 'EUR', 'JPY', 'GBP', 'AUD', 'NZD', 'CAD', 'CHF']
def base_quote(pair):
    currs = ['AUD', 'CAD', 'CHF', 'EUR', 'GBP', 'JPY', 'NZD', 'USD']
    for c in currs:
        if pair.startswith(c):
            return c, pair[len(c):]
    return None, None

curr_pairs = {c: [] for c in currencies}
for j, pair in enumerate(pairs_list):
    base, quote = base_quote(pair)
    if base and quote:
        if base in curr_pairs:
            curr_pairs[base].append((j, 1.0, pairs_list[j]))
        if quote in curr_pairs:
            curr_pairs[quote].append((j, -1.0, pairs_list[j]))

# Compute currency returns (volatility-weighted)
curr_rets = np.zeros((n_lr, len(currencies)))
for ci, c in enumerate(currencies):
    pairs = curr_pairs[c]
    if len(pairs) == 0:
        continue
    cols = [p[0] for p in pairs]
    signs = [p[1] for p in pairs]
    rets = lr[:, cols] * np.array(signs)
    # Volatility-weight: give less weight to high-vol pairs
    vols = np.nanstd(rets, axis=0) + 1e-10
    weights = (1.0 / vols) / np.sum(1.0 / vols)
    curr_rets[:, ci] = np.nansum(rets * weights, axis=1)

# Full-period Z-score
curr_z = np.zeros_like(curr_rets)
for ci in range(len(currencies)):
    m = np.mean(curr_rets[:, ci])
    s = np.std(curr_rets[:, ci])
    curr_z[:, ci] = (curr_rets[:, ci] - m) / s

# Spreads (bps)
spreads = {'EURUSD':1.5,'GBPUSD':2.0,'USDJPY':1.8,'AUDUSD':1.5,'NZDUSD':2.0,
    'USDCAD':2.0,'USDCHF':2.0,'EURJPY':2.5,'GBPJPY':4.0,'EURGBP':2.0,
    'EURAUD':2.5,'EURCHF':2.0,'EURCAD':2.0,'EURNZD':2.5,
    'GBPAUD':2.5,'GBPCAD':2.5,'GBPCHF':3.0,'GBPNZD':3.0,
    'AUDJPY':2.5,'AUDCAD':2.5,'AUDCHF':2.5,'AUDNZD':2.5,
    'NZDJPY':3.0,'NZDCAD':2.5,'NZDCHF':2.5,'CADJPY':2.5,'CADCHF':2.5,'CHFJPY':3.0}
default_spread = 2.0

# Split into monthly periods for consistency check
month_labels = []
for t in timestamps:
    dt = pd.Timestamp(t)
    month_labels.append(f"{dt.year}-{dt.month:02d}")
month_labels = np.array(month_labels)
unique_months = sorted(set(month_labels))
print(f"\nMonths: {unique_months}")

# ================================================================
# FULL BACKTEST
# ================================================================
print("\n" + "=" * 60)
print("CURRENCY INVENTORY PRESSURE — 3-MONTH BACKTEST")
print("=" * 60)

spreads_bps = {p: spreads.get(p, default_spread) for p in pairs_list}

# For each currency and threshold
focus_currencies = ['NZD', 'CHF', 'GBP', 'USD', 'JPY']

for z_thresh in [2.0, 2.5]:
    for fwd in [5, 10, 15]:
        all_trades = []
        
        for ci, cname in enumerate(currencies):
            if cname not in focus_currencies:
                continue
            for i in range(30, n_lr - fwd):
                if abs(curr_z[i, ci]) < z_thresh:
                    continue
                direction = np.sign(curr_z[i, ci])
                for j, sign, pair_name in curr_pairs[cname]:
                    expected_dir = direction * sign
                    fwd_ret = np.sum(lr[i:i+fwd, j])
                    pnl = fwd_ret * expected_dir
                    month = month_labels[i]
                    all_trades.append({
                        'currency': cname,
                        'pair': pair_name,
                        'month': month,
                        'z': curr_z[i, ci],
                        'fwd': fwd,
                        'pnl': pnl,
                        'correct': np.sign(fwd_ret) == expected_dir,
                        'fwd_ret_bps': fwd_ret * 10000,
                    })
        
        if not all_trades:
            continue
        
        tdf = pd.DataFrame(all_trades)
        
        # Per-pair best
        pair_stats = []
        for pair_name in tdf['pair'].unique():
            psub = tdf[tdf['pair'] == pair_name]
            sprd = spreads_bps.get(pair_name, default_spread)
            wr = np.mean(psub['correct']) * 100
            avg_pnl = np.mean(psub['pnl']) * 10000
            net = avg_pnl - sprd
            win_avg = np.mean(psub[psub['correct']]['pnl']) * 10000 if sum(psub['correct']) > 0 else 0
            loss_avg = np.mean(psub[~psub['correct']]['pnl']) * 10000 if sum(~psub['correct']) > 0 else 0
            n_events = len(psub)
            n_correct = int(sum(psub['correct']))
            pair_stats.append({
                'pair': pair_name, 'n': n_events, 'n_correct': n_correct,
                'wr': wr, 'avg': avg_pnl, 'net': net,
                'win_avg': win_avg, 'loss_avg': loss_avg,
            })
        
        psdf = pd.DataFrame(pair_stats)
        best = psdf.loc[psdf['net'].idxmax()]
        
        if best['n'] < 10:
            continue
        
        print(f"\nZ>{z_thresh:.1f} Fwd={fwd:>2d}m:")
        print(f"  BEST: {best['pair']:>7s} | n={int(best['n']):>5d} | WR={best['wr']:.0f}% | "
              f"avg={best['avg']:+.2f}bps | win={best['win_avg']:+.2f} loss={best['loss_avg']:+.2f} | "
              f"net={best['net']:+.2f}bps")
        
        # Monthly breakdown for the best pair
        bp_sub = tdf[tdf['pair'] == best['pair']]
        print(f"  Monthly breakdown:")
        for mn in unique_months:
            msub = bp_sub[bp_sub['month'] == mn]
            if len(msub) < 5:
                continue
            mwr = np.mean(msub['correct']) * 100
            mavg = np.mean(msub['pnl']) * 10000
            print(f"    {mn}: n={len(msub):>4d} WR={mwr:.0f}% avg={mavg:+.2f}bps")
        
        # Per-currency contribution
        print(f"  Per currency contributing to best pair trades:")
        for cname in focus_currencies:
            csub = bp_sub[bp_sub['currency'] == cname]
            if len(csub) < 5:
                continue
            cwr = np.mean(csub['correct']) * 100
            cavg = np.mean(csub['pnl']) * 10000
            print(f"    {cname:>4s}: n={len(csub):>4d} WR={cwr:.0f}% avg={cavg:+.2f}bps")

# ================================================================
# MULTI-SIGNAL COMBINATION — What if we trade ALL signals at once?
# ================================================================
print("\n" + "=" * 60)
print("MULTI-SIGNAL PORTFOLIO (Z>2.0, fwd=5, all currencies)")
print("=" * 60)

z_thresh = 2.0
fwd = 5

# For each currency, pick the best single pair
best_pairs = {}
for ci, cname in enumerate(currencies):
    pair_trades = {}
    for i in range(30, n_lr - fwd):
        if abs(curr_z[i, ci]) < z_thresh:
            continue
        direction = np.sign(curr_z[i, ci])
        for j, sign, pair_name in curr_pairs[cname]:
            expected_dir = direction * sign
            fwd_ret = np.sum(lr[i:i+fwd, j])
            pnl = fwd_ret * expected_dir
            if pair_name not in pair_trades:
                pair_trades[pair_name] = []
            pair_trades[pair_name].append({'pnl': pnl, 'correct': np.sign(fwd_ret) == expected_dir})
    
    # Find best pair for this currency
    best_net = -999
    best_pair = None
    for pn, trades in pair_trades.items():
        if len(trades) < 5:
            continue
        avg_pnl = np.mean([t['pnl'] for t in trades]) * 10000
        sprd = spreads_bps.get(pn, default_spread)
        net = avg_pnl - sprd
        if net > best_net:
            best_net = net
            best_pair = pn
            best_wr = np.mean([t['correct'] for t in trades]) * 100
            best_n = len(trades)
    
    if best_pair and best_n >= 10:
        best_pairs[cname] = (best_pair, best_wr, best_net, best_n)
        print(f"  {cname:>4s} -> {best_pair:>7s}: WR={best_wr:.0f}% net={best_net:+.2f}bps n={best_n}")

# Simulate trading ALL best signals simultaneously
print(f"\n  Simulated combined portfolio:")
all_portfolio_trades = []
for i in range(30, n_lr - fwd):
    for cname, (pair_name, wr, net, n_trades) in best_pairs.items():
        ci = currencies.index(cname)
        if abs(curr_z[i, ci]) < z_thresh:
            continue
        direction = np.sign(curr_z[i, ci])
        for j, sign, pn in curr_pairs[cname]:
            if pn != pair_name:
                continue
            expected_dir = direction * sign
            fwd_ret = np.sum(lr[i:i+fwd, j])
            pnl = fwd_ret * expected_dir
            all_portfolio_trades.append({
                'currency': cname, 'pair': pair_name, 'pnl': pnl,
                'correct': np.sign(fwd_ret) == expected_dir,
                'month': month_labels[i],
            })

if all_portfolio_trades:
    ptdf = pd.DataFrame(all_portfolio_trades)
    overall_wr = np.mean(ptdf['correct']) * 100
    overall_avg = np.mean(ptdf['pnl']) * 10000
    n_total = len(ptdf)
    # Weighted average spread cost
    total_spread = sum(spreads_bps.get(p, default_spread) for p in ptdf['pair']) / n_total
    overall_net = overall_avg - total_spread
    print(f"  Total trades: {n_total} ({n_total/63:.1f}/day)")
    print(f"  Overall WR: {overall_wr:.0f}%")
    print(f"  Overall avg: {overall_avg:+.2f}bps")
    print(f"  Avg spread: {total_spread:.1f}bps")
    print(f"  Net: {overall_net:+.2f}bps")
    
    for mn in unique_months:
        msub = ptdf[ptdf['month'] == mn]
        if len(msub) < 5:
            continue
        mwr = np.mean(msub['correct']) * 100
        mavg = np.mean(msub['pnl']) * 10000
        print(f"  {mn}: n={len(msub):>4d} WR={mwr:.0f}% avg={mavg:+.2f}bps")
