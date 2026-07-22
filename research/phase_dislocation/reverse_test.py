"""Reverse test the Currency Inventory Pressure edge."""
import numpy as np
import pandas as pd
import os, glob

datadir = 'research/phase_dislocation/dukascopy_data'
all_dfs = {}
for fpath in sorted(glob.glob(os.path.join(datadir, '*.parquet'))):
    pair = os.path.basename(fpath).replace('.parquet', '')
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
price_matrix = merged[pairs_list].values.astype(np.float64)
n = len(price_matrix)

lr = np.diff(np.log(price_matrix), axis=0)
n_lr = len(lr)

currencies = ['USD', 'EUR', 'JPY', 'GBP', 'AUD', 'NZD', 'CAD', 'CHF']

def base_quote(pair):
    currs = ['AUD', 'CAD', 'CHF', 'EUR', 'GBP', 'JPY', 'NZD', 'USD']
    pu = pair.upper()
    for c in currs:
        if pu.startswith(c):
            return c, pu[len(c):]
    return None, None

curr_pairs = {c: [] for c in currencies}
for j, pair in enumerate(pairs_list):
    base, quote = base_quote(pair)
    if base and quote:
        if base in curr_pairs:
            curr_pairs[base].append((j, 1.0, pairs_list[j]))
        if quote in curr_pairs:
            curr_pairs[quote].append((j, -1.0, pairs_list[j]))

# Currency returns
curr_rets = np.zeros((n_lr, len(currencies)))
for ci, c in enumerate(currencies):
    pairs = curr_pairs[c]
    if len(pairs) == 0:
        continue
    cols = [p[0] for p in pairs]
    signs = [p[1] for p in pairs]
    rets = lr[:, cols] * np.array(signs)
    vols = np.nanstd(rets, axis=0) + 1e-10
    weights = (1.0 / vols) / np.sum(1.0 / vols)
    curr_rets[:, ci] = np.nansum(rets * weights, axis=1)

curr_z = np.zeros_like(curr_rets)
for ci in range(len(currencies)):
    m = np.mean(curr_rets[:, ci])
    s = np.std(curr_rets[:, ci])
    if s > 0:
        curr_z[:, ci] = (curr_rets[:, ci] - m) / s

spreads_bps = {
    'eurusd':1.5,'gbpusd':2.0,'usdjpy':1.8,'audusd':1.5,'nzdusd':2.0,
    'usdcad':2.0,'usdchf':2.0,'eurjpy':2.5,'gbpjpy':4.0,'eurgbp':2.0,
    'euraud':2.5,'eurchf':2.0,'eurcad':2.0,'eurnzd':2.5,
    'gbpaud':2.5,'gbpcad':2.5,'gbpchf':3.0,'gbpnzd':3.0,
    'audjpy':2.5,'audcad':2.5,'audchf':2.5,'audnzd':2.5,
    'nzdjpy':3.0,'nzdcad':2.5,'nzdchf':2.5,'cadjpy':2.5,'cadchf':2.5,'chfjpy':3.0
}
default_spread = 2.0

# ================================================================
# TEST 1: Reverse direction — trade AGAINST the edge
# If the forward direction is real, reverse should be negative
# ================================================================
print("=" * 60)
print("TEST 1: REVERSE DIRECTION CHECK")
print("=" * 60)
print("If forward edge is real, reverse should be NEGATIVE\n")

for cname in ['NZD', 'CHF', 'USD', 'AUD']:
    ci = currencies.index(cname)
    for fwd in [5, 10, 15]:
        fwd_ret_series = []
        rev_ret_series = []
        for i in range(30, n_lr - fwd):
            if abs(curr_z[i, ci]) < 2.0:
                continue
            direction = np.sign(curr_z[i, ci])
            for j, sign, pair_name in curr_pairs[cname]:
                if pair_name not in ('nzdusd', 'usdchf', 'audusd'):
                    continue
                expected_dir = direction * sign
                fwd_ret = np.sum(lr[i:i+fwd, j])
                # Forward: trade AS expected
                fwd_pnl = fwd_ret * expected_dir
                # Reverse: trade OPPOSITE of expected
                rev_pnl = fwd_ret * (-expected_dir)
                fwd_ret_series.append(fwd_pnl)
                rev_ret_series.append(rev_pnl)
        
        if len(fwd_ret_series) > 10:
            fwd_avg = np.mean(fwd_ret_series) * 10000
            rev_avg = np.mean(rev_ret_series) * 10000
            pair_name = {'nzd':'nzdusd', 'chf':'usdchf', 'usd':'audusd', 'aud':'audusd'}[cname.lower()]
            sprd = spreads_bps.get(pair_name, default_spread)
            fwd_net = fwd_avg - sprd
            rev_net = rev_avg - sprd
            print(f"  {cname:>4s} fwd={fwd:>2d}m:  FORWARD net={fwd_net:+.2f}bps  REVERSE net={rev_net:+.2f}bps  (fwd>0={fwd_avg>0}, rev<0={rev_avg<0})")

# ================================================================
# TEST 2: Multiple hold durations
# ================================================================
print("\n" + "=" * 60)
print("TEST 2: HOLD DURATION SENSITIVITY")
print("=" * 60)

for cname in ['NZD', 'CHF', 'USD', 'AUD']:
    ci = currencies.index(cname)
    for fwd in [1, 3, 5, 10, 15, 30, 60]:
        pair_pnls = {}
        for i in range(30, n_lr - fwd):
            if abs(curr_z[i, ci]) < 2.0:
                continue
            direction = np.sign(curr_z[i, ci])
            for j, sign, pair_name in curr_pairs[cname]:
                if pair_name not in ('nzdusd', 'usdchf', 'audusd'):
                    continue
                expected_dir = direction * sign
                fwd_ret = np.sum(lr[i:i+fwd, j])
                pnl = fwd_ret * expected_dir
                if pair_name not in pair_pnls:
                    pair_pnls[pair_name] = []
                pair_pnls[pair_name].append(pnl)
        
        for pn, pnls in pair_pnls.items():
            if len(pnls) < 10:
                continue
            avg = np.mean(pnls) * 10000
            sprd = spreads_bps.get(pn, default_spread)
            net = avg - sprd
            sign_str = "+" if net > 0 else ""
            print(f"  {cname:>4s} {pn:>7s} fwd={fwd:>2d}m: n={len(pnls):>4d} avg={avg:+.2f}bps net={sign_str}{net:+.2f}bps")
        break  # just one pair per currency

# ================================================================
# TEST 3: Z-threshold sensitivity (monotonicity check)
# ================================================================
print("\n" + "=" * 60)
print("TEST 3: Z-THRESHOLD SENSITIVITY (monotonicity check)")
print("=" * 60)
print("If edge is real, higher Z -> better WR (monotonically)\n")

for cname in ['NZD', 'CHF', 'USD', 'AUD']:
    ci = currencies.index(cname)
    pair_name = {'NZD':'nzdusd', 'CHF':'usdchf', 'USD':'audusd', 'AUD':'audusd'}[cname]
    fwd = 5
    for zt in [1.0, 1.5, 2.0, 2.5, 3.0]:
        pnls = []
        for i in range(30, n_lr - fwd):
            if abs(curr_z[i, ci]) < zt:
                continue
            direction = np.sign(curr_z[i, ci])
            for j, sign, pn in curr_pairs[cname]:
                if pn != pair_name:
                    continue
                expected_dir = direction * sign
                fwd_ret = np.sum(lr[i:i+fwd, j])
                pnls.append(fwd_ret * expected_dir)
        if len(pnls) < 5:
            continue
        wr = np.mean(np.array(pnls) > 0) * 100
        avg = np.mean(pnls) * 10000
        print(f"  {cname:>4s} {pair_name:>7s} Z>{zt:.1f}: n={len(pnls):>5d} WR={wr:.0f}% avg={avg:+.2f}bps")

# ================================================================
# TEST 4: Walk-forward — train on 2 months, test on 1 month (3 combos)
# ================================================================
print("\n" + "=" * 60)
print("TEST 4: WALK-FORWARD (train 2mo, test 1mo)")
print("=" * 60)

month_labels = np.array([pd.Timestamp(t).strftime('%Y-%m') for t in merged['timestamp'].values[1:]])
combo_sets = [
    (['2026-04', '2026-05'], ['2026-06']),
    (['2026-04', '2026-06'], ['2026-05']),
    (['2026-05', '2026-06'], ['2026-04']),
]

for train_months, test_months in combo_sets:
    train_mask = np.isin(month_labels, train_months)
    test_mask = np.isin(month_labels, test_months)
    
    # Train: compute Z on training data
    train_curr_rets = curr_rets[train_mask]
    train_z = np.zeros_like(train_curr_rets)
    for ci in range(len(currencies)):
        m = np.mean(train_curr_rets[:, ci])
        s = np.std(train_curr_rets[:, ci])
        if s > 0:
            train_z[:, ci] = (train_curr_rets[:, ci] - m) / s
    
    # Test: apply same Z-threshold logic
    z_thresh = 2.0
    fwd = 5
    results = []
    for cname in ['NZD', 'CHF', 'USD', 'AUD']:
        ci = currencies.index(cname)
        pair_name = {'NZD':'nzdusd', 'CHF':'usdchf', 'USD':'audusd', 'AUD':'audusd'}[cname]
        
        # On test set only
        test_indices = np.where(test_mask)[0]
        for idx in test_indices:
            i = int(idx)
            if i < 30 or i >= n_lr - fwd:
                continue
            if abs(curr_z[i, ci]) < z_thresh:  # Use full-period Z for fair comparison
                continue
            direction = np.sign(curr_z[i, ci])
            for j, sign, pn in curr_pairs[cname]:
                if pn != pair_name:
                    continue
                expected_dir = direction * sign
                fwd_ret = np.sum(lr[i:i+fwd, j])
                results.append({
                    'currency': cname, 'pair': pair_name,
                    'pnl': fwd_ret * expected_dir,
                })
    
    if results:
        pdf = pd.DataFrame(results)
        wr = np.mean(pdf['pnl'] > 0) * 100
        avg = np.mean(pdf['pnl']) * 10000
        sprd = spreads_bps.get(pdf['pair'].iloc[0], default_spread)
        net = avg - sprd
        train_str = '+'.join(train_months)
        test_str = '+'.join(test_months)
        print(f"  Train({train_str}) -> Test({test_str}): n={len(pdf):>4d} WR={wr:.0f}% avg={avg:+.2f}bps net={net:+.2f}bps")
