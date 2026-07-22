"""Currency Inventory Pressure backtest with fast alignment."""
import numpy as np
import pandas as pd
import os, glob

datadir = 'research/phase_dislocation/dukascopy_data'
parquet_files = sorted(glob.glob(os.path.join(datadir, '*.parquet')))
print(f"Parquet files: {len(parquet_files)}")

# Load all pairs and merge on timestamp
all_dfs = {}
for fpath in parquet_files:
    pair = os.path.basename(fpath).replace('.parquet', '')
    df = pd.read_parquet(fpath)
    df = df[['timestamp', 'close']].rename(columns={'close': pair})
    all_dfs[pair] = df

# Merge all on timestamp
merged = None
for pair, df in all_dfs.items():
    if merged is None:
        merged = df
    else:
        merged = merged.merge(df, on='timestamp', how='outer')

merged = merged.sort_values('timestamp').dropna().reset_index(drop=True)
print(f"Merged shape: {merged.shape}")
print(f"Time range: {merged['timestamp'].min()} -> {merged['timestamp'].max()}")

pairs_list = [c for c in merged.columns if c != 'timestamp']
n = len(merged)
npairs = len(pairs_list)
print(f"Bars: {n}, Pairs: {npairs}")

price_matrix = merged[pairs_list].values.astype(np.float64)
timestamps = merged['timestamp'].values

lr = np.diff(np.log(price_matrix), axis=0)
timestamps = timestamps[1:]
n_lr = len(lr)

# Currency mapping
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

print(f"\nCurrency pair counts:")
for c, pairs in curr_pairs.items():
    print(f"  {c}: {len(pairs)} pairs")

# Fast currency returns with volatility weighting
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

# Full-period Z
curr_z = np.zeros_like(curr_rets)
for ci in range(len(currencies)):
    m = np.mean(curr_rets[:, ci])
    s = np.std(curr_rets[:, ci])
    if s > 0:
        curr_z[:, ci] = (curr_rets[:, ci] - m) / s

print(f"\nCurrency Z-score ranges:")
for ci, c in enumerate(currencies):
    print(f"  {c}: [{curr_z[:,ci].min():.2f}, {curr_z[:,ci].max():.2f}]")

spreads_bps = {
    'eurusd':1.5,'gbpusd':2.0,'usdjpy':1.8,'audusd':1.5,'nzdusd':2.0,
    'usdcad':2.0,'usdchf':2.0,'eurjpy':2.5,'gbpjpy':4.0,'eurgbp':2.0,
    'euraud':2.5,'eurchf':2.0,'eurcad':2.0,'eurnzd':2.5,
    'gbpaud':2.5,'gbpcad':2.5,'gbpchf':3.0,'gbpnzd':3.0,
    'audjpy':2.5,'audcad':2.5,'audchf':2.5,'audnzd':2.5,
    'nzdjpy':3.0,'nzdcad':2.5,'nzdchf':2.5,'cadjpy':2.5,'cadchf':2.5,'chfjpy':3.0
}
default_spread = 2.0

# Monthly labels
month_labels = np.array([pd.Timestamp(t).strftime('%Y-%m') for t in timestamps])
unique_months = sorted(set(month_labels))
print(f"\nMonths: {unique_months}")

# ================================================================
# PER-CURRENCY BEST PAIR (Z>2.0, fwd=5)
# ================================================================
print("\n" + "=" * 60)
print("PER-CURRENCY BEST PAIR (Z>2.0 | FWD=5)")
print("=" * 60)

z_thresh = 2.0
fwd = 5

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
            pair_trades[pair_name].append({
                'pnl': pnl, 'correct': np.sign(fwd_ret) == expected_dir,
                'month': month_labels[i]
            })
    
    best_net = -999
    best_pair = None
    for pn, trades in pair_trades.items():
        if len(trades) < 10:
            continue
        avg = np.mean([t['pnl'] for t in trades]) * 10000
        wr = np.mean([t['correct'] for t in trades]) * 100
        sprd = spreads_bps.get(pn, default_spread)
        net = avg - sprd
        if net > best_net:
            best_net = net
            best_pair = pn
            best_wr = wr
            best_avg = avg
            best_n = len(trades)
            best_trades = trades
    
    if best_pair and best_net > 0:
        print(f"\n  {cname:>4s} -> {best_pair:>7s}: n={best_n:>4d} WR={best_wr:.0f}% avg={best_avg:+.2f}bps net={best_net:+.2f}bps")
        # Monthly breakdown
        for mn in unique_months:
            m_trades = [t for t in best_trades if t['month'] == mn]
            if len(m_trades) < 5:
                continue
            m_wr = np.mean([t['correct'] for t in m_trades]) * 100
            m_avg = np.mean([t['pnl'] for t in m_trades]) * 10000
            events_per_day = len(m_trades) / 30
            print(f"    {mn}: n={len(m_trades):>3d} ({events_per_day:.1f}/d) WR={m_wr:.0f}% avg={m_avg:+.2f}bps")

# ================================================================
# TRY LOWER THRESHOLD
# ================================================================
print("\n" + "=" * 60)
print("LOWER THRESHOLD CHECK (Z>1.5 | FWD=5)")
print("=" * 60)

for ci, cname in enumerate(['NZD', 'CHF']):
    pair_trades = {}
    for i in range(30, n_lr - 5):
        if abs(curr_z[i, ci]) < 1.5:
            continue
        direction = np.sign(curr_z[i, ci])
        for j, sign, pair_name in curr_pairs[cname]:
            expected_dir = direction * sign
            fwd_ret = np.sum(lr[i:i+5, j])
            pnl = fwd_ret * expected_dir
            if pair_name not in pair_trades:
                pair_trades[pair_name] = []
            pair_trades[pair_name].append({'pnl': pnl, 'correct': np.sign(fwd_ret) == expected_dir})
    
    for pn, trades in pair_trades.items():
        if len(trades) < 20:
            continue
        avg = np.mean([t['pnl'] for t in trades]) * 10000
        wr = np.mean([t['correct'] for t in trades]) * 100
        sprd = spreads_bps.get(pn, default_spread)
        net = avg - sprd
        print(f"  {cname:>4s} {pn:>7s}: n={len(trades):>4d} WR={wr:.0f}% avg={avg:+.2f}bps net={net:+.2f}bps")

# ================================================================
# FULL PORTFOLIO SIMULATION
# ================================================================
print("\n" + "=" * 60)
print("FULL PORTFOLIO (all positive-net signals)")
print("=" * 60)

portfolio = []
for ci, cname in enumerate(currencies):
    pair_trades = {}
    for i in range(30, n_lr - 5):
        if abs(curr_z[i, ci]) < 2.0:
            continue
        direction = np.sign(curr_z[i, ci])
        for j, sign, pair_name in curr_pairs[cname]:
            expected_dir = direction * sign
            fwd_ret = np.sum(lr[i:i+5, j])
            pnl = fwd_ret * expected_dir
            if pair_name not in pair_trades:
                pair_trades[pair_name] = []
            pair_trades[pair_name].append({
                'pnl': pnl, 'correct': np.sign(fwd_ret) == expected_dir,
                'month': month_labels[i],
                'currency': cname,
            })
    
    best_net = -999
    best_choice = None
    for pn, trades in pair_trades.items():
        if len(trades) < 10:
            continue
        avg = np.mean([t['pnl'] for t in trades]) * 10000
        sprd = spreads_bps.get(pn, default_spread)
        net = avg - sprd
        if net > best_net:
            best_net = net
            best_choice = (pn, trades)
    
    if best_choice and best_net > 0:
        pair_name, trades = best_choice
        portfolio.extend(trades)
        print(f"  {cname:>4s} -> {pair_name:>7s}: {len(trades)} trades, net={best_net:+.2f}bps")

if portfolio:
    pdf = pd.DataFrame(portfolio)
    overall_wr = np.mean(pdf['correct']) * 100
    overall_avg = np.mean(pdf['pnl']) * 10000
    avg_spread = np.mean([spreads_bps.get(p, default_spread) for p in pdf['pair']]) if 'pair' in pdf.columns else default_spread
    print(f"\n  Portfolio: {len(pdf)} trades ({len(pdf)/63:.1f}/day)")
    print(f"  WR: {overall_wr:.0f}% | avg: {overall_avg:+.2f}bps | net: {overall_avg-avg_spread:+.2f}bps")
    
    for mn in unique_months:
        msub = pdf[pdf['month'] == mn]
        if len(msub) < 5:
            continue
        m_wr = np.mean(msub['correct']) * 100
        m_avg = np.mean(msub['pnl']) * 10000
        print(f"  {mn}: n={len(msub):>4d} WR={m_wr:.0f}% avg={m_avg:+.2f}bps")
else:
    print("  No positive-net signals found at Z>2.0")
    
    # Try Z>1.5 for the portfolio
    print("\n  Retrying with Z>1.5...")
    portfolio = []
    for ci, cname in enumerate(currencies):
        pair_trades = {}
        for i in range(30, n_lr - 5):
            if abs(curr_z[i, ci]) < 1.5:
                continue
            direction = np.sign(curr_z[i, ci])
            for j, sign, pair_name in curr_pairs[cname]:
                expected_dir = direction * sign
                fwd_ret = np.sum(lr[i:i+5, j])
                pnl = fwd_ret * expected_dir
                if pair_name not in pair_trades:
                    pair_trades[pair_name] = []
                pair_trades[pair_name].append({
                    'pnl': pnl, 'correct': np.sign(fwd_ret) == expected_dir,
                    'month': month_labels[i], 'currency': cname,
                })
        
        best_net = -999
        best_choice = None
        for pn, trades in pair_trades.items():
            if len(trades) < 20:
                continue
            avg = np.mean([t['pnl'] for t in trades]) * 10000
            sprd = spreads_bps.get(pn, default_spread)
            net = avg - sprd
            if net > best_net:
                best_net = net
                best_choice = (pn, trades)
        
        if best_choice and best_net > 0:
            pair_name, trades = best_choice
            portfolio.extend(trades)
            print(f"  {cname:>4s} -> {pair_name:>7s}: {len(trades)} trades, net={best_net:+.2f}bps")
    
    if portfolio:
        pdf = pd.DataFrame(portfolio)
        overall_wr = np.mean(pdf['correct']) * 100
        overall_avg = np.mean(pdf['pnl']) * 10000
        print(f"\n  Portfolio: {len(pdf)} trades ({len(pdf)/63:.1f}/day)")
        print(f"  WR: {overall_wr:.0f}% | avg: {overall_avg:+.2f}bps")
        for mn in unique_months:
            msub = pdf[pdf['month'] == mn]
            if len(msub) < 5:
                continue
            m_wr = np.mean(msub['correct']) * 100
            m_avg = np.mean(msub['pnl']) * 10000
            print(f"  {mn}: n={len(msub):>4d} WR={m_wr:.0f}% avg={m_avg:+.2f}bps")
