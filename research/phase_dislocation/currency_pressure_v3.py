"""Currency Inventory Pressure — event frequency vs edge quality analysis."""
import numpy as np
import pandas as pd
import os

mdir = 'data/market'
fx_pairs = sorted([f.replace('.parquet', '') for f in os.listdir(mdir) 
                   if f.endswith('.parquet') and f.replace('.parquet', '') not in ('NAS100', 'XAUUSD')])

def base_quote(pair):
    currs = ['AUD', 'CAD', 'CHF', 'EUR', 'GBP', 'JPY', 'NZD', 'USD']
    for c in currs:
        if pair.startswith(c):
            return c, pair[len(c):]
    return None, None

all_data = {}
for pair in fx_pairs:
    df = pd.read_parquet(f'{mdir}/{pair}.parquet').sort_values('timestamp')
    all_data[pair] = df

ref = all_data['EURJPY']
timestamps = ref['timestamp' if 'timestamp' in ref.columns else 'time'].values
n = len(timestamps)
pairs_list = sorted(all_data.keys())
price_matrix = np.full((n, len(pairs_list)), np.nan)
for j, pair in enumerate(pairs_list):
    df = all_data[pair]
    tcol = 'timestamp' if 'timestamp' in df.columns else 'time'
    times = df[tcol].values
    p_col = df['close'].values
    for i, t in enumerate(timestamps):
        match = np.where(times == t)[0]
        if len(match) > 0:
            price_matrix[i, j] = p_col[match[0]]

valid_rows = ~np.isnan(price_matrix).any(axis=1)
price_matrix = price_matrix[valid_rows]
n = len(price_matrix)

lr = np.diff(np.log(price_matrix), axis=0)
n_lr = len(lr)

currencies = ['USD', 'EUR', 'JPY', 'GBP', 'AUD', 'NZD', 'CAD', 'CHF']
curr_pairs = {c: [] for c in currencies}
for j, pair in enumerate(pairs_list):
    base, quote = base_quote(pair)
    if base and quote:
        if base in curr_pairs:
            curr_pairs[base].append((j, 1.0, pairs_list[j]))
        if quote in curr_pairs:
            curr_pairs[quote].append((j, -1.0, pairs_list[j]))

curr_rets = np.zeros((n_lr, len(currencies)))
for ci, c in enumerate(currencies):
    pairs = curr_pairs[c]
    if len(pairs) == 0:
        continue
    cols = [p[0] for p in pairs]
    signs = [p[1] for p in pairs]
    rets = lr[:, cols] * np.array(signs)
    curr_rets[:, ci] = np.nanmean(rets, axis=1)

curr_z = np.zeros_like(curr_rets)
for ci in range(len(currencies)):
    m = np.mean(curr_rets[:, ci])
    s = np.std(curr_rets[:, ci])
    curr_z[:, ci] = (curr_rets[:, ci] - m) / s

spreads = {'EURUSD':1.5,'GBPUSD':2.0,'USDJPY':1.8,'AUDUSD':1.5,'NZDUSD':2.0,'USDCAD':2.0,'USDCHF':2.0,'EURJPY':2.5,'GBPJPY':4.0,'EURGBP':2.0,'EURAUD':2.5,'EURCHF':2.0,'EURCAD':2.0,'EURNZD':2.5,'GBPAUD':2.5,'GBPCAD':2.5,'GBPCHF':3.0,'GBPNZD':3.0,'AUDJPY':2.5,'AUDCAD':2.5,'AUDCHF':2.5,'AUDNZD':2.5,'NZDJPY':3.0,'NZDCAD':2.5,'NZDCHF':2.5,'CADJPY':2.5,'CADCHF':2.5,'CHFJPY':3.0}
default_spread = 2.0

# Focus on NZD (best performer) and CHF (most reliable)
focus = ['NZD', 'CHF', 'GBP', 'USD']

for cname in focus:
    ci = currencies.index(cname)
    print(f"\n{'='*60}")
    print(f"CURRENCY: {cname}")
    print(f"{'='*60}")
    
    for z_thresh in [1.5, 2.0, 2.5, 3.0]:
        for fwd in [5, 10, 15]:
            # Collect ALL events, then analyze per-pair
            all_events = []
            for i in range(30, n_lr - fwd):
                if abs(curr_z[i, ci]) < z_thresh:
                    continue
                direction = np.sign(curr_z[i, ci])
                for j, sign, pair_name in curr_pairs[cname]:
                    expected_dir = direction * sign
                    fwd_ret = lr[i:i+fwd, j].sum()
                    net_pnl = fwd_ret * expected_dir  # positive = good
                    all_events.append({
                        'pair': pair_name,
                        'z': curr_z[i, ci],
                        'direction': direction,
                        'fwd_ret': fwd_ret,
                        'expected_dir': expected_dir,
                        'pnl': net_pnl,
                        'correct': np.sign(fwd_ret) == expected_dir,
                    })
            
            if not all_events:
                continue
            
            edf = pd.DataFrame(all_events)
            
            # Overall stats
            n_events = len(edf)
            n_unique_bars = len(set([e['pair'] + str(round(e['z'],2)) for e in all_events]))
            
            # Per-pair breakdown
            pair_stats = []
            for pair_name in edf['pair'].unique():
                psub = edf[edf['pair'] == pair_name]
                sprd = spreads.get(pair_name, default_spread)
                wr = np.mean(psub['correct']) * 100
                avg_pnl = np.mean(psub['pnl']) * 10000
                net = avg_pnl - sprd
                win_avg = np.mean(psub[psub['correct']]['pnl']) * 10000 if sum(psub['correct']) > 0 else 0
                loss_avg = np.mean(psub[~psub['correct']]['pnl']) * 10000 if sum(~psub['correct']) > 0 else 0
                pair_stats.append({
                    'pair': pair_name,
                    'n': len(psub),
                    'wr': wr,
                    'avg': avg_pnl,
                    'spread': sprd,
                    'net': net,
                    'win_avg': win_avg,
                    'loss_avg': loss_avg,
                })
            
            psdf = pd.DataFrame(pair_stats)
            
            # Only show if there's at least one pair with positive net
            best = psdf.loc[psdf['net'].idxmax()]
            if best['net'] > 0 and best['n'] >= 5:
                print(f"  Z>{z_thresh:.1f} Fwd={fwd:>2d}m: events={n_events} bars={n_unique_bars}")
                print(f"    Best={best['pair']:>7s} n={int(best['n']):>3d} WR={best['wr']:.0f}% avg={best['avg']:+.2f}bps win={best['win_avg']:+.2f} loss={best['loss_avg']:+.2f} net={best['net']:+.2f}bps")
                print(f"    Events/day: {n_unique_bars/30:.1f}")
    
    # Deep dive: what happens if we use rolling window instead of full-period Z?
    print(f"\n  [Rolling Z-score analysis]")
    # Recompute z with rolling 60-bar window
    roll_window = 60
    roll_z = np.zeros_like(curr_rets)
    for ci in range(len(currencies)):
        for i in range(roll_window, n_lr):
            win = curr_rets[i-roll_window:i, ci]
            m = np.mean(win)
            s = np.std(win)
            roll_z[i, ci] = (curr_rets[i, ci] - m) / s if s > 0 else 0
    
    for z_thresh in [1.5, 2.0, 2.5]:
        for fwd in [5, 10]:
            all_events = []
            ci_local = currencies.index(cname)
            for i in range(roll_window, n_lr - fwd):
                if abs(roll_z[i, ci_local]) < z_thresh:
                    continue
                direction = np.sign(roll_z[i, ci_local])
                for j, sign, pair_name in curr_pairs[cname]:
                    expected_dir = direction * sign
                    fwd_ret = lr[i:i+fwd, j].sum()
                    pnl = fwd_ret * expected_dir
                    all_events.append({
                        'pair': pair_name,
                        'z': roll_z[i, ci_local],
                        'pnl': pnl,
                        'correct': np.sign(fwd_ret) == expected_dir,
                    })
            
            if len(all_events) < 5:
                continue
            
            edf = pd.DataFrame(all_events)
            n_unique = len(edf.drop_duplicates(subset=['pair', 'z']))
            
            pair_stats = []
            for pair_name in edf['pair'].unique():
                psub = edf[edf['pair'] == pair_name]
                sprd = spreads.get(pair_name, default_spread)
                wr = np.mean(psub['correct']) * 100
                avg_pnl = np.mean(psub['pnl']) * 10000
                net = avg_pnl - sprd
                pair_stats.append({'pair': pair_name, 'n': len(psub), 'wr': wr, 'avg': avg_pnl, 'net': net})
            
            psdf = pd.DataFrame(pair_stats)
            best = psdf.loc[psdf['avg'].idxmax()]
            print(f"    RollZ>{z_thresh:.1f} Fwd={fwd:>2d}: Best={best['pair']:>7s} n={int(best['n']):>3d} WR={best['wr']:.0f}% avg={best['avg']:+.2f} net={best['net']:+.2f}  events/day={n_unique/30:.1f}")
