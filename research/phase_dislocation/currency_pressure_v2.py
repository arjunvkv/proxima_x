"""Currency Inventory Pressure — find best single pair per currency."""
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

# For each currency, find which single pair reacts most reliably
print("=" * 60)
print("BEST SINGLE PAIR PER CURRENCY")
print("=" * 60)

spreads = {
    'EURUSD': 1.5, 'GBPUSD': 2.0, 'USDJPY': 1.8, 'AUDUSD': 1.5, 'NZDUSD': 2.0,
    'USDCAD': 2.0, 'USDCHF': 2.0, 'EURJPY': 2.5, 'GBPJPY': 4.0, 'EURGBP': 2.0,
    'EURAUD': 2.5, 'EURCHF': 2.0, 'EURCAD': 2.0, 'EURNZD': 2.5,
    'GBPAUD': 2.5, 'GBPCAD': 2.5, 'GBPCHF': 3.0, 'GBPNZD': 3.0,
    'AUDJPY': 2.5, 'AUDCAD': 2.5, 'AUDCHF': 2.5, 'AUDNZD': 2.5,
    'NZDJPY': 3.0, 'NZDCAD': 2.5, 'NZDCHF': 2.5,
    'CADJPY': 2.5, 'CADCHF': 2.5, 'CHFJPY': 3.0,
}
default_spread = 2.0

z_thresh = 2.5

for ci, c in enumerate(currencies):
    pairs = curr_pairs[c]
    if len(pairs) == 0:
        continue
    
    for fwd in [5, 10, 15]:
        pair_results = {}
        
        for j, sign, pair_name in pairs:
            expected = []  # expected dir * actual forward return
            correct = []
            rets_list = []
            
            for i in range(30, n_lr - fwd):
                if abs(curr_z[i, ci]) < z_thresh:
                    continue
                direction = np.sign(curr_z[i, ci])
                expected_dir = direction * sign
                fwd_ret = lr[i:i+fwd, j].sum()
                rets_list.append(fwd_ret * expected_dir)  # positive = correct direction
                correct.append(np.sign(fwd_ret) == expected_dir)
                expected.append(fwd_ret)
            
            if len(correct) >= 5:
                wr = np.mean(correct) * 100
                avg_ret = np.mean(rets_list) * 10000 if rets_list else 0
                pair_results[pair_name] = {
                    'n': len(correct),
                    'wr': wr,
                    'avg_ret_bps': avg_ret,
                    'avg_abs_ret': np.mean(np.abs(expected)) * 10000,
                }
        
        if pair_results:
            best = max(pair_results.items(), key=lambda x: x[1]['avg_ret_bps'])
            pn, pd = best
            sprd = spreads.get(pn, default_spread)
            print(f"  {c:>4s} fwd={fwd:>2d}m: Best={pn:>7s} n={pd['n']:>3d} WR={pd['wr']:.0f}% avg={pd['avg_ret_bps']:+.2f}bps spread={sprd:.1f}bps net={pd['avg_ret_bps']-sprd:+.2f}bps")

# For the best combo (JPY with NZD/CHF pairs), deep dive
print("\n" + "=" * 60)
print("DEEP DIVE: JPY PRESSURE")
print("=" * 60)

ci = currencies.index('JPY')
for j, sign, pair_name in curr_pairs['JPY']:
    sprd = spreads.get(pair_name, default_spread)
    for fwd in [5, 10, 15]:
        rets_win = []
        rets_loss = []
        total_rets = []
        for i in range(30, n_lr - fwd):
            if abs(curr_z[i, ci]) < 2.5:
                continue
            direction = np.sign(curr_z[i, ci])
            expected_dir = direction * sign
            fwd_ret = lr[i:i+fwd, j].sum()
            realized_ret = fwd_ret * expected_dir  # positive = we made money
            total_rets.append(realized_ret)
            if realized_ret > 0:
                rets_win.append(realized_ret)
            else:
                rets_loss.append(realized_ret)

        if len(total_rets) >= 5:
            wr = len(rets_win) / len(total_rets) * 100
            avg_win = np.mean(rets_win) * 10000 if rets_win else 0
            avg_loss = np.mean(rets_loss) * 10000 if rets_loss else 0
            avg_all = np.mean(total_rets) * 10000
            print(f"  {pair_name:>7s} fwd={fwd:>2d}: n={len(total_rets):>3d} WR={wr:.0f}% avg={avg_all:+.2f}bps win={avg_win:+.2f} loss={avg_loss:+.2f} spread={sprd:.1f}")
