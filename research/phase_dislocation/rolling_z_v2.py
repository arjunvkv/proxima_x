"""
Fast rolling Z-score backtest — vectorized rolling mean/std per window.
"""
import numpy as np
import pandas as pd
import os, glob, json

DATADIR = 'research/phase_dislocation/dukascopy_data'
Z_THRESH = 2.0
FWD = 5
LOT = 0.5
CONTRACT = 100000
WINDOWS = [100, 500, 1000, 2000, 4000]
CURRENCIES = ['USD','EUR','JPY','GBP','AUD','NZD','CAD','CHF']

SPREADS = {'EURUSD':1.5,'GBPUSD':2.0,'USDJPY':1.8,'AUDUSD':1.5,'NZDUSD':2.0,'USDCAD':2.0,'USDCHF':2.0,'EURJPY':2.5,'GBPJPY':4.0,'EURGBP':2.0,'EURAUD':2.5,'EURCHF':2.0,'EURCAD':2.0,'EURNZD':2.5,'GBPAUD':2.5,'GBPCAD':2.5,'GBPCHF':3.0,'GBPNZD':3.0,'AUDJPY':2.5,'AUDCAD':2.5,'AUDCHF':2.5,'AUDNZD':2.5,'NZDJPY':3.0,'NZDCAD':2.5,'NZDCHF':2.5,'CADJPY':2.5,'CADCHF':2.5,'CHFJPY':3.0}
BEST_PAIR = {"USD":"AUDUSD","EUR":"EURUSD","JPY":"NZDJPY","GBP":"GBPUSD","AUD":"AUDUSD","NZD":"NZDUSD","CAD":"NZDCAD","CHF":"USDCHF"}

def base_quote(pair):
    for c in ['AUD','CAD','CHF','EUR','GBP','JPY','NZD','USD']:
        if pair.startswith(c): return c, pair[len(c):]
    return None, None

files = sorted(glob.glob(os.path.join(DATADIR, '*.parquet')))
all_data = {}
for fpath in files:
    pair = os.path.basename(fpath).replace('.parquet','').upper()
    all_data[pair] = pd.read_parquet(fpath).sort_values('timestamp')

pairs_list = sorted(all_data.keys())
ref = all_data[pairs_list[0]]
timestamps = ref['timestamp'].values
n = len(timestamps)

pm = np.full((n, len(pairs_list)), np.nan)
for j, pair in enumerate(pairs_list):
    df = all_data[pair]
    times, prices = df['timestamp'].values, df['close'].values
    for i, t in enumerate(timestamps):
        m = np.where(times == t)[0]
        pm[i,j] = prices[m[0]] if len(m) > 0 else (pm[i-1,j] if i > 0 else np.nan)

first = np.where(~np.isnan(pm).any(axis=1))[0]
if len(first) > 0: pm = pm[first[0]:]
n = len(pm)
lr = np.diff(np.log(pm), axis=0)
print(f"Data: {n} bars, {len(pairs_list)} pairs")

curr_pairs = {c: [] for c in CURRENCIES}
for j, pair in enumerate(pairs_list):
    b,q = base_quote(pair)
    if b and q:
        if b in curr_pairs: curr_pairs[b].append((j, 1.0, pair))
        if q in curr_pairs: curr_pairs[q].append((j, -1.0, pair))

# Pre-compute currency returns
nlr = lr.shape[0]
curr_rets = np.zeros((nlr, len(CURRENCIES)))
for ci, c in enumerate(CURRENCIES):
    pairs = curr_pairs[c]
    if len(pairs) == 0: continue
    cols = [p[0] for p in pairs]
    signs = [p[1] for p in pairs]
    raw = lr[:, cols] * np.array(signs)
    vols = np.nanstd(raw[:min(200, nlr)], axis=0) + 1e-10
    w = (1.0/vols) / np.sum(1.0/vols)
    curr_rets[:, ci] = np.nansum(raw * w, axis=1)

# For each pair, pre-compute price at each bar for dollar PnL
pair_idx = {}
for j, p in enumerate(pairs_list):
    pair_idx[p] = j

print(f"\n{'Window':>6s} | {'Trades':>7s} | {'WR':>6s} | {'Mean($)':>10s} | {'Med($)':>10s} | {'P5($)':>10s} | {'P95($)':>10s} | {'NetBps':>8s} | {'MinH':>5s}")
print("-" * 80)

for W in WINDOWS:
    trades = []

    # Compute rolling mean/std for each currency using pandas (fast)
    cr_df = pd.DataFrame(curr_rets, columns=CURRENCIES)
    roll_mean = cr_df.rolling(W, min_periods=5).mean().values
    roll_std = cr_df.rolling(W, min_periods=5).std().values

    for i in range(W, nlr - FWD):
        for ci, c in enumerate(CURRENCIES):
            m = roll_mean[i, ci]
            s = roll_std[i, ci]
            if np.isnan(m) or np.isnan(s) or s < 1e-12:
                continue

            ret = curr_rets[i, ci]
            if np.isnan(ret):
                continue
            z = (ret - m) / s
            if abs(z) < Z_THRESH:
                continue

            pair = BEST_PAIR[c]
            if pair not in pair_idx:
                continue
            j = pair_idx[pair]

            sign = 1.0
            for idx, sg, pn in curr_pairs[c]:
                if pn == pair:
                    sign = sg
                    break

            direction = 1 if z > 0 else -1
            expected_dir = direction * sign

            entry_price = pm[i, j]
            exit_price = pm[i+FWD, j]
            fwd_ret = np.sum(lr[i:i+FWD, j])
            pnl = fwd_ret * expected_dir

            # Dollar PnL (use expected_dir for sign consistency with bps)
            base_price_diff = LOT * CONTRACT * (exit_price - entry_price) * expected_dir
            if pair in ('EURUSD','GBPUSD','AUDUSD','NZDUSD'):
                gross_dollars = base_price_diff
            elif pair in ('USDJPY',):
                usdjpy = pm[i+FWD, pair_idx['USDJPY']] if 'USDJPY' in pair_idx else 100
                gross_dollars = base_price_diff / usdjpy
            elif pair.endswith('JPY'):
                usdjpy = pm[i+FWD, pair_idx['USDJPY']] if 'USDJPY' in pair_idx else 100
                gross_dollars = base_price_diff / usdjpy
            elif pair.endswith('CAD'):
                usdcad = pm[i+FWD, pair_idx['USDCAD']] if 'USDCAD' in pair_idx else 1
                gross_dollars = base_price_diff / usdcad
            elif pair.endswith('CHF'):
                usdchf = pm[i+FWD, pair_idx['USDCHF']] if 'USDCHF' in pair_idx else 1
                gross_dollars = base_price_diff / usdchf
            else:
                gross_dollars = base_price_diff

            pnl_bps = fwd_ret * 10000 * expected_dir

            trades.append({
                'currency': c, 'pair': pair, 'pnl_bps': pnl_bps,
                'pnl_dollars': gross_dollars,
                'correct': np.sign(fwd_ret) == expected_dir,
                'hist_len': min(i, W),
            })

    if not trades:
        print(f"{W:>6d} | {'0':>7} | {'N/A':>6}")
        continue

    tdf = pd.DataFrame(trades)
    n_t = len(tdf)
    wr = np.mean(tdf['correct']) * 100
    d = tdf['pnl_dollars']
    bp = np.mean(tdf['pnl_bps'])
    hist_min = int(tdf['hist_len'].min())

    # Per-pair with net bps
    pair_rows = []
    for pair in tdf['pair'].unique():
        psub = tdf[tdf['pair']==pair]
        if len(psub) < 5: continue
        pair_rows.append({
            'pair': pair, 'n': len(psub),
            'wr': np.mean(psub['correct'])*100,
            'mean_bps': np.mean(psub['pnl_bps']),
            'net_bps': np.mean(psub['pnl_bps']) - SPREADS.get(pair, 2.0),
            'mean_dollars': np.mean(psub['pnl_dollars']),
        })
    pair_rows.sort(key=lambda x: x['net_bps'], reverse=True)
    best_pair = pair_rows[0]['pair'] if pair_rows else 'NONE'
    best_net = pair_rows[0]['net_bps'] if pair_rows else 0

    print(f"{W:>6d} | {n_t:>7d} | {wr:>5.1f}% | ${d.mean():>+8.2f} | ${d.median():>+8.2f} | ${np.percentile(d,5):>+8.2f} | ${np.percentile(d,95):>+8.2f} | {bp:>+7.2f} | {hist_min:>5d}")
    print(f"  Best pair: {best_pair} net={best_net:+.2f}bps")
    for pr in pair_rows[:5]:
        print(f"    {pr['pair']:>7s}: n={pr['n']:>4d} WR={pr['wr']:5.1f}% meanBps={pr['mean_bps']:+.2f} netBps={pr['net_bps']:+.2f} ${pr['mean_dollars']:+.2f}")

print("\nDone.")
