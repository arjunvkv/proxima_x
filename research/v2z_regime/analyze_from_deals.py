"""
Parse V2z trades from tester log by matching deal events.
Each trade = buy deal #N followed by sell deal #N+1 (or vice versa).
Match with OPEN/ENTRYBAR lines by sequential position.
"""
import re
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

log_path = "C:\\Users\\Arjun Sasi\\AppData\\Roaming\\MetaQuotes\\Tester\\D0E8209F77C8CF37AD8BF550E51FF075\\Agent-127.0.0.1-3000\\logs\\20260726.log"
with open(log_path, 'rb') as f:
    sz = f.seek(0, 2)
    f.seek(max(0, sz - 2_000_000))
    raw = f.read()
text = raw.decode('utf-16-le', errors='replace')
lines = text.split('\n')
print(f"Loaded {len(lines)} lines from tail")

# Parse deal events
deals = []
for l in lines:
    m = re.search(r'deal #(\d+) (buy|sell) [\d.]+ (\w+) at ([\d.]+) done', l)
    if m:
        deals.append({'ticket': int(m.group(1)), 'side': m.group(2), 'symbol': m.group(3), 'price': float(m.group(4))})

print(f"Found {len(deals)} deals")

# Match consecutive deals by ticket into trades
trades_from_deals = []
i = 0
while i < len(deals) - 1:
    d1 = deals[i]
    # Find the matching close deal (opposite side)
    for j in range(i+1, len(deals)):
        d2 = deals[j]
        if d2['side'] != d1['side'] and d2['symbol'] == d1['symbol']:
            # Compute pnl
            lots = 0.75
            cs = 100000
            if d1['side'] == 'buy':
                raw_pnl = (d2['price'] - d1['price']) * lots * cs
            else:
                raw_pnl = (d1['price'] - d2['price']) * lots * cs
            trades_from_deals.append({
                'ticket_open': d1['ticket'],
                'ticket_close': d2['ticket'],
                'side': d1['side'],
                'entry_price': d1['price'],
                'exit_price': d2['price'],
                'raw_pnl': raw_pnl,
            })
            i = j + 1
            break
    else:
        i += 1

# Now also parse OPEN/ENTRYBAR lines
entries = []
for l in lines:
    if 'ENTRYBAR EURAUD' in l:
        m = re.search(r'z=([\d.eE+-]+) open=([\d.]+) high=([\d.]+) low=([\d.]+) close=([\d.]+) tv=([\d.]+) sp=([\d.]+)', l)
        if m:
            entries.append({
                'z': float(m.group(1)),
                'bar_open': float(m.group(2)),
                'bar_high': float(m.group(3)),
                'bar_low': float(m.group(4)),
                'bar_close': float(m.group(5)),
                'tick_vol': float(m.group(6)),
                'spread_raw': float(m.group(7)),
            })

print(f"Open deals: {len(trades_from_deals)}, ENTRYBAR lines: {len(entries)}")

# Match by sequential position
n = min(len(trades_from_deals), len(entries))
for i in range(n):
    trades_from_deals[i].update(entries[i])

df = pd.DataFrame(trades_from_deals[:n])
if len(df) == 0:
    print("No trades matched")
    exit()

df['won'] = df['raw_pnl'] > 0
print(f"\nMatched {len(df)} trades")
print(f"Total raw PnL: ${df['raw_pnl'].sum():.2f}")
print(f"Win rate: {df['won'].mean()*100:.1f}%")

# Features
df['range'] = df['bar_high'] - df['bar_low']
df['body'] = (df['bar_close'] - df['bar_open']).abs()
df['smoothness'] = df['body'] / df['range'].clip(lower=1e-10)
df['abs_z'] = df['z'].abs()
df['body_pips'] = df['body'] * 10000
df['range_pips'] = df['range'] * 10000
df['spread_pips'] = df['spread_raw'] / 10.0

long_mask = df['side'] == 'buy'
short_mask = df['side'] == 'sell'
bb = df[['bar_open', 'bar_close']]
df['wick_with_dir'] = 0.0
df.loc[long_mask, 'wick_with_dir'] = (df.loc[long_mask, 'bar_high'] - bb.loc[long_mask].max(axis=1)) / df.loc[long_mask, 'range'].clip(lower=1e-10)
df.loc[short_mask, 'wick_with_dir'] = (bb.loc[short_mask].min(axis=1) - df.loc[short_mask, 'bar_low']) / df.loc[short_mask, 'range'].clip(lower=1e-10)

df['ret_range_ratio'] = df['body'] / df['range'].clip(lower=1e-10)
df['vol_ratio'] = df['tick_vol'] / df['tick_vol'].median()

# Analysis
features = ['abs_z', 'smoothness', 'spread_pips', 'tick_vol', 'range_pips',
            'body_pips', 'wick_with_dir', 'ret_range_ratio', 'vol_ratio']

print("\n" + "="*70)
print("FEATURE ANALYSIS")
print("="*70)
for col in features:
    w = df[df['won']][col].dropna()
    l = df[~df['won']][col].dropna()
    if len(w) > 3 and len(l) > 3:
        wm, lm = w.mean(), l.mean()
        t = (wm - lm) / np.sqrt(w.var()/len(w) + l.var()/len(l))
        print(f"  {col:18s}: win={wm:10.4f}  lose={lm:10.4f}  t={t:+7.2f}")

print("\n" + "-"*70)
print("BEST SINGLE-FEATURE FILTERS (mean pnl when feature > threshold)")
print("-"*70)
base_pnl = df['raw_pnl'].sum()
for col in features:
    v = df[col].dropna()
    if v.nunique() < 3: continue
    best_h, best_t = -9999, None
    for p in range(5, 96, 5):
        t = v.quantile(p/100)
        m = df[col] > t
        if m.sum() < 3 or (~m).sum() < 3: continue
        h = df.loc[m, 'raw_pnl'].mean()
        if h > best_h: best_h, best_t = h, t
    if best_t is not None:
        m = df[col] > best_t
        hp = df.loc[m, 'raw_pnl'].mean()
        lp = df.loc[~m, 'raw_pnl'].mean()
        print(f"  {col:18s} > {best_t:8.4f}: high=${hp:+7.2f}  low=${lp:+7.2f}  N={m.sum()}/{df.shape[0]}")

print("\n" + "-"*70)
print("LOGISTIC REGRESSION")
print("-"*70)
X = df[features].fillna(0)
y = (df['raw_pnl'] > 0).astype(int)
scaler = StandardScaler()
X_std = scaler.fit_transform(X)
clf = LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000)
clf.fit(X_std, y)
for c, coef in sorted(zip(features, clf.coef_[0]), key=lambda x: abs(x[1]), reverse=True):
    print(f"  {c:18s}: {coef:+8.4f}")

probs = clf.predict_proba(X_std)[:, 1]
for th in np.arange(0.3, 0.8, 0.1):
    m = probs >= th
    if m.sum() >= 3:
        print(f"  Prob > {th:.1f}: {m.sum()} trades, ${df.loc[m, 'raw_pnl'].sum():+7.2f} (total ${base_pnl:+7.2f})")

# Best/worst trades
print("\n" + "="*70)
print("BEST 5 TRADES")
print("="*70)
for _, r in df.sort_values('raw_pnl', ascending=False).head(5).iterrows():
    print(f"  ${r['raw_pnl']:+7.2f}  |z|={r['abs_z']:5.1f}  sprd={r['spread_pips']:4.1f}  tv={r['tick_vol']:3.0f}  body={r['body_pips']:5.1f}  rng={r['range_pips']:5.1f}  wick={r['wick_with_dir']:.2f}")

print("\nWORST 5 TRADES")
for _, r in df.sort_values('raw_pnl').head(5).iterrows():
    print(f"  ${r['raw_pnl']:+7.2f}  |z|={r['abs_z']:5.1f}  sprd={r['spread_pips']:4.1f}  tv={r['tick_vol']:3.0f}  body={r['body_pips']:5.1f}  rng={r['range_pips']:5.1f}  wick={r['wick_with_dir']:.2f}")

print("\nDone!")
