"""
Parse REAL V2z trades from MT5 Terminal tester log.
Extracts OPEN + ENTRYBAR + CLOSE triplets for each trade.
"""
import re
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

log_path = Path("C:\\Users\\Arjun Sasi\\AppData\\Roaming\\MetaQuotes\\Terminal\\D0E8209F77C8CF37AD8BF550E51FF075\\Tester\\logs\\20260726.log")
if not log_path.exists():
    print("Log not found")
    exit()

# Read the last 10MB of the file (the latest runs)
sz = log_path.stat().st_size
with open(log_path, 'rb') as f:
    f.seek(max(0, sz - 10_000_000))
    raw = f.read()
text = raw.decode('utf-16-le', errors='replace')
lines = text.split('\n')
print(f"Read {len(lines)} lines from end of log")

# Find the last backtest run (look for "V2z_v2_Clean" patterns)
# Strategy: scan backwards, find the last occurrence of ENTRYBAR, then match all
trades = []
open_data = None
entrybar_data = None

for line in lines:
    if 'ENTRYBAR EURAUD' in line:
        m = re.search(r'ENTRYBAR EURAUD z=([\d.eE+-]+) open=([\d.]+) high=([\d.]+) low=([\d.]+) close=([\d.]+) tv=([\d.]+) sp=([\d.]+)', line)
        if m:
            entrybar_data = {
                'z': float(m.group(1)),
                'bar_open': float(m.group(2)),
                'bar_high': float(m.group(3)),
                'bar_low': float(m.group(4)),
                'bar_close': float(m.group(5)),
                'tick_vol': float(m.group(6)),
                'spread': float(m.group(7)),
            }
    
    elif 'OPEN EURAUD' in line and 'ENTRYBAR' not in line:
        m = re.search(r'(\d{4}\.\d{2}\.\d{2} \d{2}:\d{2}:\d{2}).*OPEN.*dir=(-?\d+)\s+entry=([\d.]+)\s+sl=([\d.]+)\s+atr=([\d.eE+-]+).*slip=([\d.]+)', line)
        if m:
            open_data = {
                'time': m.group(1),
                'dir': int(m.group(2)),
                'entry': float(m.group(3)),
                'sl': float(m.group(4)),
                'atr': float(m.group(5)),
                'slip': float(m.group(6)),
            }
    
    elif 'CLOSE EURAUD' in line and open_data is not None and entrybar_data is not None:
        m = re.search(r'rsn=(\w+)\s+exit=([\d.]+)\s+raw=([\d.eE+-]+)\s+comm=([\d.]+)\s+pnl=([\d.eE+-]+)', line)
        if m:
            trades.append({
                'entry_time': open_data['time'],
                'dir': open_data['dir'],
                'entry': open_data['entry'],
                'sl': open_data['sl'],
                'atr': open_data['atr'],
                'z': entrybar_data['z'],
                'bar_open': entrybar_data['bar_open'],
                'bar_high': entrybar_data['bar_high'],
                'bar_low': entrybar_data['bar_low'],
                'bar_close': entrybar_data['bar_close'],
                'tick_vol': entrybar_data['tick_vol'],
                'spread_raw': entrybar_data['spread'],
                'exit_reason': m.group(1),
                'exit_price': float(m.group(2)),
                'raw_pnl': float(m.group(3)),
                'pnl': float(m.group(5)),
            })
            open_data = None
            entrybar_data = None

print(f"Parsed {len(trades)} trades")

if len(trades) == 0:
    print("No trades found")
    exit()

df = pd.DataFrame(trades)
df['won'] = df['pnl'] > 0
print(f"Period: {df['entry_time'].min()} to {df['entry_time'].max()}")
print(f"Win rate: {df['won'].mean()*100:.1f}%")
print(f"Total PnL: ${df['pnl'].sum():.2f}")
print(f"Commission: {df['raw_pnl'].sum() - df['pnl'].sum():.2f}")

# Compute features
df['range'] = df['bar_high'] - df['bar_low']
df['body'] = (df['bar_close'] - df['bar_open']).abs()
df['smoothness'] = df['body'] / df['range'].clip(lower=1e-10)
df['abs_z'] = df['z'].abs()
df['body_pips'] = df['body'] * 10000
df['range_pips'] = df['range'] * 10000
df['spread_pips'] = df['spread_raw'] / 10.0

# Wick in signal direction (fading direction)
long_mask = df['dir'] == 1
short_mask = df['dir'] == -1
df['wick_with_dir'] = 0.0
bb = df[['bar_open', 'bar_close']]
df.loc[long_mask, 'wick_with_dir'] = (df.loc[long_mask, 'bar_high'] - bb.loc[long_mask].max(axis=1)) / df.loc[long_mask, 'range'].clip(lower=1e-10)
df.loc[short_mask, 'wick_with_dir'] = (bb.loc[short_mask].min(axis=1) - df.loc[short_mask, 'bar_low']) / df.loc[short_mask, 'range'].clip(lower=1e-10)

# Gap opposes trade
prev_close = None  # We don't have prev bar, skip gap

# Ret/Range ratio
df['ret_range_ratio'] = df['body'] / df['range'].clip(lower=1e-10)

# Volume ratio (compared to recent - use simple BTS approach)
df['vol_ratio'] = df['tick_vol'] / df['tick_vol'].median()

print("\n" + "="*70)
print("FEATURE ANALYSIS (REAL MT5 FORWARD TRADES)")
print("="*70)

features = ['abs_z', 'smoothness', 'spread_pips', 'tick_vol', 'range_pips',
            'body_pips', 'wick_with_dir', 'ret_range_ratio', 'vol_ratio']

for col in features:
    w = df[df['won']][col].dropna()
    l = df[~df['won']][col].dropna()
    if len(w) > 3 and len(l) > 3:
        wm, lm = w.mean(), l.mean()
        t = (wm - lm) / np.sqrt(w.var()/len(w) + l.var()/len(l)) if w.var() > 0 and l.var() > 0 else 0
        print(f"  {col:18s}: win={wm:10.4f}  lose={lm:10.4f}  t={t:+7.2f}")

# Single-feature filters
print("\n" + "-"*70)
print("BEST SINGLE-FEATURE FILTERS")
print("-"*70)
base_pnl = df['pnl'].sum()
for col in features:
    v = df[col].dropna()
    if v.nunique() < 3:
        continue
    best_h, best_t = -9999, None
    for p in range(5, 96, 5):
        t = v.quantile(p/100)
        m = df[col] > t
        if m.sum() < 3 or (~m).sum() < 3:
            continue
        h = df.loc[m, 'pnl'].mean()
        if h > best_h:
            best_h, best_t = h, t
    if best_t is not None:
        m = df[col] > best_t
        hp = df.loc[m, 'pnl'].mean()
        lp = df.loc[~m, 'pnl'].mean()
        print(f"  {col:18s} > {best_t:8.4f}: high=${hp:+7.2f}  low=${lp:+7.2f}  N={m.sum()}/{df.shape[0]}")

# Logistic regression
print("\n" + "-"*70)
print("LOGISTIC REGRESSION")
print("-"*70)
X = df[features].fillna(0)
y = (df['pnl'] > 0).astype(int)
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
        fp = df.loc[m, 'pnl'].sum()
        print(f"  Prob > {th:.1f}: {m.sum()} trades, ${fp:+7.2f} (total ${base_pnl:+7.2f})")

# Exit analysis
print("\n" + "-"*70)
print("EXIT REASON ANALYSIS")
print("-"*70)
for reason in df['exit_reason'].unique():
    sub = df[df['exit_reason'] == reason]
    print(f"  {reason:6s}: {len(sub):3d} trades, WR={sub['won'].mean()*100:.0f}%, avg=${sub['pnl'].mean():+.1f}")

# Best/worst trades
print("\n" + "="*70)
print("BEST 5 TRADES")
print("="*70)
for _, r in df.sort_values('pnl', ascending=False).head(5).iterrows():
    print(f"  ${r['pnl']:+7.2f}  |z|={r['abs_z']:5.1f}  sprd={r['spread_pips']:4.1f}  tv={r['tick_vol']:3.0f}  body={r['body_pips']:5.1f}  rng={r['range_pips']:5.1f}  wick={r['wick_with_dir']:.2f}  rsn={r['exit_reason']}")

print("\nWORST 5 TRADES")
for _, r in df.sort_values('pnl').head(5).iterrows():
    print(f"  ${r['pnl']:+7.2f}  |z|={r['abs_z']:5.1f}  sprd={r['spread_pips']:4.1f}  tv={r['tick_vol']:3.0f}  body={r['body_pips']:5.1f}  rng={r['range_pips']:5.1f}  wick={r['wick_with_dir']:.2f}  rsn={r['exit_reason']}")

print("\nDone!")
