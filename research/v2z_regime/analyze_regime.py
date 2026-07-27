"""
Regime detection for V2+z using available MT5 data.
Find internal microstructure patterns that distinguish winning vs losing trades.
"""
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import MetaTrader5 as mt5
mt5.shutdown()  # Ensure clean state before initialize
import warnings
warnings.filterwarnings('ignore')

symbol = "EURAUD"
print("Connecting to MT5...")
import time
connected = False
for attempt in range(3):
    if mt5.initialize():
        time.sleep(3)
        mt5.symbol_select(symbol, True)
        time.sleep(2)
        connected = True
        break
    print(f"  Attempt {attempt+1} failed, retrying...")
    time.sleep(2)
if not connected:
    print("FAILED to connect to MT5")
    exit()

print("Getting rates...")
rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 50000)
mt5.shutdown()
if rates is None:
    print(f"No data: error {mt5.last_error()}")
    # Try one more time with fresh terminal
    time.sleep(2)
    mt5.initialize()
    time.sleep(3)
    mt5.symbol_select(symbol, True)
    time.sleep(2)
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 50000)
    mt5.shutdown()
    if rates is None:
        print(f"Still no data: error {mt5.last_error()}")
        exit()
print(f"Got {len(rates)} rates")

df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')
print(f"Loaded {len(df)} bars: {df['time'].min()} to {df['time'].max()}")

# Filter to our forward period
df = df[(df['time'] >= '2026-06-08') & (df['time'] < '2026-07-26')].copy()
print(f"Forward period: {len(df)} bars")

# Compute returns and z-scores
df['ret'] = df['close'].diff()
df['atr'] = (df['high'] - df['low']).rolling(20).mean()

Z_WINDOW = 50
def rolling_z(series, w=Z_WINDOW):
    rets = series.diff()
    result = pd.Series(0.0, index=series.index)
    for i in range(w+1, len(series)):
        cur = rets.iloc[i]
        hist = rets.iloc[i-w:i]
        m = hist.mean()
        s = hist.std()
        if s > 1e-14:
            result.iloc[i] = (cur - m) / s
    return result

df['z'] = rolling_z(df['close'])
print(f"Z-score range: [{df['z'].min():.2f}, {df['z'].max():.2f}]")

# ------------------------------------------------
# Simulate strategy and collect per-trade features
# ------------------------------------------------
def simulate_and_collect(df, z_thresh=3.5, stop_a=3.0, trig_a=1.0, gap_a=0.05,
                          max_hold=54, lot=0.75):
    trades = []
    pos = 0
    entry_price = 0.0
    entry_time = None
    entry_idx = 0
    best_price = 0.0
    current_stop = 0.0
    balance = 10000.0
    
    for i in range(Z_WINDOW + 3, len(df)):
        bar = df.iloc[i]
        hour = bar.time.hour
        
        if pos != 0:
            elapsed = (bar.time - entry_time).total_seconds()
            atr_v = bar.atr
            
            # Trailing stop (using bar H/L as proxy for tick-level)
            if pos > 0 and not pd.isna(bar.high):
                if bar.high > best_price:
                    best_price = bar.high
                    if best_price - entry_price > trig_a * atr_v:
                        ns = best_price - gap_a * atr_v
                        if ns > current_stop:
                            current_stop = ns
            elif pos < 0 and not pd.isna(bar.low):
                if bar.low < best_price:
                    best_price = bar.low
                    if entry_price - best_price > trig_a * atr_v:
                        ns = best_price + gap_a * atr_v
                        if ns < current_stop:
                            current_stop = ns
            
            # Stop check
            stopped = False
            if pos > 0 and not pd.isna(bar.low):
                if bar.low <= current_stop:
                    exit_px = current_stop
                    raw = (exit_px - entry_price) * lot * 100000
                    trades.append(collect_features(df, entry_idx, i, entry_price, exit_px, raw, 'stop', pos))
                    balance += raw
                    stopped = True
            elif pos < 0 and not pd.isna(bar.high):
                if bar.high >= current_stop:
                    exit_px = current_stop
                    raw = (entry_price - exit_px) * lot * 100000
                    trades.append(collect_features(df, entry_idx, i, entry_price, exit_px, raw, 'stop', pos))
                    balance += raw
                    stopped = True
            
            if stopped:
                pos = 0
                continue
            
            # Expiry
            if elapsed >= max_hold * 60:
                exit_px = bar.close
                if pos > 0:
                    raw = (exit_px - entry_price) * lot * 100000
                else:
                    raw = (entry_price - exit_px) * lot * 100000
                trades.append(collect_features(df, entry_idx, i, entry_price, exit_px, raw, 'expiry', pos))
                balance += raw
                pos = 0
                continue
        
        # Entry
        if pos == 0 and hour >= 0 and hour < 7:
            z_val = df.iloc[i].z
            if abs(z_val) >= z_thresh:
                dir = -1 if z_val > 0 else 1
                slip = 0.0001
                entry_price = bar.close + slip if dir > 0 else bar.close - slip
                entry_time = bar.time
                entry_idx = i
                atr_v = bar.atr
                current_stop = entry_price - stop_a * atr_v if dir > 0 else entry_price + stop_a * atr_v
                best_price = entry_price
                pos = dir
    
    return trades

def collect_features(df, entry_i, exit_i, entry_px, exit_px, raw_pnl, reason, pos):
    """Extract microstructure features at entry bar."""
    bar = df.iloc[entry_i]
    rng = bar.high - bar.low
    
    # Direction
    signal_dir = 1 if pos > 0 else -1
    
    # 1. Directional persistence
    lookback = 20
    if entry_i >= lookback:
        same_dir = (df['ret'].iloc[entry_i-lookback:entry_i] * signal_dir > 0).sum() / lookback
    else:
        same_dir = 0.5
    
    # 2. Bar smoothness (body/range)
    body = abs(bar.close - bar.open)
    smoothness = body / rng if rng > 0 else 0
    
    # 3. Wick ratio on signal side
    if signal_dir > 0:
        wick_with = bar.high - max(bar.open, bar.close)
    else:
        wick_with = min(bar.open, bar.close) - bar.low
    wick_ratio = wick_with / rng if rng > 0 else 0.5
    
    # 4. Consecutive extreme bars (last 5 bars with z > 0.7*threshold in same direction)
    extreme_count = 0
    z_thresh = 3.5
    for j in range(max(1, entry_i-5), entry_i):
        if abs(df.iloc[j].z) > z_thresh * 0.7:
            if (df.iloc[j].z * signal_dir) > 0:
                extreme_count += 1
    
    # 5. Range z-score
    if entry_i >= 20:
        recent_rng = df['high'].iloc[entry_i-20:entry_i] - df['low'].iloc[entry_i-20:entry_i]
        rng_z = (rng - recent_rng.mean()) / recent_rng.std() if recent_rng.std() > 0 else 0
    else:
        rng_z = 0
    
    # 6. Intra-bar reversal: did gap direction oppose signal?
    prev_close = df['close'].iloc[entry_i-1]
    gap_dir_up = bar.open > prev_close
    gap_opposes = gap_dir_up != (signal_dir > 0)
    
    # 7. Z-score magnitude and sign consistency
    abs_z = abs(df.iloc[entry_i].z)
    
    # 8. Spread level
    spread = bar.spread
    
    # 9. Tick volume anomaly
    if entry_i >= 20:
        vol_mean = df['tick_volume'].iloc[entry_i-20:entry_i].mean()
        vol_ratio = bar.tick_volume / vol_mean if vol_mean > 0 else 1
    else:
        vol_ratio = 1
    
    # 10. NOVEL: Return "shape" - was the extreme bar's return made smoothly or in bursts?
    # Compare 1-min return vs bar range: if |ret| ≈ range, it was smooth; if |ret| << range, it was noisy
    ret_magnitude = abs(bar.ret)
    return_range_ratio = ret_magnitude / rng if rng > 0 else 0
    
    hold_min = (df.iloc[exit_i].time - bar.time).total_seconds() / 60
    
    return {
        'pnl': raw_pnl,
        'won': raw_pnl > 0,
        'same_dir_pct': same_dir,
        'smoothness': smoothness,
        'wick_ratio': wick_ratio,
        'extreme_count': extreme_count,
        'rng_z': rng_z,
        'gap_opposes': int(gap_opposes),
        'abs_z': abs_z,
        'spread': spread,
        'vol_ratio': vol_ratio,
        'return_range_ratio': return_range_ratio,
        'hold_min': hold_min,
        'reason': reason,
    }

print("\nSimulating strategy...")
trades = simulate_and_collect(df)
feat = pd.DataFrame(trades)
print(f"Total trades: {len(feat)}")
print(f"Win rate: {feat['won'].mean()*100:.1f}%")
print(f"Avg PnL: ${feat['pnl'].mean():.2f}")
print(f"Total PnL: ${feat['pnl'].sum():.2f}")

# ------------------------------------------------
# Feature Importance
# ------------------------------------------------
print("\n" + "="*60)
print("FEATURE ANALYSIS: What separates winners from losers?")
print("="*60)

feature_cols = ['same_dir_pct', 'smoothness', 'wick_ratio', 'extreme_count', 
                'rng_z', 'gap_opposes', 'abs_z', 'spread', 'vol_ratio', 'return_range_ratio']

for col in feature_cols:
    winners = feat[feat['won']][col]
    losers = feat[~feat['won']][col]
    if len(winners) > 5 and len(losers) > 5:
        t_stat = (winners.mean() - losers.mean()) / np.sqrt(winners.var()/len(winners) + losers.var()/len(losers))
        print(f"  {col:20s}: win_mean={winners.mean():.4f}  lose_mean={losers.mean():.4f}  t={t_stat:+.2f}")

# ------------------------------------------------
# Simple decision rule: find best single split
# ------------------------------------------------
print("\n" + "-"*60)
print("BEST SINGLE-FEATURE FILTER")
print("-"*60)

base_pnl = feat['pnl'].sum()
base_win = feat['won'].mean()

for col in feature_cols:
    if feat[col].nunique() < 3:
        continue
    best_split = feat[col].median()
    best_delta = -999
    
    for pct in np.arange(10, 91, 5):
        thresh = feat[col].quantile(pct/100)
        mask = feat[col] > thresh
        if mask.sum() < 5 or (~mask).sum() < 5:
            continue
        high_pnl = feat[mask]['pnl'].mean()
        low_pnl = feat[~mask]['pnl'].mean()
        delta = abs(high_pnl - low_pnl)
        if delta > best_delta:
            best_delta = delta
            best_split = thresh
    
    mask = feat[col] > best_split
    high_pnl = feat[mask]['pnl'].mean()
    low_pnl = feat[~mask]['pnl'].mean()
    print(f"  {col:20s} > {best_split:.4f}: high=${high_pnl:.1f} low=${low_pnl:.1f} (N={mask.sum()}/{feat.shape[0]})")

# ------------------------------------------------
# Logistic regression on all features
# ------------------------------------------------
print("\n" + "-"*60)
print("LOGISTIC REGRESSION (trained on ALL forward data)")
print("-"*60)

X = feat[feature_cols].fillna(0)
y = (feat['pnl'] > 0).astype(int)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

clf = LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000)
clf.fit(X_scaled, y)

for col, coef in sorted(zip(feature_cols, clf.coef_[0]), key=lambda x: abs(x[1]), reverse=True):
    print(f"  {col:20s}: {coef:+.4f}")

# Apply filter
probs = clf.predict_proba(X_scaled)[:, 1]
for thresh in np.arange(0.3, 0.8, 0.1):
    mask = probs >= thresh
    if mask.sum() > 0:
        filtered_pnl = feat.loc[mask, 'pnl'].sum()
        print(f"  Prob > {thresh:.1f}: {mask.sum()} trades, ${filtered_pnl:.1f} (total ${base_pnl:.1f})")

# ------------------------------------------------
# THE KEY FINDING: Build simplified rule
# ------------------------------------------------
print("\n" + "="*60)
print("SIMPLIFIED RULE")
print("="*60)

# Find the most important 2-3 features and build a simple boolean rule
top_features = sorted(zip(feature_cols, clf.coef_[0]), key=lambda x: abs(x[1]), reverse=True)[:3]
print(f"Top features: {[f[0] for f in top_features]}")

# Try all 2-feature boolean combinations
print("\n2-feature boolean filters:")
from itertools import combinations

for n_feats in [2, 3]:
    for feats in combinations(feature_cols, n_feats):
        # Try median split on each
        masks = []
        for f in feats:
            med = feat[f].median()
            masks.append(feat[f] > med)
        
        combined = pd.concat(masks, axis=1).all(axis=1)
        if combined.sum() < 5:
            continue
        
        filtered = feat[combined]
        excluded = feat[~combined]
        if filtered['pnl'].mean() > excluded['pnl'].mean() + 1:
            print(f"  {feats} all > median: {combined.sum()} trades, avg=${filtered['pnl'].mean():.1f} (rest=${excluded['pnl'].mean():.1f})")

print("\nDone!")
