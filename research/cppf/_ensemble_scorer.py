"""Multi-feature ensemble scorer for 30 trades/day at 65%+ WR."""
import pandas as pd, numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).parent.parent.parent / "research" / "phase_dislocation" / "dukascopy_data"
ALL_PAIRS = ['gbpnzd', 'eurnzd', 'gbpaud', 'euraud', 'gbpcad', 'audnzd', 'eurgbp', 'gbpchf']
LEG_MAP = {
    'gbpnzd': ('gbpusd', 'nzdusd'), 'eurnzd': ('eurusd', 'nzdusd'),
    'gbpaud': ('gbpusd', 'audusd'), 'euraud': ('eurusd', 'audusd'),
    'gbpcad': ('gbpusd', 'usdcad'), 'audnzd': ('audusd', 'nzdusd'),
    'eurgbp': ('eurusd', 'gbpusd'), 'gbpchf': ('gbpusd', 'usdchf'),
}
SPREAD_MAP = {
    'gbpnzd': 5.0, 'eurnzd': 4.0, 'gbpaud': 4.0, 'euraud': 3.0,
    'gbpcad': 4.0, 'audnzd': 3.0, 'eurgbp': 2.0, 'gbpchf': 4.0,
}

# Load data
data = {}
needed = set(ALL_PAIRS)
for v in LEG_MAP.values():
    needed.add(v[0]); needed.add(v[1])
for p in sorted(needed):
    f = DATA_DIR / f'{p}.parquet'
    if f.exists():
        data[p] = pd.read_parquet(f).set_index('timestamp').astype(float)
    else:
        print(f"  WARN: {p}.parquet not found")

available = [p for p in ALL_PAIRS if p in data]
print(f"Loaded {len(data)} symbols, {len(available)} tradeable pairs")

# Precompute features for each bar of each pair
def compute_features(pair):
    """Compute all features for every bar."""
    d = data[pair]
    c = d['close']; h = d['high']; l_ = d['low']
    atr = (h - l_).shift(1).rolling(20).mean()

    # ---------- Raw returns ----------
    ret = c.diff()
    ret_pips = ret * 10000

    # ---------- Z-scores on multiple windows ----------
    def z(close, window):
        r = close.diff()
        mu = r.shift(1).rolling(window).mean()
        sig = r.shift(1).rolling(window).std().clip(1e-10)
        return (r - mu) / sig

    z20 = z(c, 20)
    z50 = z(c, 50)
    z100 = z(c, 100)

    # ---------- Leg features ----------
    base, quote = LEG_MAP[pair]
    z_base = z(data[base]['close'], 50)
    z_quote = z(data[quote]['close'], 50)
    leg_max = pd.concat([z_base.abs(), z_quote.abs()], axis=1).max(axis=1)

    # ---------- Market breadth ----------
    cross_z = pd.DataFrame()
    for p in available:
        if p == pair: continue
        cross_z[p] = z(data[p]['close'], 50).abs()
    breadth = (cross_z > 2.0).sum(axis=1)

    # ---------- ATR regime ----------
    atr_median = atr.rolling(50).median()
    atr_ratio = atr / atr_median

    # ---------- Bollinger distance ----------
    ma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std().clip(1e-10)
    bb_dist = (c - ma20) / std20

    # ---------- Consecutive bars ----------
    same_dir = ret.gt(0).astype(int)
    streak = same_dir.groupby((same_dir != same_dir.shift()).cumsum()).cumcount() + 1
    streak = streak * (ret.gt(0).astype(int) * 2 - 1)

    # ---------- Z-rate-of-change (acceleration) ----------
    z_accel = z50.diff()

    # ---------- Target: did we make money fading this bar? ----------
    # For a short mean-reversion trade: fade the direction
    # If bar closes up (positive ret), we'd SHORT, so we profit if price goes DOWN next
    # Target = 1 if next 3-bar change is opposite direction
    next_ret_3 = c.shift(-3) - c
    target = ((ret > 0) & (next_ret_3 < 0) | (ret < 0) & (next_ret_3 > 0)).astype(int)
    # But we only care about bars with LARGE moves
    large_move = ret.abs() > c * 0.0001  # at least 1 pip

    # Assemble features
    features = pd.DataFrame({
        'z20': z20, 'z50': z50, 'z100': z100,
        'ret_pips': ret_pips,
        'leg_max': leg_max,
        'breadth': breadth,
        'atr_ratio': atr_ratio,
        'bb_dist': bb_dist,
        'streak': streak,
        'z_accel': z_accel,
    })
    return features, target, large_move

# Build dataset
print("\nBuilding feature dataset...")
all_features = []
all_targets = []
all_weights = []
all_pair_names = []
for pair in available:
    f, t, w = compute_features(pair)
    valid = f.notna().all(axis=1) & t.notna()
    all_features.append(f[valid])
    all_targets.append(t[valid])
    all_weights.append(w[valid])
    all_pair_names.append(pd.Series(pair, index=f[valid].index))
    print(f"  {pair}: {valid.sum():>6d} bars")

features = pd.concat(all_features)
targets = pd.concat(all_targets).astype(int)
weights = pd.concat(all_weights)
pair_names = pd.concat(all_pair_names)

print(f"\nTotal samples: {len(features)}")
print(f"Target mean reversion rate (unconditional): {targets.mean():.1%}")
print(f"Bars with large moves (>1pip): {weights.mean():.1%}")

# ═══════════════════════════════════════════════════════════
# TEST 1: Individual feature predictive power
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 1: Individual feature predictive power")
print("(Each row = strategy using that feature alone)")

features['session'] = features.index.hour
features['hour_sin'] = np.sin(2 * np.pi * features['session'] / 24)
features['hour_cos'] = np.cos(2 * np.pi * features['session'] / 24)

feature_configs = [
    ('z50>3.0', lambda f: (f.z50.abs() > 3.0)),
    ('z50>3.0+leg<0.5', lambda f: (f.z50.abs() > 3.0) & (f.leg_max < 0.5)),
    ('z50>3.0+leg<0.7', lambda f: (f.z50.abs() > 3.0) & (f.leg_max < 0.7)),
    ('z50>3.0+leg<1.0', lambda f: (f.z50.abs() > 3.0) & (f.leg_max < 1.0)),
    ('z100>2.5', lambda f: (f.z100.abs() > 2.5)),
    ('z100>3.0', lambda f: (f.z100.abs() > 3.0)),
    ('z20>3.5', lambda f: (f.z20.abs() > 3.5)),
    ('bb_dist>2.5', lambda f: (f.bb_dist.abs() > 2.5)),
    ('atr_ratio>1.3', lambda f: (f.atr_ratio > 1.3)),
    ('atr_ratio<0.7', lambda f: (f.atr_ratio < 0.7)),
    ('breadth<2', lambda f: (f.breadth < 2)),
    ('breadth==0', lambda f: (f.breadth == 0)),
    ('streak>3', lambda f: (f.streak.abs() > 3)),
    ('z_accel>0.5', lambda f: (f.z_accel.abs() > 0.5)),
    ('session Sydney', lambda f: (f.session >= 21) | (f.session <= 23)),
    ('session Asia', lambda f: (f.session >= 0) & (f.session <= 8)),
    ('session London', lambda f: (f.session >= 7) & (f.session <= 16)),
    ('session NY', lambda f: (f.session >= 12) & (f.session <= 21)),
    ('z50+leg<0.5+Sydney', lambda f: (f.z50.abs() > 3.0) & (f.leg_max < 0.5) & ((f.session >= 21) | (f.session <= 23))),
    ('z50+leg<0.5+highvol', lambda f: (f.z50.abs() > 3.0) & (f.leg_max < 0.5) & (f.atr_ratio > 1.2)),
    ('z50+leg<0.5+breadth0', lambda f: (f.z50.abs() > 3.0) & (f.leg_max < 0.5) & (f.breadth == 0)),
    ('z100+leg<0.7', lambda f: (f.z100.abs() > 2.5) & (f.leg_max < 0.7)),
    ('ret>3pips+z50>2', lambda f: (f.ret_pips.abs() > 3.0) & (f.z50.abs() > 2.0)),
    ('ret>3pips+z50>3', lambda f: (f.ret_pips.abs() > 3.0) & (f.z50.abs() > 3.0)),
]

for name, cond in feature_configs:
    mask = cond(features)
    n = mask.sum()
    if n > 0:
        wr = targets[mask].mean()
        tpd = n / 90  # ~90 trading days
        print(f"  {name:28s}: n={n:>6d}  WR={wr:.1%}  ~{tpd:.1f}/day")

# ═══════════════════════════════════════════════════════════
# TEST 2: Find optimal consensus threshold
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 2: Consensus scoring — combine features into single score")
print("=" * 70)

# Create binary signals
features['sig_z50'] = (features.z50.abs() > 2.5).astype(int)
features['sig_z100'] = (features.z100.abs() > 2.5).astype(int)
features['sig_z20'] = (features.z20.abs() > 3.0).astype(int)
features['sig_legs'] = (features.leg_max < 0.7).astype(int)
features['sig_breadth'] = (features.breadth < 2).astype(int)
features['sig_atr_high'] = (features.atr_ratio > 1.2).astype(int)
features['sig_atr_low'] = (features.atr_ratio < 0.8).astype(int)
features['sig_bb'] = (features.bb_dist.abs() > 2.0).astype(int)
features['sig_accel'] = (features.z_accel.abs() > 0.5).astype(int)
features['sig_ret3'] = (features.ret_pips.abs() > 2.0).astype(int)
features['sig_sydney'] = ((features.session >= 21) | (features.session <= 23)).astype(int)

sig_cols = ['sig_z50', 'sig_z100', 'sig_z20', 'sig_legs', 'sig_breadth',
            'sig_atr_high', 'sig_atr_low', 'sig_bb', 'sig_accel', 'sig_ret3', 'sig_sydney']

# Weighted score (equal weights = 1)
features['score_linear'] = features[sig_cols].sum(axis=1)

# Also compute a "directional score" — only bullish/bearish signals
# Direction = sign of ret (+ = up bar)
features['direction_signal'] = 0
direction_cols = []
for c in ['sig_z50', 'sig_z100', 'sig_z20', 'sig_bb', 'sig_accel']:
    dc = c + '_dir'
    features[dc] = features[c] * features['z50'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    direction_cols.append(dc)
features['dir_score'] = features[direction_cols].sum(axis=1).abs()

print(f"\nScore distribution:")
for score_thresh in range(1, 12):
    mask = features.score_linear >= score_thresh
    n = mask.sum()
    wr = targets[mask].mean() if n > 0 else 0
    tpd = n / 90 if n > 0 else 0
    if n > 0:
        print(f"  Score >= {score_thresh:2d}: n={n:>7,d}  WR={wr:.1%}  ~{tpd:.1f}/day")

# And directional scores
print(f"\nDirectional score distribution:")
for score_thresh in range(1, 12):
    mask = features.dir_score >= score_thresh
    n = mask.sum()
    wr = targets[mask].mean() if n > 0 else 0
    tpd = n / 90 if n > 0 else 0
    if n > 0:
        print(f"  DirScore >= {score_thresh:2d}: n={n:>7,d}  WR={wr:.1%}  ~{tpd:.1f}/day")

# Best combined: linear score + direction
print(f"\nCombined score with direction:")
features['final_score'] = features['score_linear'] * features['dir_score']
for score_thresh in range(1, 30):
    mask = features.final_score >= score_thresh
    n = mask.sum()
    wr = targets[mask].mean() if n > 0 else 0
    tpd = n / 90 if n > 0 else 0
    if n > 0 and wr >= 0.55:
        print(f"  Final >= {score_thresh:2d}: n={n:>7,d}  WR={wr:.1%}  ~{tpd:.1f}/day")

# ═══════════════════════════════════════════════════════════
# TEST 3: Logistic regression (train on first 60 days, test on last 30)
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 3: Logistic regression (train/test split)")
print("=" * 70)

# Split by time
dates = sorted(features.index.unique())
split_idx = int(len(dates) * 0.67)
train_dates = dates[:split_idx]
test_dates = dates[split_idx:]

train_mask = features.index.isin(train_dates)
test_mask = features.index.isin(test_dates)

# Features for ML (all numeric)
ml_features = ['z20', 'z50', 'z100', 'leg_max', 'breadth', 'atr_ratio',
               'bb_dist', 'streak', 'z_accel', 'ret_pips',
               'hour_sin', 'hour_cos']

X_train = features[train_mask][ml_features].replace([np.inf, -np.inf], np.nan).fillna(0).values
y_train = targets[train_mask].values
X_test = features[test_mask][ml_features].replace([np.inf, -np.inf], np.nan).fillna(0).values
y_test = targets[test_mask].values

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

lr = LogisticRegression(C=0.1, max_iter=1000, class_weight='balanced')
lr.fit(X_train_s, y_train)

# Get predictions
probs = lr.predict_proba(X_test_s)[:, 1]

print(f"\nTrain samples: {len(X_train)}, Test samples: {len(X_test)}")
print(f"Train MR rate: {y_train.mean():.1%}, Test MR rate: {y_test.mean():.1%}")

# Test at various probability thresholds
for thr in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
    pred = (probs >= thr).astype(int)
    n = pred.sum()
    wr = (pred == y_test).mean() if n > 0 else 0
    tpd = n / 30  # ~30 trading days in test
    if n > 0:
        print(f"  prob>={thr:.2f}: n={n:>6d}  WR={wr:.1%}  ~{tpd:.1f}/day")

# Feature importance
coefs = pd.Series(lr.coef_[0], index=ml_features).sort_values()
print(f"\nTop features (positive → predicts reversion):")
print(coefs.tail(10).to_string())

# ═══════════════════════════════════════════════════════════
# TEST 4: Full backtest with best ensemble config
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 4: Full backtest with best combined score threshold")
print("=" * 70)

def z_arr(close, window=50):
    ret = close.diff()
    mu = ret.shift(1).rolling(window).mean()
    sigma = ret.shift(1).rolling(window).std().clip(1e-10)
    return (ret - mu) / sigma

def backtest_ensemble(pair, score_thresh=5, z_thresh=3.0, leg_thresh=1.0,
                      stop_a=5.0, trig_a=0.7, gap_a=0.05,
                      max_bars=54, use_backtest_rules=False):
    """Backtest using ensemble score."""
    d = data[pair]
    c = d['close']; h = d['high']; l_ = d['low']
    atr = (h - l_).shift(1).rolling(20).mean()
    z50 = z_arr(c, 50)

    base, quote = LEG_MAP[pair]
    z_base = z_arr(data[base]['close'], 50)
    z_quote = z_arr(data[quote]['close'], 50)
    leg_max = pd.concat([z_base.abs(), z_quote.abs()], axis=1).max(axis=1)

    # Features for scoring
    z20 = z_arr(c, 20); z100 = z_arr(c, 100)
    ret = c.diff(); bb_dist = (c - c.rolling(20).mean()) / c.rolling(20).std().clip(1e-10)
    same_dir = ret.gt(0).astype(int); streak = same_dir.groupby((same_dir != same_dir.shift()).cumsum()).cumcount() + 1
    streak = streak * (ret.gt(0).astype(int) * 2 - 1)
    z_accel = z50.diff()

    cross_z = pd.DataFrame()
    for p in available:
        if p == pair: continue
        cross_z[p] = z_arr(data[p]['close'], 50).abs()
    breadth = (cross_z > 2.0).sum(axis=1)
    atr_median = atr.rolling(50).median(); atr_ratio = atr / atr_median
    hour = np.sin(2 * np.pi * c.index.hour / 24)

    # Score
    score = pd.DataFrame(index=c.index)
    score['s1'] = (z50.abs() > 2.5).astype(int)
    score['s2'] = (z20.abs() > 3.0).astype(int)
    score['s3'] = (z100.abs() > 2.5).astype(int)
    score['s4'] = (leg_max < 0.7).astype(int)
    score['s5'] = (breadth < 2).astype(int)
    score['s6'] = (atr_ratio > 1.2).astype(int)
    score['s7'] = (atr_ratio < 0.8).astype(int)
    score['s8'] = (bb_dist.abs() > 2.0).astype(int)
    score['s9'] = (z_accel.abs() > 0.5).astype(int)
    score['s10'] = (ret.abs() * 10000 > 2.0).astype(int)
    score['s11'] = ((c.index.hour >= 21) | (c.index.hour <= 23)).astype(int)
    score['total'] = score.sum(axis=1)

    # Direction info
    score['direction'] = z50.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    for c in ['s1', 's2', 's3', 's8', 's9']:
        score[c+'_d'] = score[c] * score['direction']
    score['dir_total'] = score[[c+'_d' for c in ['s1', 's2', 's3', 's8', 's9']]].sum(axis=1).abs()

    score['final'] = score['total'] * score['dir_total']
    signal = score['final'] >= score_thresh

    idxs = np.where(signal.values)[0]
    trades, in_trade = [], -1
    for pos in idxs:
        if pos <= in_trade or pos + 2 >= len(d): continue
        direction = -1 if z50.iloc[pos] > 0 else 1
        entry = c.iloc[pos]
        atr_v = atr.iloc[pos]
        s = stop_a * atr_v; tg = trig_a * atr_v; gp = gap_a * atr_v
        best = entry
        for j in range(1, max_bars + 1):
            bp = pos + j
            if bp >= len(d): break
            if direction == 1:
                if h.iloc[bp] > best: best = h.iloc[bp]
                sl = entry - s
                if best - entry > tg: sl = best - gp
                if l_.iloc[bp] <= sl:
                    pnl = (sl - entry) * 10000 - SPREAD_MAP[pair]
                    trades.append(pnl)
                    in_trade = bp; break
            else:
                if l_.iloc[bp] < best: best = l_.iloc[bp]
                sl = entry + s
                if entry - best > tg: sl = best + gp
                if h.iloc[bp] >= sl:
                    pnl = (entry - sl) * 10000 - SPREAD_MAP[pair]
                    trades.append(pnl)
                    in_trade = bp; break
        else:
            eb = min(pos + max_bars, len(d) - 1)
            pnl = (c.iloc[eb] - entry) * direction * 10000 - SPREAD_MAP[pair]
            trades.append(pnl)
            in_trade = eb
    return np.array(trades) if trades else np.array([])


# Test various score thresholds
for thr in [3, 4, 5, 6, 7, 8, 9, 10]:
    all_pnls = []
    for pair in available:
        pnls = backtest_ensemble(pair, score_thresh=thr)
        all_pnls.extend(pnls)
    if not all_pnls: continue
    p = np.array(all_pnls)
    n_days = 90
    tpd = len(p) / n_days
    wr = (p > 0).mean()
    net = p.sum()
    avg = p.mean()
    print(f"  Score>={thr:2d}: n={len(p):>5d}  WR={wr:.1%}  PnL={net:>+7.1f}p  "
          f"avg={avg:>+5.1f}p  ~{tpd:.1f}/day  maxDD={p.cumsum().min():>.0f}p")
