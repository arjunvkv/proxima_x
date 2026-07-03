from __future__ import annotations

import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:\Trading\Agentic_Trading\proxima_x")

import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=120)

from core import StateVector, StateEngine, EventEngine, Event, EventType, normalize_state, state_distance, state_similarity
from research import *
from features import *
from ml import StateClassifier, StateClusterer, StateSimilaritySearch
from backtesting import StateBacktest, TransitionAnalyzer
from utils import Profiler

import polars as pl
from sklearn.datasets import make_classification

np.random.seed(42)
n = 1000
price = np.cumsum(np.random.randn(n) * 0.01) + 100
high = price + np.abs(np.random.randn(n) * 0.005)
low = price - np.abs(np.random.randn(n) * 0.005)
open_ = price - np.random.randn(n) * 0.002
volume = np.random.exponential(1000, n).astype(np.float64)
returns = np.diff(price, prepend=price[0])

print("=" * 100)
print("PROXIMA X -- HIDDEN-STATE MARKET RESEARCH ENGINE")
print("=" * 100)

# CORE
print("\n-- CORE STATE ENGINE --")
sv = StateVector(0.5, 0.3, 0.8, 0.2, 0.6, 0.4, 0.1)
print(f"StateVector: {sv}")
print(f"as_array:     {sv.as_array}")

v = np.array([0.5, 0.3, 0.8, 0.2, 0.6, 0.4, 0.1], dtype=np.float32)
nv = normalize_state(v)
print(f"normalized:   {nv}")
print(f"distance:     {state_distance(v, nv):.4f}")
print(f"similarity:   {state_similarity(v, nv):.4f}")

engine = StateEngine()
for i in range(0, n, 10):
    sv = StateVector(np.sin(i * 0.05), np.cos(i * 0.03), np.tanh(i * 0.01),
                     np.exp(-i * 0.02), np.sin(i * 0.07), np.cos(i * 0.04), np.sin(i * 0.02))
    engine.update(sv, i)
print(f"state_stability:     {engine.state_stability:.4f}")
print(f"transition_rate:     {engine.transition_rate:.4f}")
print(f"transition_count:    {len(engine.recent_transitions)}")

ee = EventEngine()
print(f"EventEngine created: {type(ee).__name__}")

# 1. MARKET MEMORY FIELD
print("\n-- 1. MARKET MEMORY FIELD --")
mf = MemoryFieldResearch()
mf_res = mf.compute_all(price, returns)
for k, v in mf_res.items():
    print(f"  {k:25s}  shape={v.shape}  [{v[100]:.4f}, {v[200]:.4f}, {v[300]:.4f}]  nan={np.isnan(v).sum()}")

# 2. TEMPORAL DNA
print("\n-- 2. TEMPORAL DNA --")
td = TemporalDNAResearch()
ohlc = {"high": high, "low": low, "close": price, "volume": volume, "returns": returns}
td_res = td.compute_all(ohlc)
for k, v in td_res.items():
    if v.ndim == 1:
        print(f"  {k:25s}  shape={v.shape}  [{v[100]:.4f}, {v[200]:.4f}, {v[300]:.4f}]  nan={np.isnan(v).sum()}")
    else:
        print(f"  {k:25s}  shape={v.shape}  (multi-dim vector)")

# 3. INFORMATION PRESSURE
print("\n-- 3. INFORMATION PRESSURE --")
ip = InformationPressureResearch()
ip_res = ip.compute_all(returns, volume)
for k, v in ip_res.items():
    print(f"  {k:25s}  shape={v.shape}  [{v[100]:.4f}, {v[200]:.4f}, {v[300]:.4f}]  nan={np.isnan(v).sum()}")

# 4. LIQUIDITY MIGRATION
print("\n-- 4. LIQUIDITY MIGRATION --")
lm = LiquidityMigrationResearch()
lm_res = lm.compute_all(price)
for k, v in lm_res.items():
    print(f"  {k:25s}  shape={v.shape}  [{v[100]:.4f}, {v[200]:.4f}, {v[300]:.4f}]  nan={np.isnan(v).sum()}")

# 5. COHORT SIMULATION
print("\n-- 5. COHORT SIMULATION --")
cs = CohortSimulationResearch()
cs_res = cs.compute_all(volume, price, returns)
for k, v in cs_res.items():
    if v.ndim == 1:
        print(f"  {k:25s}  shape={v.shape}  [{v[100]:.4f}, {v[200]:.4f}, {v[300]:.4f}]  nan={np.isnan(v).sum()}")
    else:
        print(f"  {k:25s}  shape={v.shape}  (multi-dim)  nan={np.isnan(v).sum()}")

# 6. STATE ENTANGLEMENT
print("\n-- 6. STATE ENTANGLEMENT --")
se = StateEntanglementResearch()
tf_returns = {tf: returns for tf in se.TIMEFRAMES}
se_res = se.compute_all(tf_returns)
count = 0
for k, v in se_res.items():
    print(f"  {k:35s}  shape={v.shape}  [{v[100]:.4f}, {v[200]:.4f}, {v[300]:.4f}]")
    count += 1
    if count >= 6:
        print(f"  ... ({len(se_res) - count} more entanglement pairs)")
        break

# 7. MARKET TENSION TENSOR
print("\n-- 7. MARKET TENSION TENSOR --")
tt = TensionTensorResearch()
inputs = {
    "memory": mf_res["memory_strength"],
    "pressure": ip_res["pressure_build"],
    "liquidity": lm_res["liquidity_mass"],
    "cohort": cs_res["cohort_conflict"],
    "volatility": np.abs(returns).astype(np.float32),
    "state_alignment": np.ones(n, dtype=np.float32),
}
tt_res = tt.compute_all(inputs)
for k, v in tt_res.items():
    print(f"  {k:25s}  shape={v.shape}  [{v[100]:.4f}, {v[200]:.4f}, {v[300]:.4f}]  nan={np.isnan(v).sum()}")

# 8. BEHAVIORAL ECHOES
print("\n-- 8. BEHAVIORAL ECHOES --")
be = BehavioralEchoesResearch()
for i in range(20):
    ev = np.random.randn(7).astype(np.float32)
    rv = np.random.randn(7).astype(np.float32)
    be.store_event_response(ev, rv)
chain = [(be._echo_events[i], be._echo_responses[i]) for i in range(len(be._echo_events))]
echo_strength = be.compute_echo_strength(chain)
echo_decay = be.compute_echo_decay(20)
sim = be.compute_echo_similarity(be._echo_events[0], be._echo_events[-1])
print(f"  echo_chain_length:    {len(be._echo_events)}")
print(f"  echo_strength:        {echo_strength:.4f}")
print(f"  echo_decay(20):       {echo_decay:.4f}")
print(f"  echo_similarity:      {sim:.4f}")
pattern = be.build_echo_pattern([be._echo_events])
print(f"  patterns_found:       {pattern}")

# FEATURES
print("\n-- FEATURE COMPUTATION (Numba JIT) --")
print(f"  rolling_zscore(20):     {rolling_zscore(price, 20)[100]:.4f}  {rolling_zscore(price, 20)[200]:.4f}  {rolling_zscore(price, 20)[300]:.4f}")
print(f"  rolling_skew(30):       {rolling_skew(price, 30)[100]:.4f}  {rolling_skew(price, 30)[200]:.4f}  {rolling_skew(price, 30)[300]:.4f}")
print(f"  rolling_kurtosis(30):   {rolling_kurtosis(price, 30)[100]:.4f}  {rolling_kurtosis(price, 30)[200]:.4f}  {rolling_kurtosis(price, 30)[300]:.4f}")
print(f"  rolling_entropy(50):    {rolling_entropy(returns, 50)[100]:.4f}  {rolling_entropy(returns, 50)[200]:.4f}  {rolling_entropy(returns, 50)[300]:.4f}")
print(f"  rolling_hurst(100):     {rolling_hurst(returns, 100)[200]:.4f}  {rolling_hurst(returns, 100)[400]:.4f}  {rolling_hurst(returns, 100)[600]:.4f}")
print(f"  efficiency_ratio(20):   {efficiency_ratio(price, 20)[100]:.4f}  {efficiency_ratio(price, 20)[200]:.4f}  {efficiency_ratio(price, 20)[300]:.4f}")
print(f"  ATR(14):                {atr(high, low, price, 14)[100]:.4f}  {atr(high, low, price, 14)[200]:.4f}  {atr(high, low, price, 14)[300]:.4f}")
print(f"  super_smoother(10):     {super_smoother(price, 10)[100]:.4f}  {super_smoother(price, 10)[200]:.4f}  {super_smoother(price, 10)[300]:.4f}")
print(f"  price_position:         {price_position(high, low, price)[100]:.4f}  {price_position(high, low, price)[200]:.4f}  {price_position(high, low, price)[300]:.4f}")

# Polars
df = pl.DataFrame({"open": open_, "high": high, "low": low, "close": price, "volume": volume})
fg = FeatureGenerator()
result_lf = fg.generate_all(df.lazy())
result_df = result_lf.collect()
exclude = {"open", "high", "low", "close", "volume"}
feature_cols = [c for c in result_df.columns if c not in exclude]
print(f"  polars_features:       {result_df.shape[0]} rows x {result_df.shape[1]} cols")
print(f"  feature columns:       {feature_cols}")

# BACKTESTING
print("\n-- BACKTESTING --")
sb = StateBacktest()
fake_states = np.random.randint(0, 5, n).astype(np.int32)
state_df = pl.DataFrame({"state_cluster": fake_states, "close": price, "forward_return": np.random.randn(n) * 0.02})
result = sb.run(state_df, state_col="state_cluster", forward_window=20)
print("  state_backtest:")
print(result)

ta = TransitionAnalyzer()
tmat = sb.compute_state_transition_matrix(fake_states, 5)
print(f"  transition_matrix ({tmat.shape[0]}x{tmat.shape[1]}):")
for i in range(5):
    print(f"    state {i}: {tmat[i]}")
print(f"  state_persistence:     {sb.compute_state_persistence(fake_states):.4f}")
print(f"  state_entropy:         {sb.compute_state_entropy(fake_states):.4f}")
print(f"  absorbing_states:      {ta.compute_absorbing_states(tmat)}")
print(f"  transient_states:      {ta.compute_transient_states(tmat)}")
expected = ta.compute_expected_duration(tmat)
print(f"  expected_durations:    {expected}")
sig = ta.find_significant_transitions(tmat)
print(f"  significant_trans:     {sig[:5]}...")

# ML
print("\n-- ML PIPELINE --")
X, y = make_classification(n_samples=500, n_features=7, n_classes=5, n_informative=5, random_state=42)
sc = StateClassifier()
try:
    sc.train(X.astype(np.float32), y.astype(np.int32))
    preds = sc.predict(X.astype(np.float32))
    acc = (preds == y).mean()
    print(f"  classifier_acc:        {acc:.4f}")
    fi = sc.feature_importance()
    print(f"  feature_importance:    {fi}")
except Exception as e:
    print(f"  classifier:            {e}")

sc2 = StateClusterer(method="minibatch_kmeans", params={"n_clusters": 5})
labels = sc2.fit_predict(X.astype(np.float32))
print(f"  kmeans_labels:         {np.unique(labels)}")
stats = sc2.cluster_stats(labels)
print(f"  cluster_sizes:         {stats}")

sss = StateSimilaritySearch(dim=7, index_type="flat")
sss.build_index(X.astype(np.float32))
dists, idxs = sss.search(X[0].astype(np.float32), top_k=5)
print(f"  faiss_top5_dist:       {dists[0]}")
print(f"  faiss_top5_idx:        {idxs[0]}")
print(f"  index_size:            {sss.index_size()}")

# UTILITIES
print("\n-- UTILITIES --")
prof = Profiler()
prof.start("compute_test")
_ = np.sin(np.random.randn(1000000))
result_prof = prof.stop("compute_test")
print(f"  profiler:              {result_prof}")
print("=" * 100)
print("PROXIMA X -- SYSTEM COMPLETE")
print("=" * 100)
