import sys, time
sys.path.insert(0, ".")
sys.path.insert(0, "proxima_x")

t0 = time.perf_counter()

from research.dpl_x.state_extractor import StateExtractor
from research.dpl_x.forward_surface import ForwardSurface
from research.dpl_x.propagation_mapper import PropagationMapper
from research.dpl_x.friction_validator import FrictionValidator

symbols = ["EURJPY", "USDJPY"]
start = "2026-04-01"
end = "2026-04-14"

print(f"[DPL-X SMOKE] Source: EURJPY -> Target: USDJPY")
print(f"[DPL-X SMOKE] Date range: {start} to {end}")
print(f"[DPL-X SMOKE] Horizon: 60s")

# Step 1: Extract source state
print(f"[DPL-X SMOKE] Step 1: Extracting EURJPY state...")
t1 = time.perf_counter()
extractor = StateExtractor(symbols=["EURJPY"], start=start, end=end, n_ticks=50000)
source_df = extractor.extract()
print(f"[DPL-X SMOKE]   EURJPY state: {len(source_df)} rows in {time.perf_counter()-t1:.1f}s")
print(f"[DPL-X SMOKE]   Buckets: {sorted(source_df['oss_bucket'].unique().to_list())}")
print(f"[DPL-X SMOKE]   Tross cross counts: {source_df['tross_cross'].value_counts()}")

# Step 2: Build target forward surface
print(f"[DPL-X SMOKE] Step 2: Building USDJPY forward surface (60s)...")
t1 = time.perf_counter()
surface = ForwardSurface(horizons=[60])
target_df = surface.build("USDJPY", start=start, end=end, n_ticks=50000)
print(f"[DPL-X SMOKE]   USDJPY forward: {len(target_df)} rows in {time.perf_counter()-t1:.1f}s")
if len(target_df) > 0:
    fwd_col = [c for c in target_df.columns if "forward" in c][0]
    print(f"[DPL-X SMOKE]   Forward return stats: mean={target_df[fwd_col].mean():.4f}, std={target_df[fwd_col].std():.4f}")

# Step 3: Map propagation
print(f"[DPL-X SMOKE] Step 3: Mapping propagation...")
t1 = time.perf_counter()
mapper = PropagationMapper()
result = mapper.map(source_df, target_df, "EURJPY", "USDJPY", 60)
elapsed = time.perf_counter() - t1
print(f"[DPL-X SMOKE]   n={result.n}, wr={result.wr:.4f}, pf={result.pf:.4f}, aer={result.aer:.4f}")
print(f"[DPL-X SMOKE]   expected_return={result.expected_return:.4f}, transfer_entropy={result.transfer_entropy:.4f}")
print(f"[DPL-X SMOKE]   lag_score={result.lag_score}")
print(f"[DPL-X SMOKE]   Done in {elapsed:.1f}s")

# Step 4: Friction validation
print(f"[DPL-X SMOKE] Step 4: Friction validation (CME)...")
t1 = time.perf_counter()
validator = FrictionValidator()
fric_result = validator.evaluate(result, "cme_fx")
print(f"[DPL-X SMOKE]   pf_after_cost={fric_result['pf_after_cost']:.4f}")
print(f"[DPL-X SMOKE]   expectancy_after_cost={fric_result['expectancy_after_cost']:.4f}")
print(f"[DPL-X SMOKE]   slippage_ratio={fric_result['slippage_ratio']:.4f}")
print(f"[DPL-X SMOKE]   net_edge={fric_result['net_edge']:.4f}")
print(f"[DPL-X SMOKE]   Done in {time.perf_counter()-t1:.1f}s")

# Step 5: All profiles
print(f"[DPL-X SMOKE] Step 5: All profiles...")
t1 = time.perf_counter()
all_profiles = validator.evaluate_all(result)
for r in all_profiles:
    print(f"  {r['profile']:15s}: pf={r['pf_after_cost']:.4f}, edge={r['net_edge']:.4f}")
print(f"[DPL-X SMOKE]   Done in {time.perf_counter()-t1:.1f}s")

total = time.perf_counter() - t0
print(f"\n[DPL-X SMOKE] TOTAL: {total:.1f}s")
print(f"[DPL-X SMOKE] VERDICT: {'PROMOTION CANDIDATE' if result.pf > 1.4 and result.wr > 0.58 else 'NO EDGE'}")
