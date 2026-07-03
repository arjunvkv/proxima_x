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
horizons = [60, 300, 900]

print("=" * 70)
print("DPL-X SMOKE TEST — FULL PAIR MATRIX")
print(f"Date: {start} to {end}")
print("=" * 70)

# Step 1: Extract state for both symbols
extractor = StateExtractor(symbols=symbols, start=start, end=end, n_ticks=50000)
state_df = extractor.extract()
print(f"\nState data: {len(state_df)} total rows")

# Step 2: Build forward surfaces
surface = ForwardSurface(horizons=horizons)
fwd_df = surface.build_all(symbols=symbols, start=start, end=end, n_ticks=50000)
print(f"Forward data: {len(fwd_df)} total rows")
print(f"  Forward columns: {[c for c in fwd_df.columns if 'forward' in c]}")

# Step 3: Run propagation for all pairs x horizons
mapper = PropagationMapper()
validator = FrictionValidator()

pairs = [
    ("EURJPY", "EURJPY"),  # in-source control
    ("EURJPY", "USDJPY"),  # cross-asset
    ("USDJPY", "EURJPY"),  # reverse cross-asset
    ("USDJPY", "USDJPY"),  # in-source control 2
]

all_results = []
for src, tgt in pairs:
    src_df = state_df.filter(state_df["symbol"] == src)
    tgt_df = fwd_df.filter(fwd_df["symbol"] == tgt)
    for h in horizons:
        fwd_col = f"forward_{h}"
        if fwd_col not in tgt_df.columns:
            continue
        res = mapper.map(src_df, tgt_df, src, tgt, h)
        fric = validator.evaluate_all(res)
        all_results.append((res, fric))

print("\n" + "=" * 70)
print("RESULTS TABLE")
print("=" * 70)
print(f"{'Source->Target':20s} {'Horizon':8s} {'n':6s} {'WR':8s} {'PF':8s} {'AER':8s} {'TE':8s} {'BestPF':8s}")
print("-" * 70)
for res, fric_list in all_results:
    best_pf = max(r["pf_after_cost"] for r in fric_list) if fric_list else 0
    best_profile = fric_list[[r["pf_after_cost"] for r in fric_list].index(best_pf)]["profile"] if fric_list else ""
    pair_name = f"{res.source}->{res.target}"
    print(f"{pair_name:20s} {res.horizon:<8d} {res.n:<6d} {res.wr:<8.4f} {res.pf:<8.4f} {res.aer:<8.4f} {res.transfer_entropy:<8.4f} {best_pf:<8.4f}")

t = time.perf_counter() - t0
print(f"\nTotal time: {t:.1f}s")
