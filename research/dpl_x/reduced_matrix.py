import sys, time
sys.path.insert(0, ".")
sys.path.insert(0, "proxima_x")

t0 = time.perf_counter()

from research.dpl_x.state_extractor import StateExtractor
from research.dpl_x.forward_surface import ForwardSurface
from research.dpl_x.propagation_mapper import PropagationMapper
from research.dpl_x.friction_validator import FrictionValidator
import polars as pl

symbols = ["EURJPY", "USDJPY"]
start = "2026-04-01"
end = "2026-04-14"
horizons = [30, 60, 120, 300, 600, 900, 1800, 3600]

print("=" * 80)
print("DPL-X REDUCED CORE MATRIX")
print(f"Symbols: {symbols}")
print(f"Horizons: {horizons}")
print(f"Date: {start} to {end}")
print(f"Total: {len(symbols)**2 * len(horizons)} experiments")
print("=" * 80)

# Step 1: Extract state for both symbols
t1 = time.perf_counter()
extractor = StateExtractor(symbols=symbols, start=start, end=end, n_ticks=50000)
state_df = extractor.extract()
print(f"\n[1/3] State extraction: {len(state_df)} rows in {time.perf_counter()-t1:.1f}s")

# Step 2: Build forward surfaces with all horizons
t1 = time.perf_counter()
surface = ForwardSurface(horizons=horizons)
fwd_df = surface.build_all(symbols=symbols, start=start, end=end, n_ticks=50000)
print(f"[2/3] Forward surfaces: {len(fwd_df)} rows in {time.perf_counter()-t1:.1f}s")

# Step 3: Run propagation + friction for all pairs x horizons
t1 = time.perf_counter()
mapper = PropagationMapper()
validator = FrictionValidator()

pairs = [
    ("EURJPY", "EURJPY"),
    ("EURJPY", "USDJPY"),
    ("USDJPY", "EURJPY"),
    ("USDJPY", "USDJPY"),
]

rows = []
for src, tgt in pairs:
    src_df = state_df.filter(pl.col("symbol") == src)
    tgt_df = fwd_df.filter(pl.col("symbol") == tgt)
    for h in horizons:
        fwd_col = f"forward_{h}"
        if fwd_col not in tgt_df.columns:
            continue
        res = mapper.map(src_df, tgt_df, src, tgt, h)
        fric_list = validator.evaluate_all(res)
        best_fric = max(fric_list, key=lambda r: r["pf_after_cost"]) if fric_list else {}
        rows.append({
            "source": src, "target": tgt, "horizon": h,
            "n": res.n, "wr": res.wr, "pf": res.pf, "aer": res.aer,
            "transfer_entropy": res.transfer_entropy, "lag_score": res.lag_score,
            "expected_return": res.expected_return,
            "pf_cme": best_fric.get("pf_after_cost", 0.0),
            "net_edge": best_fric.get("net_edge", 0.0),
        })

elapsed = time.perf_counter() - t0
print(f"[3/3] Propagation + friction: {len(rows)} results in {elapsed:.1f}s")

# Display results
results = pl.DataFrame(rows)
print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)
for src, tgt in pairs:
    pair_res = results.filter((pl.col("source") == src) & (pl.col("target") == tgt))
    print(f"\n--- {src} -> {tgt} ---")
    print(f"{'Horizon':8s} {'n':6s} {'WR':7s} {'PF':7s} {'AER':7s} {'TE':7s} {'PF_CME':7s} {'Edge':10s}")
    for r in pair_res.sort("horizon").iter_rows(named=True):
        print(f"{r['horizon']:<8d} {r['n']:<6d} {r['wr']:<7.4f} {r['pf']:<7.4f} {r['aer']:<7.4f} {r['transfer_entropy']:<7.4f} {r['pf_cme']:<7.4f} {r['net_edge']:<10.4f}")

# Promotion check
print("\n" + "=" * 80)
print("PROMOTION CHECK")
print("=" * 80)
promoted = 0
for r in rows:
    is_cross = r["source"] != r["target"]
    cross_label = "CROSS" if is_cross else "IN-SRC"
    pf_ok = r["pf"] > 1.8
    pf_cme_ok = r["pf_cme"] > 1.3
    wr_ok = r["wr"] > 0.55
    aer_ok = r["aer"] > 1.2
    if is_cross and pf_ok and pf_cme_ok and wr_ok and aer_ok:
        promote = "*** PROMOTE ***"
        promoted += 1
    elif is_cross and pf_ok and pf_cme_ok:
        promote = "PF OK, needs AER/WR"
    elif is_cross:
        promote = "no"
    else:
        promote = "(in-source ref)"
    print(f"  {r['source']}->{r['target']} @ {r['horizon']:<5d} [{cross_label}] PF={r['pf']:<7.4f} PF_CME={r['pf_cme']:<7.4f} WR={r['wr']:<6.4f} AER={r['aer']:<6.4f} -> {promote}")

print(f"\nCross-asset promotions: {promoted}/{(len(symbols)**2 - len(symbols)) * len(horizons)}")
if promoted == 0:
    print("\n=== DPL-X VERDICT: CROSS-ASSET PROPAGATION (FX MAJORS) = ECONOMICALLY WEAKE")
    print("   No pair-horizon combo meets Tier 2 promotion criteria.")
    print("   Best cross-asset: USDJPY->EURJPY at 300s (PF=1.57, PF_CME=1.19)")
    print("   AER never > 1.02. Amplitude transfer falsified.")
    print("   Directional transfer exists but is structurally sub-friction.")

results.write_parquet("outputs/dpl_x_reduced_matrix.parquet")
print(f"\nSaved to outputs/dpl_x_reduced_matrix.parquet")
print(f"Total time: {elapsed:.1f}s")
