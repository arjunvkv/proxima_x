"""Bucket × Entropy conditional matrix — test entropy as state conditioner."""
import sys; sys.path.insert(0, '.')
import time
from collections import defaultdict

from research.replay_cache import ReplayCache
from evaluation.delayed_outcome_engine import DelayedOutcomeEngine

H = "=" * 70
TICK_LIMIT = 50000
SEED = 42

FOLDS = [
    dict(name="Fold 1", syms=["EURJPY","USDJPY"], train_start="2026-03-12", train_end="2026-03-28",
         test_start="2026-03-29", test_end="2026-04-14"),
    dict(name="Fold 2", syms=["EURJPY"], train_start="2026-04-01", train_end="2026-04-20",
         test_start="2026-04-21", test_end="2026-05-10"),
    dict(name="Fold 3", syms=["EURJPY"], train_start="2026-05-01", train_end="2026-05-20",
         test_start="2026-05-21", test_end="2026-06-08"),
]


def ecdf_sig(t):
    e = t.get("ecdf", 0.5) if isinstance(t, dict) else 0.5
    d = e - 0.5
    return 1 if d > 0.05 else (-1 if d < -0.05 else 0)

def get_records(ticks):
    doa = DelayedOutcomeEngine(horizon_ticks=20)
    records = []
    ed = {}
    for t in ticks:
        s = t["sym"]
        sig = ecdf_sig(t)
        ed[s] = {"price": t["price"], "ecdf_rank": t["ecdf"], "entropy": t["entropy"], "signal": sig}
        doa.record_snapshot(ed)
        if doa.ready:
            cp = {s2: ed[s2]["price"] for s2 in ed}
            outcomes = doa.evaluate(cp)
            for s2, outcome in outcomes.items():
                records.append({
                    "sym": s2,
                    "ecdf": ed[s2]["ecdf_rank"],
                    "entropy": ed[s2]["entropy"],
                    "outcome": outcome,
                })
    return records


t0 = time.perf_counter()

print(f"{H}", flush=True)
print("  BUCKET × ENTROPY CONDITIONAL MATRIX", flush=True)
print(H, flush=True)

for f in FOLDS:
    name = f["name"]
    print(f"\n  --- {name} (TEST window) ---", flush=True)

    test_ticks = ReplayCache(f["syms"], f["test_start"], f["test_end"], tick_limit=TICK_LIMIT, seed=SEED).compute()
    records = get_records(test_ticks)

    # Build 11 × 10 matrix: ECDF bucket × Entropy decile
    matrix = defaultdict(lambda: {"n": 0, "up": 0, "sum": 0.0})

    for r in records:
        ecdf_b = int(r["ecdf"] * 10)
        ent_b = int(r["entropy"] * 10)
        if ent_b >= 10: ent_b = 9
        key = (ecdf_b, ent_b)
        o = r["outcome"]
        matrix[key]["n"] += 1
        matrix[key]["sum"] += o
        if o > 0:
            matrix[key]["up"] += 1

    # Print matrix
    print(f"\n  Hit rate (Up%) by ECDF × Entropy decile:")
    header = "".join(f" E{0.5+e*0.5:.1f}" for e in range(10))
    print(f"  {'':>8}{header}", flush=True)
    for eb in range(11):
        row = f"  EC={eb/10:.1f}"
        for en in range(10):
            key = (eb, en)
            if key in matrix:
                m = matrix[key]
                wp = m["up"] / m["n"] * 100 if m["n"] else 0
                row += f" {wp:>5.1f}"
            else:
                row += "     -"
        print(row, flush=True)

    # Check if entropy modulates any bucket
    print(f"\n  Entropy modulation check (edge difference high vs low entropy per bucket):", flush=True)
    mod_count = 0
    for eb in range(11):
        high_ent = matrix.get((eb, 8), None) or matrix.get((eb, 7), None)
        low_ent = matrix.get((eb, 1), None) or matrix.get((eb, 2), None)
        mid_ent = matrix.get((eb, 4), None) or matrix.get((eb, 5), None)
        if high_ent and low_ent and high_ent["n"] > 100 and low_ent["n"] > 100:
            high_wp = high_ent["up"] / high_ent["n"] * 100
            low_wp = low_ent["up"] / low_ent["n"] * 100
            diff = abs(high_wp - low_wp)
            if diff > 10:
                mod_count += 1
                print(f"    Bucket {eb/10:.1f}: LowEnt={low_wp:.1f}%({low_ent['n']}) HighEnt={high_wp:.1f}%({high_ent['n']}) diff={diff:.1f}pp MODULATED", flush=True)
    if mod_count == 0:
        print(f"    No bucket shows >10pp entropy modulation", flush=True)

print(f"\n{H}", flush=True)
print("  ENTROPY CONDITIONING SUMMARY", flush=True)
print(H, flush=True)

# Across all folds — does entropy-predictive bucket 0.4-0.6 change with entropy?
combined_eb = [4, 5, 6]
combined = {"low": {"n": 0, "up": 0}, "mid": {"n": 0, "up": 0}, "high": {"n": 0, "up": 0}}

all_records = []
for f in FOLDS:
    test_ticks = ReplayCache(f["syms"], f["test_start"], f["test_end"], tick_limit=TICK_LIMIT, seed=SEED).compute()
    all_records.extend(get_records(test_ticks))

for r in all_records:
    eb = int(r["ecdf"] * 10)
    en = int(r["entropy"] * 10)
    if eb in [4, 5, 6]:
        if en <= 3:
            combined["low"]["n"] += 1
            if r["outcome"] > 0: combined["low"]["up"] += 1
        elif en <= 6:
            combined["mid"]["n"] += 1
            if r["outcome"] > 0: combined["mid"]["up"] += 1
        else:
            combined["high"]["n"] += 1
            if r["outcome"] > 0: combined["high"]["up"] += 1

print(f"  Buckets 0.4-0.6 (highest edge) conditioned on entropy (all folds):")
for label in ["low", "mid", "high"]:
    c = combined[label]
    wp = c["up"] / c["n"] * 100 if c["n"] else 0
    print(f"    {label:>5} entropy: {c['n']:>6} records, {wp:>5.1f}% UP")

print(f"\nTotal: {time.perf_counter()-t0:.1f}s", flush=True)
