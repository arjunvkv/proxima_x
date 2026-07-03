"""Multi-fold OSS walk-forward validation — folds 1-3 with aggregate statistics.

Fold 1: Train Mar 12-28 (EURJPY+USDJPY), Test Mar 29 - Apr 14 (EURJPY+USDJPY)
Fold 2: Train Apr 1-20 (EURJPY), Test Apr 21 - May 10 (EURJPY)
Fold 3: Train May 1-20 (EURJPY), Test May 21 - Jun 8 (EURJPY)
"""
import sys; sys.path.insert(0, '.')
import time
from collections import defaultdict

from research.replay_cache import ReplayCache
from research.outcome_surface_signal import OutcomeSurfaceSignal
from evaluation.delayed_outcome_engine import DelayedOutcomeEngine
from validation.wfv_engine import WalkForwardValidator, StatisticalEdgeTest

HEADER = "=" * 70

FOLDS = [
    dict(name="Fold 1", syms=["EURJPY","USDJPY"], train_start="2026-03-12", train_end="2026-03-28",
         test_start="2026-03-29", test_end="2026-04-14"),
    dict(name="Fold 2", syms=["EURJPY"], train_start="2026-04-01", train_end="2026-04-20",
         test_start="2026-04-21", test_end="2026-05-10"),
    dict(name="Fold 3", syms=["EURJPY"], train_start="2026-05-01", train_end="2026-05-20",
         test_start="2026-05-21", test_end="2026-06-08"),
]
TICK_LIMIT = 50000
SEED = 42


def get_records(ticks, signal_fn, doa_horizon=20):
    doa = DelayedOutcomeEngine(horizon_ticks=doa_horizon)
    records = []
    ed = {}
    for t in ticks:
        s = t["sym"]
        ed[s] = {"price": t["price"], "ecdf_rank": t["ecdf"], "entropy": t["entropy"], "signal": signal_fn(t)}
        doa.record_snapshot(ed)
        if doa.ready:
            cp = {s2: ed[s2]["price"] for s2 in ed}
            outcomes = doa.evaluate(cp)
            for s2, outcome in outcomes.items():
                records.append({
                    "sym": s2,
                    "signal": ed[s2]["signal"],
                    "outcome": outcome,
                    "ecdf": ed[s2]["ecdf_rank"],
                    "entropy": ed[s2]["entropy"],
                })
    return records


def ecdf_signal(r):
    e = r.get("ecdf", 0.5)
    d = e - 0.5
    return 1 if d > 0.05 else (-1 if d < -0.05 else 0)


def always_sell(r):
    return -1


t0 = time.perf_counter()

# ───── Load all caches ─────
print("Loading caches...", flush=True)
cache_store = {}
for f in FOLDS:
    for phase in ["train", "test"]:
        key = f"{f['name']}_{phase}"
        start = f[f"{phase}_start"]
        end = f[f"{phase}_end"]
        syms = f["syms"]
        c = ReplayCache(syms, start, end, tick_limit=TICK_LIMIT, seed=SEED).compute()
        cache_store[key] = c
        print(f"  {key}: {len(c)} ticks", flush=True)

# ───── Run multi-fold analysis ─────
print(f"\n{HEADER}", flush=True)
print("  MULTI-FOLD WALK-FORWARD VALIDATION", flush=True)
print(HEADER, flush=True)

fold_results = []
bucket_stability = defaultdict(list)

for f in FOLDS:
    name = f["name"]
    train_key = f"{name}_train"
    test_key = f"{name}_test"

    train_ticks = cache_store[train_key]
    test_ticks = cache_store[test_key]

    # Build OSS from TRAIN
    print(f"\n  --- {name} ---", flush=True)
    train_recs = get_records(train_ticks, ecdf_signal)
    oss = OutcomeSurfaceSignal.from_pipeline_records(train_recs, ev_threshold=0.05)

    # Test OSS on TRAIN
    train_eval = oss.evaluate(train_recs)
    train_always = WalkForwardValidator(train_size=5, test_size=3).run(
        [{"signal": always_sell(r), "outcome": r["outcome"]} for r in train_recs])
    print(f"  TRAIN: OSS acc={train_eval['accuracy']:.4f} Always={train_always.get('accuracy',0):.4f}", flush=True)

    # Test OSS on TEST (OOS)
    test_recs = get_records(test_ticks, ecdf_signal)
    test_eval = oss.evaluate(test_recs)
    test_always = WalkForwardValidator(train_size=5, test_size=3).run(
        [{"signal": always_sell(r), "outcome": r["outcome"]} for r in test_recs])
    print(f"  TEST:  OSS acc={test_eval['accuracy']:.4f} pnl={test_eval['pnl']:+8.1f} sig={test_eval['n_signals']}", flush=True)
    print(f"         Always-SELL acc={test_always.get('accuracy',0):.4f}", flush=True)

    uplift = test_eval["accuracy"] - test_always.get("accuracy", 0)
    print(f"         Uplift vs Always-SELL: {uplift:+.4f}", flush=True)

    fold_results.append({
        "fold": name,
        "train_oss_acc": train_eval["accuracy"],
        "test_oss_acc": test_eval["accuracy"],
        "test_oss_pnl": test_eval["pnl"],
        "test_always_acc": test_always.get("accuracy", 0),
        "test_always_pnl": test_always.get("pnl_proxy", 0),
        "uplift": uplift,
        "n_signals": test_eval["n_signals"],
        "n_records": test_eval["n_records"],
        "n_buckets": oss.bucket_count(),
        "signal_density": oss.signal_density(),
    })

    # Bucket sign stability
    test_buckets = defaultdict(lambda: {"n": 0, "sum": 0.0})
    for r in test_recs:
        ek = str(int(r["ecdf"] * 10) / 10)
        test_buckets[ek]["n"] += 1
        test_buckets[ek]["sum"] += r["outcome"]

    for k in sorted(oss._buckets.keys(), key=float):
        train_ev = oss._buckets[k]["ev"]
        test_ev = test_buckets.get(k, {}).get("sum", 0) / max(1, test_buckets.get(k, {}).get("n", 1))
        sign_ok = (train_ev * test_ev) >= 0 or abs(train_ev) < 0.01
        bucket_stability[k].append({"train_ev": train_ev, "test_ev": test_ev, "sign_ok": sign_ok})

    # Bias test on TEST
    for label, cond in [("UP", lambda r: r["outcome"] > 0), ("DOWN", lambda r: r["outcome"] < 0)]:
        sub = [r for r in test_recs if cond(r)]
        if sub:
            sub_eval = oss.evaluate(sub)
            sub_always = WalkForwardValidator(train_size=5, test_size=3).run(
                [{"signal": always_sell(r), "outcome": r["outcome"]} for r in sub])
            print(f"         {label} days: OSS={sub_eval['accuracy']:.4f} Always={sub_always.get('accuracy',0):.4f}", flush=True)


# ───── Aggregate ─────
print(f"\n{HEADER}", flush=True)
print("  AGGREGATE RESULTS (3 folds)", flush=True)
print(HEADER, flush=True)

accs = [r["test_oss_acc"] for r in fold_results]
uplifts = [r["uplift"] for r in fold_results]
pnls = [r["test_oss_pnl"] for r in fold_results]
always_accs = [r["test_always_acc"] for r in fold_results]

import math
mean_acc = sum(accs) / len(accs)
std_acc = math.sqrt(sum((a - mean_acc)**2 for a in accs) / len(accs))
mean_uplift = sum(uplifts) / len(uplifts)
mean_always = sum(always_accs) / len(always_accs)

print(f"  Mean OOS accuracy:        {mean_acc:.4f} (std={std_acc:.4f})", flush=True)
print(f"  Mean Always-SELL accuracy:{mean_always:.4f}", flush=True)
print(f"  Mean uplift:             {mean_uplift:+.4f}", flush=True)
print(f"  Total PnL (3 folds):     {sum(pnls):+.0f}", flush=True)

# Per-fold detail
print(f"\n  {'Fold':>10} {'OSS_acc':>8} {'Always':>8} {'Uplift':>8} {'PnL':>10} {'Sig%':>6}", flush=True)
for r in fold_results:
    sig_pct = r["n_signals"] / r["n_records"] * 100 if r["n_records"] else 0
    print(f"  {r['fold']:>10} {r['test_oss_acc']:.4f}   {r['test_always_acc']:.4f}   {r['uplift']:+.4f}   {r['test_oss_pnl']:+8.1f}  {sig_pct:>5.1f}%", flush=True)

# Bucket sign persistence across folds
print(f"\n  Bucket Sign Persistence (across all folds):", flush=True)
print(f"  {'Bucket':>7} {'Folds':>6} {'Stable':>7} {'Pct':>6}", flush=True)
stable_total = 0
fold_total = 0
for k in sorted(bucket_stability.keys(), key=float):
    entries = bucket_stability[k]
    stable = sum(1 for e in entries if e["sign_ok"])
    total = len(entries)
    pct = stable / total * 100
    stable_total += stable
    fold_total += total
    print(f"  {k:>7} {total:>6} {stable:>7} {pct:>5.0f}%", flush=True)
print(f"  Overall: {stable_total}/{fold_total} = {stable_total/fold_total*100:.0f}% sign-stable", flush=True)

# Check acceptance criteria
print(f"\n  ACCEPTANCE CRITERIA:", flush=True)
criteria = [
    ("Mean OOS accuracy > 0.53", mean_acc > 0.53, f"{mean_acc:.4f} > 0.53"),
    ("Beats Always-SELL by >2%", mean_uplift > 0.02, f"{mean_uplift:.4f} > 0.02"),
    ("Bucket sign stability >80%", stable_total/fold_total > 0.8, f"{stable_total/fold_total*100:.0f}% > 80%"),
]
for label, ok, detail in criteria:
    print(f"  {'PASS' if ok else 'FAIL'}: {label} ({detail})", flush=True)

print(f"\nTotal: {time.perf_counter()-t0:.1f}s", flush=True)
