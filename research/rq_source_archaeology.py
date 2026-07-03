"""Signal Source Archaeology v3 — ECDF-only pipeline with real DOA outcomes."""
import sys; sys.path.insert(0, '.')
import time
from collections import defaultdict, Counter

from replay.environment import build_replay_environment, ReplayConfig
from replay.clock_patcher import patch_clock
from research.layer_config import LayerConfig
from research.light_ablation_runner import LightAblationRunner
from validation.wfv_engine import WalkForwardValidator, StatisticalEdgeTest

CFG = ReplayConfig(symbols=["EURJPY", "USDJPY"], start="2026-03-12", end="2026-03-28",
    speed=500000, burst=True, latency=False, slippage=False, seed=42)
TICK_LIMIT = 50000
HEADER = "=" * 70


class CaptureRunner(LightAblationRunner):
    def run_and_capture(self, env):
        self._fusion.entropy_flip_threshold = 1.5
        res = self.run(env)
        return res, self._wfv_records


def run_once(signal_mode="ecdf"):
    env = build_replay_environment(CFG)
    patch_clock(env.clock)
    runner = CaptureRunner(LayerConfig(), tick_limit=TICK_LIMIT, signal_mode=signal_mode)
    res, records = runner.run_and_capture(env)
    return res, records


def wfv(records, signal_fn):
    recs = [{"signal": signal_fn(r), "outcome": r.get("outcome", 0)} for r in records]
    w = WalkForwardValidator(train_size=5, test_size=3).run(recs)
    e = StatisticalEdgeTest.run(w)
    return {"accuracy": e["accuracy"], "pnl_proxy": e["pnl_proxy"], "edge_detected": e["edge_detected"]}


# ───── Run pipeline with ecdf mode ─────
t0 = time.perf_counter()
print("Running pipeline with ECDF-only signals...", flush=True)
res, records = run_once("ecdf")
print(f"  {res['wall_runtime_sec']:.1f}s, {res['wfv_records']} records", flush=True)
print(f"  acc={res['wf_accuracy']} pnl={res['wf_pnl_proxy']}", flush=True)

sig_dist = Counter(r.get("signal", 0) for r in records)
out_dist = Counter()
for r in records:
    if r.get("signal", 0) != 0:
        o = r.get("outcome", 0)
        out_dist[1 if o > 0 else (-1 if o < 0 else 0)] += 1
print(f"  Signal dist: {dict(sig_dist)}", flush=True)
print(f"  Outcome dist (non-zero signal only): {dict(out_dist)}", flush=True)

# ───── 1. ECDF Calibration ─────
print(f"\n{HEADER}")
print("  ECDF CALIBRATION CURVE (ecdf-only pipeline)")
print(HEADER)

buckets = defaultdict(lambda: {"n": 0, "up": 0, "dn": 0, "sum_outcome": 0.0, "sum_signal": 0})
for r in records:
    ecdf = int(r.get("ecdf", 0.5) * 10)
    k = f"{ecdf/10:.1f}-{(ecdf+1)/10:.1f}"
    b = buckets[k]
    o = r.get("outcome", 0)
    b["n"] += 1
    b["sum_outcome"] += o
    if o > 0: b["up"] += 1
    elif o < 0: b["dn"] += 1

print(f"  {'Bucket':>10} {'Count':>7} {'Up%':>7} {'Dn%':>7} {'AvgOutcome':>10} {'Edginess':>9}")
for k in sorted(buckets):
    b = buckets[k]
    wp = b["up"] / b["n"] * 100 if b["n"] else 0
    lp = b["dn"] / b["n"] * 100 if b["n"] else 0
    ao = b["sum_outcome"] / b["n"] if b["n"] else 0
    edge = abs(wp - lp)
    print(f"  {k:>10} {b['n']:>7} {wp:>6.1f}% {lp:>6.1f}% {ao:>+10.4f} {edge:>8.1f}%")

# ───── 2. Test alternative signal mappings ─────
print(f"\n{HEADER}")
print("  ALTERNATIVE SIGNAL MAPPINGS (post-processed on same records)")
print(HEADER)

def sig_ecdf(rec):
    e = rec.get("ecdf", 0.5); d = e - 0.5
    return 1 if d > 0.05 else (-1 if d < -0.05 else 0)
def sig_ecdf_inv(rec):
    e = rec.get("ecdf", 0.5); d = 0.5 - e
    return 1 if d > 0.05 else (-1 if d < -0.05 else 0)
def sig_ecdf_tail(rec):
    e = rec.get("ecdf", 0.5)
    if e < 0.1: return 1
    if e > 0.9: return -1
    return 0
def sig_ecdf_nonlinear(rec):
    # Bucket-aware: buy if ecdf<0.1, sell if 0.2-0.9, neutral otherwise
    e = rec.get("ecdf", 0.5)
    if e < 0.1: return 1
    if 0.2 <= e <= 0.9: return -1
    return 0
def sig_always_sell(rec):
    return -1
def sig_always_buy(rec):
    return 1
def sig_original(rec):
    # The original signal from DOA-recorded data (ecdf - entropy with real entropy)
    return rec.get("signal", 0)

tests = [
    ("Original fusion (ecdf-entropy)", sig_original),
    ("ECDF-only (current baseline)", sig_ecdf),
    ("ECDF inverted", sig_ecdf_inv),
    ("ECDF tail (buy<0.1, sell>0.9)", sig_ecdf_tail),
    ("ECDF nonlinear (buy<0.1, sell 0.2-0.9)", sig_ecdf_nonlinear),
    ("Always SELL", sig_always_sell),
    ("Always BUY", sig_always_buy),
]
print(f"  {'Name':45s} {'Accuracy':>9} {'PnL':>9} {'NonZero':>8} {'Edge':>6}")
for name, fn in tests:
    result = wfv(records, fn)
    nz = sum(1 for r in records if fn(r) != 0)
    status = "EDGE" if result["edge_detected"] else ""
    print(f"  {name:45s} {result['accuracy']:.4f}   {result['pnl_proxy']:+8.1f} {nz:>8} {status:>6}")

# ───── 3. Entropy distribution summary ─────
print(f"\n{HEADER}")
print("  ENTROPY SUMMARY (real)")
print(HEADER)
ev = [r.get("entropy", 0) for r in records]
ev_sorted = sorted(ev)
n = len(ev_sorted)
print(f"  Range: {ev_sorted[0]:.4f} - {ev_sorted[-1]:.4f}")
print(f"  Mean: {sum(ev)/n:.4f}")
print(f"  Median: {ev_sorted[n//2]:.4f}")
print(f"  P10: {ev_sorted[int(n*0.1)]:.4f}  P90: {ev_sorted[int(n*0.9)]:.4f}")

print(f"\nTotal: {time.perf_counter()-t0:.1f}s")
