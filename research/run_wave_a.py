"""Wave A': FULL_V4 vs HMS24 over 10-16 trading days + parameter drift"""
import sys; sys.path.insert(0, '.')
import time

from replay.environment import build_replay_environment, ReplayConfig
from replay.clock_patcher import patch_clock
from research.layer_config import LayerConfig
from research.experiment_config import HMS24_MINIMAL
from research.light_ablation_runner import LightAblationRunner

cfg = ReplayConfig(
    symbols=["EURJPY", "USDJPY"],
    start="2026-03-12", end="2026-03-28",
    speed=500000, burst=True, latency=False, slippage=False, seed=42,
)

TICK_LIMIT = 50000

def run_one(name, layer_cfg):
    print(f"\n=== {name} (building environment...) ===", flush=True)
    env = build_replay_environment(cfg)
    patch_clock(env.clock)
    print(f"Running with {TICK_LIMIT} ticks...", flush=True)
    r = LightAblationRunner(layer_cfg, tick_limit=TICK_LIMIT).run(env)
    print(f"Runtime: {r['wall_runtime_sec']:.1f}s  Ticks: {r['total_ticks']}  TPS: {r['ticks_per_second']:.0f}", flush=True)
    print(f"DOA={r['doa_evaluations']} AFL={r['afl_updates']} CAL={r['cal_updates']} FWO={r['fwo_updates']} RSL={r['rsl_updates']} RTD={r['rtd_detections']} SSOL={r['ssol_updates']} LCT={r['lct_updates']}", flush=True)
    print(f"WFV: acc={r['wf_accuracy']} pnl={r['wf_pnl_proxy']} edge={r['wf_edge_detected']}", flush=True)
    print(f"Rotations={r['rotation_changes']} Entropy={r['allocation_entropy']}", flush=True)
    pd = r["param_drift"]
    for pk, pv in pd.items():
        print(f"  {pk}: mean={pv['mean']} range={pv['range']} std={pv['std']}", flush=True)
    return r

r_full = run_one("FULL_V4", LayerConfig())
r_hms = run_one("HMS24_MINIMAL", HMS24_MINIMAL)

print("\n" + "=" * 60, flush=True)
print("  COMPARISON FULL_V4 vs HMS24_MINIMAL", flush=True)
print("=" * 60, flush=True)

for k in ["total_ticks", "wall_runtime_sec", "ticks_per_second", "doa_evaluations",
           "afl_updates", "cal_updates", "fwo_updates", "rsl_updates", "rtd_detections",
           "lct_updates", "ssol_updates", "rotation_changes", "active_symbol_count",
           "allocation_entropy", "wfv_records", "wf_accuracy", "wf_pnl_proxy", "wf_edge_detected"]:
    v1, v2 = r_full.get(k, "?"), r_hms.get(k, "?")
    if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
        delta = v2 - v1 if isinstance(v1, (int, float)) else 0
        print(f"  {k:25s} FULL={str(v1):>12}  HMS={str(v2):>12}  Delta={delta:+.4f}", flush=True)
    else:
        print(f"  {k:25s} FULL={v1}  HMS={v2}", flush=True)

print("", flush=True)
print("  PARAMETER DRIFT", flush=True)
for pk in r_full["param_drift"]:
    f = r_full["param_drift"][pk]
    h = r_hms["param_drift"][pk]
    print(f"  {pk:25s} FULL range={f['range']} std={f['std']}  HMS range={h['range']} std={h['std']}", flush=True)
