"""Signal Source Validation — three research tracks on base source quality.

Track 1: WFV Horizon Surface — sweep 5/10/20/50/100/250/500
Track 2: Direction inversion — signal vs -signal
Track 3: Signal-path influence — FULL vs HMS signal/rotation/allocation dynamics
"""
import sys; sys.path.insert(0, '.')
import time

from replay.environment import build_replay_environment, ReplayConfig
from replay.clock_patcher import patch_clock
from research.layer_config import LayerConfig
from research.experiment_config import HMS24_MINIMAL
from research.light_ablation_runner import LightAblationRunner

CFG = ReplayConfig(
    symbols=["EURJPY", "USDJPY"],
    start="2026-03-12", end="2026-03-28",
    speed=500000, burst=True, latency=False, slippage=False, seed=42,
)
TICK_LIMIT = 20000
HEADER = "=" * 70


def build_env():
    env = build_replay_environment(CFG)
    patch_clock(env.clock)
    return env


def run_one(layer_cfg, **kwargs):
    env = build_env()
    r = LightAblationRunner(layer_cfg, tick_limit=TICK_LIMIT, **kwargs).run(env)
    return r


# ──────────────────────────────────────────────
# TRACK 1 — WFV Horizon Surface
# ──────────────────────────────────────────────
def track1_horizon_surface():
    print(f"\n{HEADER}")
    print("  TRACK 1 — WFV Horizon Surface")
    print(HEADER)

    horizons = [5, 10, 20, 50, 100, 250, 500]
    results = []

    for h in horizons:
        r = run_one(LayerConfig(), doa_horizon=h)
        results.append({"horizon": h, "acc": r["wf_accuracy"], "pnl": r["wf_pnl_proxy"],
                        "edge": r["wf_edge_detected"], "tps": r["ticks_per_second"]})
        status = "EDGE" if r["wf_edge_detected"] else "no edge"
        print(f"  hor={h:4d}  acc={r['wf_accuracy']:.4f}  pnl={r['wf_pnl_proxy']:+.1f}  {status}  {r['ticks_per_second']:.0f} t/s")

    # Find best non-inverted horizon
    valid = [r for r in results if r["acc"] > 0.5]
    print(f"\n  Best acc > 0.5: horizon={valid[0]['horizon']} acc={valid[0]['acc']:.4f}" if valid else "\n  No horizon exceeds 0.5 accuracy")
    return results


# ──────────────────────────────────────────────
# TRACK 2 — Direction Inversion
# ──────────────────────────────────────────────
def track2_inversion():
    print(f"\n{HEADER}")
    print("  TRACK 2 — Direction Inversion Test")
    print(HEADER)

    best_horizon = None
    # Pick the best horizon from track 1 if available, or default 20
    for h in [5, 10, 20, 50, 100, 250, 500]:
        r = run_one(LayerConfig(), doa_horizon=h)
        if r["wf_accuracy"] > 0.5:
            best_horizon = h
            best_acc = r["wf_accuracy"]
            r_inv = run_one(LayerConfig(), doa_horizon=h, invert_signals=True)
            print(f"  Horizon={h}: normal acc={best_acc:.4f}  inverted acc={r_inv['wf_accuracy']:.4f}  inverted pnl={r_inv['wf_pnl_proxy']:+.1f}")
            break

    if best_horizon is None:
        # No horizon > 0.5 — test inversion on default 20
        r_norm = run_one(LayerConfig(), doa_horizon=20)
        r_inv = run_one(LayerConfig(), doa_horizon=20, invert_signals=True)
        print(f"  Horizon=20: normal acc={r_norm['wf_accuracy']:.4f}  inverted acc={r_inv['wf_accuracy']:.4f}")
        if r_inv["wf_accuracy"] > 0.5:
            print(f"  *** STRUCTURAL INVERSION DETECTED at horizon=20 ***")
        else:
            print(f"  No inversion detected at any tested horizon")

    return best_horizon


# ──────────────────────────────────────────────
# TRACK 3 — Signal-Path Influence
# ──────────────────────────────────────────────
def track3_influence():
    print(f"\n{HEADER}")
    print("  TRACK 3 — Signal-Path Influence Metrics")
    print(HEADER)

    r_full = run_one(LayerConfig(), track_signals=True)
    r_hms = run_one(HMS24_MINIMAL, track_signals=True)

    print(f"  {'Metric':30s} {'FULL_V4':>12} {'HMS24':>12}")
    print(f"  {'-'*30} {'-'*12} {'-'*12}")
    for k in ["total_ticks", "ticks_per_second", "doa_evaluations",
              "afl_updates", "signal_flip_count", "rotation_changes",
              "allocation_change_count", "allocation_entropy",
              "wf_accuracy", "wf_pnl_proxy", "wf_edge_detected"]:
        v1, v2 = r_full.get(k, "?"), r_hms.get(k, "?")
        print(f"  {k:30s} {str(v1):>12} {str(v2):>12}")

    # Compute flip rates
    n = max(r_full.get("total_ticks", 1), 1)
    print(f"\n  Flip rate (per tick):")
    print(f"    SIGNAL: FULL={r_full.get('signal_flip_count', 0)/n:.4f}  HMS={r_hms.get('signal_flip_count', 0)/n:.4f}")
    print(f"    ROTATION: FULL={r_full.get('rotation_changes', 0)/n:.4f}  HMS={r_hms.get('rotation_changes', 0)/n:.4f}")
    print(f"    ALLOCATION: FULL={r_full.get('allocation_change_count', 0)/n:.4f}  HMS={r_hms.get('allocation_change_count', 0)/n:.4f}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
if __name__ == "__main__":
    t0 = time.perf_counter()

    r1 = track1_horizon_surface()
    r2 = track2_inversion()
    track3_influence()

    elapsed = time.perf_counter() - t0
    print(f"\nTotal: {elapsed:.1f}s")
