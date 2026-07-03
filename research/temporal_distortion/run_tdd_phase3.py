"""TDD Phase 3: Multi-timeframe analysis + session dependency check + robustness."""
import sys, json
from pathlib import Path
import numpy as np

SRC = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SRC))

from research.temporal_distortion.tdd_core import TDDCore, SYMBOLS, compute_sync_metrics
from research.temporal_distortion.tdd_counterfactual import interval_shuffle, compute_from_timestamps

def compute_baseline(core, horizons=[5, 20, 50]):
    baselines = {}
    for h in horizons:
        fut = core.future_returns[h]
        valid = ~np.isnan(fut)
        baselines[f"H{h}"] = {
            "p_up": float(np.mean(fut[valid] > 0)),
            "n": int(np.sum(valid)),
            "mean_ret": float(np.mean(fut[valid]))
        }
    return baselines

def compute_sync_at_window(core, bar_seconds, window_seconds):
    """Compute event rate and sync at a specific time window."""
    core.compute_event_rate(window_seconds)
    core.compute_acceleration(5)
    core.compute_distortion()
    core.build_bar_grid(bar_seconds)
    core.compute_future_returns([5, 20, 50])
    results = {}
    for h in [5, 20, 50]:
        sync = compute_sync_metrics(core.bar_alpha, core.bar_delta, core.future_returns[h], f"_H{h}")
        results[f"H{h}"] = {
            "sync_up": sync.get("sync_up_accel_high_delta", {}),
            "sync_down": sync.get("sync_down_decel_low_delta", {}),
        }
    return results

def run_phase3(symbols=None):
    if symbols is None:
        symbols = SYMBOLS
    report = {"phase": "TDD Phase 3 — Multi-Timeframe & Robustness", "symbols": {}}
    
    # Time windows to test: 1min, 5min, 15min, 30min, 60min event rate windows
    event_windows = [60, 300, 900, 1800, 3600]
    event_labels = ["1m", "5m", "15m", "30m", "60m"]
    bar_seconds = 300  # 5-min bars
    
    for sym in symbols:
        print(f"\n{'='*60}")
        print(f"Processing {sym}")
        print(f"{'='*60}")
        core = TDDCore(sym)
        core.load_ticks()
        core.detect_events()
        
        sym_result = {
            "n_events": len(core.events),
            "baselines": {},
            "sync_by_event_window": {},
            "interval_shuffle_comparison": {},
        }
        
        # Baseline
        core.compute_event_rate(60)
        core.compute_acceleration(5)
        core.compute_distortion()
        core.build_bar_grid(bar_seconds)
        core.compute_future_returns([5, 20, 50])
        sym_result["baselines"] = compute_baseline(core)
        print(f"  Baselines: {json.dumps(sym_result['baselines'])}")
        
        # Test each event rate window
        for ew, ewl in zip(event_windows, event_labels):
            ew_key = f"event_{ewl}"
            syn = compute_sync_at_window(core, bar_seconds, ew)
            sym_result["sync_by_event_window"][ew_key] = syn
            sh50 = syn.get("H50", {}).get("sync_up", {})
            sn50 = syn.get("H50", {}).get("sync_down", {})
            print(f"  Event window {ewl}: sync_up H50 P(up)={sh50.get('p_up', 'N/A')} (n={sh50.get('n', 0)}) | "
                  f"sync_down H50 P(up)={sn50.get('p_up', 'N/A')} (n={sn50.get('n', 0)})")
        
        # Interval shuffle comparison at best event window (60s based on Phase 1)
        print(f"  Interval shuffle (event window 60s):")
        best_ew = 60
        core.compute_event_rate(best_ew)
        core.compute_acceleration(5)
        core.compute_distortion()
        core.build_bar_grid(bar_seconds)
        core.compute_future_returns([5, 20, 50])
        
        real_h50 = compute_sync_metrics(core.bar_alpha, core.bar_delta, core.future_returns[50], f"{sym}_H50")
        real_sync_up = real_h50.get("sync_up_accel_high_delta", {})
        
        shuffle_results = []
        for run in range(10):
            base = TDDCore(sym)
            base.timestamps = interval_shuffle(core.timestamps, seed=run * 100 + 42)
            base.timestamps = np.sort(base.timestamps)
            base.ticks = core.ticks
            compute_from_timestamps(base, best_ew, 5)
            base.build_bar_grid(bar_seconds)
            base.compute_future_returns([5, 20, 50])
            syn_cf = compute_sync_metrics(base.bar_alpha, base.bar_delta, base.future_returns[50], f"{sym}_H50")
            sync_up_cf = syn_cf.get("sync_up_accel_high_delta", {})
            shuffle_results.append(sync_up_cf.get("p_up", np.nan))
        
        shuffle_mean = float(np.nanmean(shuffle_results)) if shuffle_results else np.nan
        shuffle_std = float(np.nanstd(shuffle_results)) if shuffle_results else np.nan
        sym_result["interval_shuffle_comparison"] = {
            "real_sync_up_p": real_sync_up.get("p_up"),
            "real_sync_up_n": real_sync_up.get("n", 0),
            "shuffle_mean_p": shuffle_mean,
            "shuffle_std_p": shuffle_std,
            "shuffle_n_values": shuffle_results,
            "edge_retention_ratio": round(float(real_sync_up.get("p_up", 0) / max(shuffle_mean, 0.001)), 3) if shuffle_mean and real_sync_up.get("p_up") else None,
        }
        print(f"    Real sync_up P(up)={real_sync_up.get('p_up', 'N/A')} (n={real_sync_up.get('n', 0)})")
        print(f"    Shuffle mean P(up)={shuffle_mean:.4f} +/- {shuffle_std:.4f}")
        print(f"    Edge retention ratio={sym_result['interval_shuffle_comparison']['edge_retention_ratio']}")
        
        report["symbols"][sym] = sym_result
    
    return report

if __name__ == "__main__":
    report = run_phase3()
    out_path = Path(__file__).resolve().parent / "reports"
    out_path.mkdir(exist_ok=True)
    with open(out_path / "TDD_PHASE3_REPORT.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved.")
