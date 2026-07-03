"""TDD Counterfactual Gate: Poisson resampling and interval shuffle tests."""
import sys
from pathlib import Path
import numpy as np

SRC = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SRC))

from research.temporal_distortion.tdd_core import (
    TDDCore, SYMBOLS, compute_directional_metrics,
    compute_inflection_metrics, compute_sync_metrics
)


def poisson_resample(timestamps_us: np.ndarray, seed: int = 42) -> np.ndarray:
    """Resample events as Poisson process with same count and time range."""
    rng = np.random.default_rng(seed)
    t_min = timestamps_us[0]
    t_max = timestamps_us[-1]
    duration = t_max - t_min
    n = len(timestamps_us)
    # Generate uniform inter-arrival times (exponential distribution)
    # Scale to match the same duration
    intervals = rng.exponential(scale=duration / n, size=n)
    synthetic = np.cumsum(intervals).astype(np.int64) + t_min
    # Clip to original range
    synthetic = np.clip(synthetic, t_min, t_max)
    return synthetic


def interval_shuffle(timestamps_us: np.ndarray, seed: int = 42) -> np.ndarray:
    """Shuffle inter-event intervals (preserve count, destroy clustering structure)."""
    rng = np.random.default_rng(seed)
    intervals = np.diff(timestamps_us)
    rng.shuffle(intervals)
    synthetic = np.zeros_like(timestamps_us)
    synthetic[0] = timestamps_us[0]
    np.cumsum(intervals, out=synthetic[1:])
    synthetic[1:] += timestamps_us[0]
    return synthetic


def run_counterfactual(symbol, event_ts, real_ticks, n_synthetic: int = 5):
    """Run counterfactual tests: Poisson resampling and interval shuffle."""
    results = {"symbol": symbol, "n_real_events": len(event_ts), "synthetic_runs": []}
    for method, gen_fn in [("poisson", poisson_resample), ("interval_shuffle", interval_shuffle)]:
        for run in range(n_synthetic):
            base = TDDCore(symbol)
            base.timestamps = gen_fn(event_ts, seed=run * 10 + hash(method) % 1000)
            base.timestamps = np.sort(base.timestamps)
            base.lmbda = None
            base.alpha = None
            base.delta = None
            base.bar_times = None
            base.future_returns = {}
            base.ticks = real_ticks  # use real tick prices for future returns
            # Compute event rate on synthetic timestamps
            compute_from_timestamps(base, window_seconds=60, smooth=5)
            base.build_bar_grid(bar_seconds=300)
            base.compute_future_returns(horizons=[5, 20, 50])
            run_results = {}
            for h in [5, 20, 50]:
                hk = f"H{h}"
                fut = base.future_returns[h]
                dir_res = compute_directional_metrics(base.bar_alpha, base.bar_delta, fut, f"{symbol}_{hk}_synth")
                inf_res = compute_inflection_metrics(base.bar_alpha, fut, f"{symbol}_{hk}_synth")
                sync_res = compute_sync_metrics(base.bar_alpha, base.bar_delta, fut, f"{symbol}_{hk}_synth")
                best_dir = max(dir_res.values(), key=lambda x: x.get("p_up", 0) if not np.isnan(x.get("p_up", np.nan)) else 0)
                run_results[hk] = {
                    "best_dir_p": best_dir.get("p_up", np.nan) if best_dir else np.nan,
                    "best_dir_n": best_dir.get("n", 0) if best_dir else 0,
                    "sync_up_p": sync_res.get("sync_up_accel_high_delta", {}).get("p_up", np.nan),
                    "sync_up_n": sync_res.get("sync_up_accel_high_delta", {}).get("n", 0),
                    "any_cross_p": inf_res.get("any_cross", {}).get("p_up", np.nan),
                }
            results["synthetic_runs"].append({"method": method, "run": run, "results": run_results})
    return results


def compute_from_timestamps(core: TDDCore, window_seconds: int = 60, smooth: int = 5):
    """Compute event rate, acceleration, distortion from existing timestamps on a core."""
    ts = core.timestamps
    ts_sec = ts.astype(np.float64) / 1_000_000
    n = len(ts_sec)
    window_start = ts_sec - window_seconds
    left_idx = np.searchsorted(ts_sec, window_start, side="left")
    right_idx = np.arange(n)
    core.lmbda = (right_idx - left_idx + 1).astype(np.float64) / window_seconds
    core._window_seconds = window_seconds
    core.alpha = np.full_like(core.lmbda, np.nan)
    if smooth < n:
        core.alpha[smooth:] = core.lmbda[smooth:] - core.lmbda[:-smooth]
    baseline = np.nanpercentile(core.lmbda, 50)
    core.delta = core.lmbda / baseline if baseline > 0 else np.ones_like(core.lmbda)


def run_phase2(symbols=None, n_synthetic: int = 5):
    if symbols is None:
        symbols = SYMBOLS
    report = {"phase": "TDD Phase 2 — Counterfactual Gate", "symbols": {}}
    for sym in symbols:
        print(f"\n{'='*60}")
        print(f"Processing {sym} - Counterfactual")
        print(f"{'='*60}")
        # Load real data
        core = TDDCore(sym)
        n_ticks = core.load_ticks()
        core.detect_events()
        print(f"Real data: {len(core.events)} events")
        # Run counterfactual
        cf_results = run_counterfactual(sym, core.timestamps, core.ticks, n_synthetic=n_synthetic)
        report["symbols"][sym] = cf_results
        # Print comparison
        # Compute real data metrics for comparison
        core.compute_event_rate(60)
        core.compute_acceleration(5)
        core.compute_distortion()
        core.build_bar_grid(300)
        core.compute_future_returns([5, 20, 50])
        print(f"  Comparison (real vs synthetic mean over {n_synthetic} runs):")
        for h in [5, 20, 50]:
            hk = f"H{h}"
            fut = core.future_returns[h]
            dir_real = compute_directional_metrics(core.bar_alpha, core.bar_delta, fut, f"{sym}_{hk}")
            sync_real = compute_sync_metrics(core.bar_alpha, core.bar_delta, fut, f"{sym}_{hk}")
            best_real = max(dir_real.values(), key=lambda x: x.get("p_up", 0) if not np.isnan(x.get("p_up", np.nan)) else 0)
            real_sync_up = sync_real.get("sync_up_accel_high_delta", {}).get("p_up", np.nan)
            syn_p_values = [r["results"][hk]["best_dir_p"] for r in cf_results["synthetic_runs"]]
            syn_sync_up = [r["results"][hk]["sync_up_p"] for r in cf_results["synthetic_runs"]]
            syn_p_mean = np.nanmean(syn_p_values) if syn_p_values else np.nan
            syn_sync_mean = np.nanmean(syn_sync_up) if syn_sync_up else np.nan
            print(f"  {hk}: real_best={best_real.get('p_up', 'N/A'):.4f} (n={best_real.get('n', 0)}) | "
                  f"synth_best_mean={syn_p_mean:.4f} | "
                  f"real_sync_up={real_sync_up:.4f} | synth_sync_up_mean={syn_sync_mean:.4f}")
    return report


if __name__ == "__main__":
    report = run_phase2(n_synthetic=5)
    out_path = Path(__file__).resolve().parent / "reports"
    out_path.mkdir(exist_ok=True)
    import json
    with open(out_path / "TDD_PHASE2_REPORT.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved to {out_path / 'TDD_PHASE2_REPORT.json'}")
