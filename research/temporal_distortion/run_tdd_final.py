"""TDD Phase 4: Final adjudication — walk-forward, cross-asset consistency, summary."""
import sys, json
from pathlib import Path
import numpy as np

SRC = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SRC))

from research.temporal_distortion.tdd_core import TDDCore, SYMBOLS, compute_sync_metrics
from research.temporal_distortion.tdd_counterfactual import interval_shuffle, compute_from_timestamps

def walk_forward_test(core, n_splits=5):
    """Walk-forward: split bar grid into n_splits, test each fold OOS."""
    n_bars = len(core.bar_times)
    fold_size = n_bars // n_splits
    idx = np.arange(n_bars)
    results = []
    for fold in range(n_splits):
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < n_splits - 1 else n_bars
        test_mask = (idx >= test_start) & (idx < test_end)
        train_mask = ~test_mask
        
        # Train: find optimal threshold on training set
        train_alpha = core.bar_alpha[train_mask]
        train_delta = core.bar_delta[train_mask]
        train_fut = core.future_returns[50][train_mask]
        valid = ~np.isnan(train_alpha) & ~np.isnan(train_delta) & ~np.isnan(train_fut)
        
        if np.sum(valid) < 50:
            continue
        
        # Test sync_up condition on test set
        test_alpha = core.bar_alpha[test_mask]
        test_delta = core.bar_delta[test_mask]
        test_fut = core.future_returns[50][test_mask]
        sync_up = (test_alpha > 0) & (test_delta > 1.0) & ~np.isnan(test_fut)
        n_test = np.sum(sync_up)
        if n_test >= 5:
            p_up = float(np.mean(test_fut[sync_up] > 0))
        else:
            p_up = None
        results.append({"fold": fold, "n_train": int(np.sum(valid)), "n_test": int(n_test), "p_up": p_up})
    return results

def check_session_dependency(core):
    """Check if sync signal depends on trading session."""
    bar_ts = core.bar_times  # seconds since epoch
    if len(bar_ts) == 0:
        return {}
    # Determine hour of day (UTC) for each bar
    hours = np.floor((bar_ts % 86400) / 3600).astype(int)
    # Sessions: Asian=0-8, London=8-16, NY=13-22
    sessions = np.full(len(hours), "other", dtype=object)
    sessions[(hours >= 0) & (hours < 8)] = "asian"
    sessions[(hours >= 8) & (hours < 16)] = "london"
    sessions[(hours >= 13) & (hours < 22)] = "ny"
    
    sync_up = (core.bar_alpha > 0) & (core.bar_delta > 1.0) & ~np.isnan(core.future_returns[50])
    results = {}
    for sess in ["asian", "london", "ny", "other"]:
        mask = sync_up & (sessions == sess)
        n = np.sum(mask)
        if n >= 10:
            p_up = float(np.mean(core.future_returns[50][mask] > 0))
            results[sess] = {"n": int(n), "p_up": round(p_up, 4)}
    return results

def run_final(symbols=None):
    if symbols is None:
        symbols = SYMBOLS
    report = {"phase": "TDD Phase 4 — Final Adjudication", "symbols": {}}
    
    for sym in symbols:
        print(f"\n{'='*60}")
        print(f"Processing {sym} — Final Adjudication")
        print(f"{'='*60}")
        
        core = TDDCore(sym)
        core.load_ticks()
        core.detect_events()
        core.compute_event_rate(60)
        core.compute_acceleration(5)
        core.compute_distortion()
        core.build_bar_grid(300)
        core.compute_future_returns([5, 20, 50])
        
        bl_p_up = float(np.nanmean(core.future_returns[50] > 0))
        
        # Sync metrics at all horizons
        sync_all = {}
        for h in [5, 20, 50]:
            sync = compute_sync_metrics(core.bar_alpha, core.bar_delta, core.future_returns[h], f"{sym}_H{h}")
            sync_all[f"H{h}"] = {
                "sync_up": sync.get("sync_up_accel_high_delta", {}),
                "sync_down": sync.get("sync_down_decel_low_delta", {}),
            }
        
        # Walk-forward
        wf_sync_up_ps = []
        wf = walk_forward_test(core, n_splits=5)
        for f in wf:
            if f.get("p_up") is not None:
                wf_sync_up_ps.append(f["p_up"])
        wf_mean_p = float(np.mean(wf_sync_up_ps)) if wf_sync_up_ps else None
        wf_std_p = float(np.std(wf_sync_up_ps)) if len(wf_sync_up_ps) > 1 else None
        
        # Interval shuffle (10 runs)
        shuffle_ps = []
        for run in range(10):
            base = TDDCore(sym)
            base.timestamps = interval_shuffle(core.timestamps, seed=run * 100 + 42)
            base.timestamps = np.sort(base.timestamps)
            base.ticks = core.ticks
            compute_from_timestamps(base, 60, 5)
            base.build_bar_grid(300)
            base.compute_future_returns([5, 20, 50])
            sync_cf = compute_sync_metrics(base.bar_alpha, base.bar_delta, base.future_returns[50], f"{sym}_H50")
            sync_up_cf = sync_cf.get("sync_up_accel_high_delta", {})
            if sync_up_cf.get("p_up") is not None:
                shuffle_ps.append(sync_up_cf["p_up"])
        shuffle_mean = float(np.mean(shuffle_ps)) if shuffle_ps else None
        shuffle_std = float(np.std(shuffle_ps)) if len(shuffle_ps) > 1 else None
        
        # Session dependency
        sessions = check_session_dependency(core)
        
        # Best event window
        best_window = None
        best_p = 0
        for ew in [60, 300, 900, 1800, 3600]:
            core.compute_event_rate(ew)
            core.compute_acceleration(5)
            core.compute_distortion()
            core.build_bar_grid(300)
            core.compute_future_returns([5, 20, 50])
            sync = compute_sync_metrics(core.bar_alpha, core.bar_delta, core.future_returns[50], f"{sym}_H50")
            sup = sync.get("sync_up_accel_high_delta", {})
            if sup.get("p_up") and sup["p_up"] > best_p:
                best_p = sup["p_up"]
                best_window = {"seconds": ew, "p_up": sup["p_up"], "n": sup["n"]}
        
        # Restore 60s window
        core.compute_event_rate(60)
        core.compute_acceleration(5)
        core.compute_distortion()
        core.build_bar_grid(300)
        core.compute_future_returns([5, 20, 50])
        
        real_p = sync_all["H50"]["sync_up"].get("p_up", None)
        edge_over_baseline = round(real_p - bl_p_up, 4) if real_p and bl_p_up else None
        edge_over_shuffle = round(real_p - shuffle_mean, 4) if real_p and shuffle_mean else None
        retention_ratio = round(real_p / shuffle_mean, 3) if real_p and shuffle_mean and shuffle_mean > 0 else None
        
        sym_result = {
            "n_ticks": len(core.ticks),
            "n_events": len(core.events),
            "n_bars": len(core.bar_times),
            "baseline_p_up_h50": round(bl_p_up, 4),
            "sync_all_horizons": sync_all,
            "walk_forward": {
                "folds": wf,
                "mean_p_up": wf_mean_p,
                "std_p_up": wf_std_p,
                "n_folds_with_trades": len(wf_sync_up_ps),
            },
            "interval_shuffle": {
                "real_p_up": round(real_p, 4) if real_p else None,
                "real_n": sync_all["H50"]["sync_up"].get("n", 0),
                "shuffle_mean_p_up": round(shuffle_mean, 4) if shuffle_mean else None,
                "shuffle_std_p_up": round(shuffle_std, 4) if shuffle_std else None,
                "edge_over_baseline": edge_over_baseline,
                "edge_over_shuffle": edge_over_shuffle,
                "retention_ratio": retention_ratio,
            },
            "session_dependency": sessions,
            "best_event_window": best_window,
        }
        
        print(f"  Baseline P(up) H50: {bl_p_up:.4f}")
        print(f"  Sync_up H50: P(up)={sync_all['H50']['sync_up'].get('p_up', 'N/A')} (n={sync_all['H50']['sync_up'].get('n', 0)})")
        print(f"  Edge over baseline: {edge_over_baseline}")
        print(f"  Shuffle mean: {shuffle_mean:.4f} +/- {shuffle_std:.4f}" if shuffle_mean else "  Shuffle: N/A")
        print(f"  Retention ratio: {retention_ratio}")
        print(f"  Walk-forward mean P(up): {wf_mean_p:.4f}" if wf_mean_p else "  Walk-forward: N/A")
        print(f"  Best event window: {best_window}")
        print(f"  Session dependency: {sessions}")
        
        # Adjudication
        if not real_p or real_p <= 0.5 + 0.02:
            adj = "NO_SIGNAL"
            reason = f"Sync_up P(up)={real_p} <= 0.52 threshold"
        elif retention_ratio and retention_ratio < 1.1:
            adj = "STRUCTURAL_ARTIFACT"
            reason = f"Retention ratio={retention_ratio} < 1.1 — interval shuffle preserves edge"
        elif wf_mean_p and wf_mean_p <= 0.5:
            adj = "WALK_FORWARD_FAILED"
            reason = f"Walk-forward mean P(up)={wf_mean_p} <= 0.5"
        elif real_p and edge_over_baseline and edge_over_baseline < 0.05:
            adj = "WEAK_MARGINAL_EDGE"
            reason = f"Edge over baseline={edge_over_baseline} < 5pp"
        else:
            adj = "DEPLOYABLE_DIRECTIONAL_EDGE"
            reason = f"Sync_up H50 P(up)={real_p}, retention_ratio={retention_ratio}, edge_over_baseline={edge_over_baseline}, WF mean={wf_mean_p}"
        
        sym_result["adjudication"] = adj
        sym_result["adjudication_reason"] = reason
        print(f"  ADJUDICATION: {adj}")
        print(f"  Reason: {reason}")
        
        report["symbols"][sym] = sym_result
    
    # Overall adjudication
    print(f"\n{'='*60}")
    print("OVERALL ADJUDICATION")
    print(f"{'='*60}")
    passed = [s for s, r in report["symbols"].items() if r.get("adjudication") == "DEPLOYABLE_DIRECTIONAL_EDGE"]
    failed = [s for s, r in report["symbols"].items() if r.get("adjudication") != "DEPLOYABLE_DIRECTIONAL_EDGE"]
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    
    if len(passed) >= 2:
        overall = "DEPLOYABLE_DIRECTIONAL_EDGE — Multi-asset confirmation"
    elif len(passed) >= 1:
        overall = f"LIMITED_DEPLOYABLE_EDGE — Single-asset ({passed[0]} only)"
    else:
        overall = "NO_DEPLOYABLE_EDGE"
    report["overall_adjudication"] = overall
    print(f"  OVERALL: {overall}")
    
    return report

if __name__ == "__main__":
    report = run_final()
    out_path = Path(__file__).resolve().parent / "reports"
    out_path.mkdir(exist_ok=True)
    with open(out_path / "TDD_FINAL_ADJUDICATION.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved.")
