"""TDD Phase 1: Event rate λ(t), acceleration α(t), distortion δ(t) computation + directional testing."""
import sys, json
from pathlib import Path
import numpy as np

SRC = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SRC))

from research.temporal_distortion.tdd_core import (
    TDDCore, SYMBOLS, compute_directional_metrics,
    compute_inflection_metrics, compute_sync_metrics
)

def run_phase1(symbols=None):
    if symbols is None:
        symbols = SYMBOLS
    report = {"phase": "TDD Phase 1 — Basic Event Rate Directional Testing", "symbols": {}}
    all_rows = []
    for sym in symbols:
        print(f"\n{'='*60}")
        print(f"Processing {sym}")
        print(f"{'='*60}")
        core = TDDCore(sym)
        n_ticks = core.run_full_pipeline(
            window_seconds=60, bar_seconds=300, smooth=5, horizons=[5, 20, 50]
        )
        print(f"Loaded {n_ticks} ticks, {len(core.events)} events, {len(core.bar_times)} bars")
        print(f"  rate range: [{np.nanmin(core.lmbda):.2f}, {np.nanmax(core.lmbda):.2f}] events/s")
        print(f"  accel range: [{np.nanmin(core.alpha):.4f}, {np.nanmax(core.alpha):.4f}] events/s2")
        print(f"  distort range: [{np.nanmin(core.delta):.4f}, {np.nanmax(core.delta):.4f}]")

        sym_result = {
            "n_ticks": n_ticks,
            "n_events": len(core.events),
            "n_bars": len(core.bar_times),
            "lmbda_stats": {
                "min": float(np.nanmin(core.lmbda)),
                "max": float(np.nanmax(core.lmbda)),
                "mean": float(np.nanmean(core.lmbda)),
                "std": float(np.nanstd(core.lmbda)),
            },
            "horizons": {},
        }
        # Test each horizon
        for h in [5, 20, 50]:
            horizon_key = f"H{h}"
            fut = core.future_returns[h]
            # Directional metrics for alpha + delta
            dir_results = compute_directional_metrics(core.bar_alpha, core.bar_delta, fut, f"{sym}_{horizon_key}")
            # Inflection metrics
            inf_results = compute_inflection_metrics(core.bar_alpha, fut, f"{sym}_{horizon_key}")
            # Sync metrics
            sync_results = compute_sync_metrics(core.bar_alpha, core.bar_delta, fut, f"{sym}_{horizon_key}")
            sym_result["horizons"][horizon_key] = {
                "n_valid_bars": int(np.sum(~np.isnan(fut))),
                "directional": {k: v for k, v in dir_results.items()},
                "inflection": {k: v for k, v in inf_results.items()},
                "sync": {k: v for k, v in sync_results.items()},
            }
            # Print summary
            best = max(dir_results.values(), key=lambda x: x.get("p_up", 0) if not np.isnan(x.get("p_up", np.nan)) else 0)
            print(f"\n  {horizon_key}: {sym_result['horizons'][horizon_key]['n_valid_bars']} valid bars")
            print(f"    Best directional: {best.get('p_up', 'N/A'):.4f} (n={best.get('n', 0)}) — {best.get('alpha_thresh', '?')}, {best.get('delta_thresh', '?')}" if best.get('p_up') else f"    Best directional: N/A")
            for k, v in inf_results.items():
                if v.get("p_up"):
                    print(f"    {k}: P(up)={v['p_up']:.4f} (n={v['n']})")
            for k, v in sync_results.items():
                if v.get("p_up"):
                    print(f"    {k}: P(up)={v['p_up']:.4f} (n={v['n']})")
            all_rows.append({"symbol": sym, "horizon": horizon_key, **sym_result["horizons"][horizon_key]})
        report["symbols"][sym] = sym_result
    report["summary"] = collapse_summary(all_rows)
    return report

def collapse_summary(rows):
    """Build a cross-symbol, cross-horizon summary."""
    lines = []
    for r in rows:
        sym = r.get("symbol", "?")
        h = r.get("horizon", "?")
        best_dir = max(r.get("directional", {}).values(), key=lambda x: x.get("p_up", 0) if not np.isnan(x.get("p_up", np.nan)) else 0)
        best_p = best_dir.get("p_up", "N/A") if best_dir else "N/A"
        best_n = best_dir.get("n", 0) if best_dir else 0
        inf_lines = [f"{k}:{v.get('p_up', 'N/A')}(n={v.get('n', 0)})" for k, v in r.get("inflection", {}).items() if v.get("p_up")]
        sync_lines = [f"{k}:{v.get('p_up', 'N/A')}(n={v.get('n', 0)})" for k, v in r.get("sync", {}).items() if v.get("p_up")]
        lines.append(f"{sym} {h}: {r['n_valid_bars']} bars, best_dir={best_p}(n={best_n}), inflect=[{' '.join(inf_lines)}], sync=[{' '.join(sync_lines)}]")
    return "\n".join(lines)

if __name__ == "__main__":
    report = run_phase1()
    out_path = Path(__file__).resolve().parent / "reports"
    out_path.mkdir(exist_ok=True)
    with open(out_path / "TDD_PHASE1_REPORT.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print("\n\n=== PHASE 1 SUMMARY ===")
    print(report["summary"])
    print(f"\nReport saved to {out_path / 'TDD_PHASE1_REPORT.json'}")
