"""TDD Lab Phase 2: Horizon Audit — Does edge increase or collapse with horizon?"""
import sys, json
from pathlib import Path
import numpy as np

SRC = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SRC))

from research.temporal_distortion.tdd_core import TDDCore

HORIZONS = [1, 5, 10, 20, 50, 100, 200, 500]
SYMBOLS = ["EURJPY", "USDJPY"]

def run_horizon_audit():
    report = {"phase": "TDD Phase 2 — Horizon Audit", "horizons_tested": HORIZONS, "symbols": {}}

    for sym in SYMBOLS:
        print(f"\n{'='*70}")
        print(f"  {sym} — Horizon Stability Audit")
        print(f"{'='*70}")

        core = TDDCore(sym)
        n_ticks = core.load_ticks()
        n_events = core.detect_events()
        core.compute_event_rate(60)
        core.compute_acceleration(5)
        core.compute_distortion()
        core.build_bar_grid(300)
        core.compute_future_returns(HORIZONS)

        n_bars_total = len(core.bar_times)
        print(f"  Ticks: {n_ticks}, Events: {n_events}, Bars: {n_bars_total}")

        rows = []
        sym_data = {"n_ticks": n_ticks, "n_events": n_events, "n_bars_total": n_bars_total, "horizons": {}}

        # Header
        header = f"  {'Horizon':>8} | {'Bars':>10} | {'SyncUp':>8} | {'P(up|syn)':>10} | {'P(up|all)':>10} | {'Edge':>10} | {'MeanRet':>10}"
        sep = "  " + "-" * len(header.strip())
        print(header)
        print(sep)

        for h in HORIZONS:
            fut = core.future_returns[h]
            valid_mask = ~np.isnan(fut)
            n_valid = int(np.sum(valid_mask))

            sync_mask = (core.bar_alpha > 0) & (core.bar_delta > 1.0) & valid_mask
            n_sync = int(np.sum(sync_mask))

            if n_sync >= 5:
                sync_pup = float(np.mean(fut[sync_mask] > 0))
                sync_mean_ret = float(np.mean(fut[sync_mask]))
            else:
                sync_pup = None
                sync_mean_ret = None

            uncond_pup = float(np.mean(fut[valid_mask] > 0)) if n_valid > 0 else None
            edge = round(sync_pup - uncond_pup, 4) if sync_pup is not None and uncond_pup is not None else None

            pup_str = f"{sync_pup:.4f}" if sync_pup is not None else "N/A"
            uncond_str = f"{uncond_pup:.4f}" if uncond_pup is not None else "N/A"
            edge_str = f"{edge:+.4f}" if edge is not None else "N/A"
            ret_str = f"{sync_mean_ret:.6f}" if sync_mean_ret is not None else "N/A"

            print(f"  {'H'+str(h):>8} | {n_valid:>10} | {n_sync:>8} | {pup_str:>10} | {uncond_str:>10} | {edge_str:>10} | {ret_str:>10}")

            sym_data["horizons"][f"H{h}"] = {
                "n_bars_valid": n_valid,
                "n_sync_up": n_sync,
                "sync_up_p_up": sync_pup,
                "unconditional_p_up": uncond_pup,
                "edge_vs_baseline": edge,
                "mean_return": sync_mean_ret,
            }

        report["symbols"][sym] = sym_data

    return report

if __name__ == "__main__":
    report = run_horizon_audit()
    out_path = Path(__file__).resolve().parent / "reports"
    out_path.mkdir(exist_ok=True)
    out_json = out_path / "TDD_HORIZON_STABILITY.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Print conclusion summary
    print(f"\n{'='*70}")
    print("  HORIZON EDGE TRAJECTORY SUMMARY")
    print(f"{'='*70}")
    for sym in ["EURJPY", "USDJPY"]:
        print(f"\n  {sym}:")
        hdata = report["symbols"][sym]["horizons"]
        for h in HORIZONS:
            d = hdata[f"H{h}"]
            if d["edge_vs_baseline"] is not None:
                print(f"    H{h:>3}: edge={d['edge_vs_baseline']:+.4f}  P(up|syn)={d['sync_up_p_up']:.4f}  n_sync={d['n_sync_up']}")
            else:
                print(f"    H{h:>3}: N/A (n_sync={d['n_sync_up']})")

    # Determine peak horizon
    for sym in SYMBOLS:
        hdata = report["symbols"][sym]["horizons"]
        best_h = max(
            [h for h in HORIZONS if hdata[f"H{h}"]["edge_vs_baseline"] is not None],
            key=lambda h: hdata[f"H{h}"]["edge_vs_baseline"],
            default=None
        )
        if best_h is not None:
            print(f"\n  >> {sym}: Peak edge at H{best_h} ({hdata[f'H{best_h}']['edge_vs_baseline']:+.4f})")
            # Check decay pattern
            edges = [hdata[f"H{h}"]["edge_vs_baseline"] for h in HORIZONS if hdata[f"H{h}"]["edge_vs_baseline"] is not None]
            if len(edges) >= 3:
                first_half = edges[:len(edges)//2]
                second_half = edges[len(edges)//2:]
                trend = "DECAYING" if np.mean(first_half) > np.mean(second_half) + 0.005 else "IMPROVING" if np.mean(second_half) > np.mean(first_half) + 0.005 else "STABLE"
                print(f"  >> {sym}: Edge trajectory: {trend}")

    print(f"\n  Report saved to: {out_json}")
