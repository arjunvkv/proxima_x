"""TDD Validation Lab Phase 5: Time Reversal Test — causality vs artifact."""
import sys, json
from pathlib import Path
import numpy as np
import polars as pl

SRC = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SRC))

from research.temporal_distortion.tdd_core import (
    TDDCore, SYMBOLS, compute_sync_metrics
)
from research.temporal_distortion.tdd_counterfactual import compute_from_timestamps


def run_time_reversal(symbols=None):
    if symbols is None:
        symbols = SYMBOLS
    report = {"phase": "TDD Phase 5 — Time Reversal Test", "symbols": {}}
    rows = []

    for sym in symbols:
        print(f"\n{'='*70}")
        print(f"  {sym} — Time Reversal Test")
        print(f"{'='*70}")

        # --- Forward pass ---
        core = TDDCore(sym)
        n_ticks = core.run_full_pipeline(
            window_seconds=60, bar_seconds=300, smooth=5, horizons=[5, 20, 50]
        )
        print(f"Forward: {n_ticks} ticks, {len(core.events)} events, {len(core.bar_times)} bars")

        fwd_sync = {}
        for h in [5, 20, 50]:
            hk = f"H{h}"
            fut = core.future_returns[h]
            sync_res = compute_sync_metrics(core.bar_alpha, core.bar_delta, fut, f"{sym}_{hk}_fwd")
            fwd_sync[hk] = sync_res.get("sync_up_accel_high_delta", {}).get("p_up", np.nan)

        # --- Time reversal ---
        # reversed_ts = t_max - (timestamps - t_min)[::-1]  ensures increasing timestamps
        ts = core.timestamps.astype(np.float64)
        t_min, t_max = ts[0], ts[-1]
        reversed_ts = t_max - (ts - t_min)[::-1]
        reversed_ts = reversed_ts.astype(np.int64)

        # Reverse tick dataframe order AND transform timestamps to be ascending
        t_min_tick = core.ticks["timestamp"].min()
        t_max_tick = core.ticks["timestamp"].max()
        rev_ticks = core.ticks[::-1]
        rev_ticks = rev_ticks.with_columns(
            (pl.lit(int(t_max_tick + t_min_tick)) - pl.col("timestamp")).alias("timestamp")
        )

        # Build reversed core
        rev = TDDCore(sym)
        rev.timestamps = reversed_ts
        rev.ticks = rev_ticks
        rev.lmbda = None
        rev.alpha = None
        rev.delta = None
        rev.bar_times = None
        rev.future_returns = {}

        # Run backward TDD
        compute_from_timestamps(rev, window_seconds=60, smooth=5)
        rev.build_bar_grid(bar_seconds=300)
        rev.compute_future_returns(horizons=[5, 20, 50])

        print(f"Reversed: {len(rev.timestamps)} events, {len(rev.bar_times)} bars")
        print(f"  Event timestamps monotonic: {np.all(np.diff(rev.timestamps) >= 0)}")
        print(f"  Event timestamps range: [{rev.timestamps[0]}, {rev.timestamps[-1]}]")
        rev_ts_sec = rev.ticks["timestamp"].to_numpy()
        print(f"  Tick timestamps monotonic: {np.all(np.diff(rev_ts_sec) >= 0)}")
        print(f"  Tick timestamps range: [{rev_ts_sec[0]}, {rev_ts_sec[-1]}]")

        rev_sync = {}
        for h in [5, 20, 50]:
            hk = f"H{h}"
            fut = rev.future_returns[h]
            sync_res = compute_sync_metrics(rev.bar_alpha, rev.bar_delta, fut, f"{sym}_{hk}_rev")
            rev_sync[hk] = sync_res.get("sync_up_accel_high_delta", {}).get("p_up", np.nan)

        # --- Comparison ---
        print(f"\n  {'Horizon':<10} {'Fwd P(up)':<12} {'Rev P(up)':<12} {'Diff':<12} {'Verdict':<15}")
        print(f"  {'-'*60}")
        sym_data = {"symbol": sym, "horizons": {}}
        for h in [5, 20, 50]:
            hk = f"H{h}"
            fp = fwd_sync.get(hk, np.nan)
            rp = rev_sync.get(hk, np.nan)
            diff = fp - rp if not (np.isnan(fp) or np.isnan(rp)) else np.nan
            # Artifact if reversed ≈ forward (within 0.1), causality if reversed ~0.5 or flips
            if np.isnan(fp) or np.isnan(rp):
                verdict = "INSUFFICIENT"
            elif abs(fp - 0.5) < 0.05 and abs(rp - 0.5) < 0.05:
                verdict = "NOISE"
            elif abs(diff) < 0.1:
                verdict = "ARTIFACT"
            elif abs(rp - 0.5) < 0.08:
                verdict = "CAUSAL"
            elif (fp > 0.55 and rp < 0.45) or (fp < 0.45 and rp > 0.55):
                verdict = "CAUSAL (FLIP)"
            else:
                verdict = "INCONCLUSIVE"
            print(f"  {hk:<10} {fp:<12.4f} {rp:<12.4f} {diff:<12.4f} {verdict:<15}")
            sym_data["horizons"][hk] = {
                "fwd_p_up": round(float(fp), 4) if not np.isnan(fp) else None,
                "rev_p_up": round(float(rp), 4) if not np.isnan(rp) else None,
                "diff": round(float(diff), 4) if not np.isnan(diff) else None,
                "verdict": verdict,
            }
            rows.append({"symbol": sym, "horizon": hk, "fwd_p_up": fp, "rev_p_up": rp, "diff": diff, "verdict": verdict})
        report["symbols"][sym] = sym_data

    # --- Overall conclusion ---
    print(f"\n{'='*70}")
    print(f"  OVERALL CONCLUSION")
    print(f"{'='*70}")
    artifact_count = sum(1 for r in rows if r["verdict"] == "ARTIFACT")
    causal_count = sum(1 for r in rows if r["verdict"] in ("CAUSAL", "CAUSAL (FLIP)"))
    noise_count = sum(1 for r in rows if r["verdict"] == "NOISE")
    other_count = sum(1 for r in rows if r["verdict"] not in ("ARTIFACT", "CAUSAL", "CAUSAL (FLIP)", "NOISE"))
    print(f"  Artifact: {artifact_count} | Causal: {causal_count} | Noise: {noise_count} | Inconclusive: {other_count}")
    if artifact_count > causal_count:
        conclusion = "DOMINANT ARTIFACT — TDD signal is symmetric in time, likely a statistical artifact"
    elif causal_count > artifact_count:
        conclusion = "DOMINANT CAUSAL — TDD signal breaks under time reversal, supports genuine temporal causality"
    else:
        conclusion = "MIXED — Further investigation needed"
    print(f"\n  CONCLUSION: {conclusion}")
    report["conclusion"] = conclusion
    report["summary_rows"] = rows
    return report


if __name__ == "__main__":
    report = run_time_reversal()
    out_path = Path(__file__).resolve().parent / "reports"
    out_path.mkdir(exist_ok=True)
    with open(out_path / "TDD_PHASE5_TIMEREVERSAL.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved to {out_path / 'TDD_PHASE5_TIMEREVERSAL.json'}")
