"""TDD Validation Lab Phase 3: Event Definition Audit.

Tests whether TDD is robust to different event definitions on EURJPY.
"""
import sys, json
from pathlib import Path
import numpy as np
import polars as pl

SRC = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SRC))

from research.temporal_distortion.tdd_core import TDDCore, SYMBOLS, compute_sync_metrics
from research.temporal_distortion.tdd_counterfactual import compute_from_timestamps

SYMBOL = "EURJPY"
TICK_PATH = "data/ticks/EURJPY_ticks.parquet"


def extract_timestamps(ticks: pl.DataFrame, method: str, tick_range_window: int = 60) -> np.ndarray:
    """Extract event timestamps from tick data based on event definition method."""
    ts = ticks["timestamp"].to_numpy()
    bid = ticks["bid"].to_numpy()
    ask = ticks["ask"].to_numpy()
    spread = ticks["spread"].to_numpy()

    if method == "bid_changes":
        mask = np.zeros(len(ts), dtype=bool)
        mask[1:] = np.abs(np.diff(bid)) > 0
        return ts[mask]

    elif method == "ask_changes":
        mask = np.zeros(len(ts), dtype=bool)
        mask[1:] = np.abs(np.diff(ask)) > 0
        return ts[mask]

    elif method == "mid_price":
        mask = np.zeros(len(ts), dtype=bool)
        mask[1:] = np.abs(np.diff((bid + ask) / 2)) > 0
        return ts[mask]

    elif method == "spread_change":
        mask = np.zeros(len(ts), dtype=bool)
        spread_diff = np.abs(np.diff(spread))
        mask[1:] = spread_diff > 0.0001
        return ts[mask]

    elif method == "volume":
        vol = ticks["volume"].to_numpy()
        mask = vol > 0
        return ts[mask]

    elif method == "high_low_range":
        ticks_sec = ts.astype(np.float64) / 1_000_000
        t_start, t_end = ticks_sec[0], ticks_sec[-1]
        event_ts = []
        window_start = t_start
        i = 0
        while window_start < t_end:
            window_end = window_start + tick_range_window
            left = np.searchsorted(ticks_sec, window_start, side="left")
            right = np.searchsorted(ticks_sec, window_end, side="left")
            if right - left >= 2:
                window_bid_high = np.max(bid[left:right])
                window_bid_low = np.min(bid[left:right])
                window_range = window_bid_high - window_bid_low
                if window_range > 0.0001:
                    event_ts.append(ts[right - 1])
            i += 1
            window_start = window_end
        return np.array(event_ts, dtype=np.int64)

    raise ValueError(f"Unknown method: {method}")


def run_event_definition_audit(symbol: str = SYMBOL):
    print("=" * 70)
    print(f"TDD Validation Lab Phase 3 — Event Definition Audit ({symbol})")
    print("=" * 70)

    # Load ticks once
    root = Path(__file__).resolve().parent.parent.parent.parent
    full_path = root / TICK_PATH
    ticks = pl.read_parquet(str(full_path))
    ticks = ticks.sort("timestamp")
    print(f"\nLoaded {len(ticks):,} ticks")

    results = []

    event_methods = [
        ("bid_changes", "Bid changes only"),
        ("ask_changes", "Ask changes only"),
        ("mid_price", "Mid-price changes (default)"),
        ("spread_change", "Spread changes >0.0001"),
        ("volume", "Volume events (vol>0)"),
        ("high_low_range", "High-low range (60s window)"),
    ]

    for method_key, method_label in event_methods:
        ev_ts = extract_timestamps(ticks, method_key)
        n_events = len(ev_ts)

        if n_events == 0:
            print(f"  {method_label:45s} | n_events=0 — SKIP")
            results.append({
                "event_def": method_label,
                "method": method_key,
                "n_events": 0,
                "sync_up_n": 0,
                "sync_up_pup": None,
            })
            continue

        core = TDDCore(symbol)
        core.timestamps = ev_ts
        core.ticks = ticks
        core.future_returns = {}

        # Run TDD pipeline via compute_from_timestamps
        compute_from_timestamps(core, window_seconds=60, smooth=5)
        core.build_bar_grid(bar_seconds=300)
        core.compute_future_returns(horizons=[50])

        fut_h50 = core.future_returns[50]
        sync = compute_sync_metrics(
            core.bar_alpha, core.bar_delta, fut_h50, f"{symbol}_H50"
        )
        sync_up = sync.get("sync_up_accel_high_delta", {})
        sync_up_n = sync_up.get("n", 0)
        sync_up_pup = sync_up.get("p_up", None)

        print(f"  {method_label:45s} | n_events={n_events:>8,} | sync_up_n={sync_up_n:>5} | sync_up_P(up)={sync_up_pup}")
        results.append({
            "event_def": method_label,
            "method": method_key,
            "n_events": n_events,
            "sync_up_n": sync_up_n,
            "sync_up_pup": round(sync_up_pup, 4) if sync_up_pup is not None else None,
        })

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Event Definition':45s} | {'n_events':>9s} | {'sync_up_n':>9s} | {'sync_up_P(up)':>13s}")
    print("-" * 85)
    for r in results:
        pup_str = f"{r['sync_up_pup']:.4f}" if r['sync_up_pup'] is not None else "N/A"
        print(f"{r['event_def']:45s} | {r['n_events']:>9,} | {r['sync_up_n']:>9} | {pup_str:>13s}")
    print("-" * 85)

    # Robustness assessment
    valid = [r for r in results if r['sync_up_pup'] is not None and r['sync_up_n'] >= 10]
    if len(valid) >= 3:
        pups = np.array([r['sync_up_pup'] for r in valid])
        mean_p, std_p = np.mean(pups), np.std(pups)
        print(f"\nRobustness: mean P(up)={mean_p:.4f} +/- {std_p:.4f} across {len(valid)} event definitions")
        if std_p < 0.03:
            print("VERDICT: TDD is ROBUST to event definition (std < 0.03)")
        else:
            print("VERDICT: TDD is SENSITIVE to event definition (std >= 0.03)")
    else:
        print("\nVERDICT: Insufficient data for robustness conclusion")

    return results


if __name__ == "__main__":
    results = run_event_definition_audit()
    out_path = Path(__file__).resolve().parent / "reports"
    out_path.mkdir(exist_ok=True)
    with open(out_path / "EVENT_DEFINITION_AUDIT.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nReport saved to {out_path / 'EVENT_DEFINITION_AUDIT.json'}")
