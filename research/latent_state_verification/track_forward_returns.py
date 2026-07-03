"""Post-hoc forward return tracking for SignalFunnel signals.

Usage:
    python research/latent_state_verification/track_forward_returns.py

This reads the funnel_stats.json, looks up price data from the tick-built M1 parquet
files, and computes forward returns (H5/H20/H50) for each signal.

Respects V2.5 freeze: does NOT modify any live pipeline code.
"""
import sys, json, os
from pathlib import Path
import numpy as np
import duckdb

SRC = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SRC))

FUNNEL_FILE = SRC / "proxima_ops" / "data" / "funnel_stats.json"
DATA_DIR = SRC / "data" / "ticks"
M1_DIR = SRC / "exports" / "python_reference_m1"

SYMBOL_M1_MAP = {
    "EURJPY": M1_DIR / "eurjpy_m1_tickbuilt.parquet",
    "USDJPY": M1_DIR / "usdjpy_m1_tickbuilt.parquet",
    "GBPUSD": M1_DIR / "gbpusd_m1_tickbuilt.parquet",
    "EURUSD": M1_DIR / "eurusd_m1_tickbuilt.parquet",
    "XAUUSD": None,
}

FUTURE_BARS = {5: "forward_return_H5", 20: "forward_return_H20", 50: "forward_return_H50"}

def load_m1_data(symbol: str, con: duckdb.DuckDBPyConnection):
    """Load M1 tickbuilt data for a symbol."""
    m1_path = SYMBOL_M1_MAP.get(symbol)
    if m1_path is None or not m1_path.exists():
        return None, None
    try:
        cols = con.execute(f"DESCRIBE SELECT * FROM '{m1_path}'").fetchall()
        col_names = [c[0] for c in cols]
        close_col = "close" if "close" in col_names else ("bid" if "bid" in col_names else None)
        if close_col is None:
            return None, None
        df = con.execute(f"SELECT minute_bin, {close_col} as price FROM '{m1_path}' ORDER BY minute_bin").fetchall()
        minute_bins = np.array([r[0] for r in df], dtype=np.int64)
        prices = np.array([r[1] for r in df], dtype=np.float64)
        return minute_bins, prices
    except Exception:
        return None, None


def compute_forward_returns(minute_bin: int, minute_bins: np.ndarray, prices: np.ndarray):
    """Compute forward returns at H5/H20/H50 from a given minute."""
    result = {}
    for h_bars, field in FUTURE_BARS.items():
        target_bin = minute_bin + h_bars
        idx = np.searchsorted(minute_bins, target_bin, side="left")
        if idx < len(minute_bins):
            # Find the original minute index
            orig_idx = np.searchsorted(minute_bins, minute_bin, side="left")
            if orig_idx < len(minute_bins) and minute_bins[orig_idx] == minute_bin:
                future_ret = (prices[idx] - prices[orig_idx]) / prices[orig_idx]
                result[field] = round(float(future_ret), 6)
    return result


def main():
    if not FUNNEL_FILE.exists():
        print(f"Funnel file not found: {FUNNEL_FILE}")
        return

    with open(FUNNEL_FILE) as f:
        data = json.load(f)

    signals = data.get("signals", {})
    if not signals:
        print("No signals found in funnel file.")
        return

    con = duckdb.connect()
    updated = 0
    skipped_missing_price = 0
    skipped_no_lookup = 0
    skipped_no_data = 0
    already_tracked = 0

    for sig_id, sig in signals.items():
        # Skip if already tracked
        if sig.get("forward_return_H5") is not None:
            already_tracked += 1
            continue

        symbol = sig.get("symbol")
        ts_str = sig.get("timestamp_generated")
        if not symbol or not ts_str:
            skipped_missing_price += 1
            continue

        # Parse timestamp
        try:
            from datetime import datetime
            ts = datetime.fromisoformat(ts_str)
            minute_bin = int(ts.timestamp() / 60)
        except Exception:
            skipped_missing_price += 1
            continue

        # Load M1 data on first use
        m1_bins, m1_prices = load_m1_data(symbol, con)
        if m1_bins is None:
            skipped_no_data += 1
            continue

        # Find the minute_bin in M1 data
        idx = np.searchsorted(m1_bins, minute_bin, side="left")
        if idx >= len(m1_bins) or m1_bins[idx] != minute_bin:
            if idx > 0:
                idx = idx - 1  # use previous minute as closest
            else:
                skipped_no_lookup += 1
                continue

        actual_bin = m1_bins[idx]
        fwd = compute_forward_returns(actual_bin, m1_bins, m1_prices)
        if fwd:
            sig.update(fwd)
            updated += 1

    # Save back
    if updated > 0:
        with open(FUNNEL_FILE, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Updated {updated}/{len(signals)} signals with forward returns.")
    else:
        print("No signals were updated.")

    print(f"  Already tracked: {already_tracked}")
    print(f"  Skipped (no price data available): {skipped_no_data}")
    print(f"  Skipped (minute not found): {skipped_no_lookup}")
    print(f"  Skipped (missing symbol/timestamp): {skipped_missing_price}")

    # Summary
    tracked = [s for s in signals.values() if s.get("forward_return_H5") is not None]
    pe_tracked = [s for s in tracked if s.get("final_state") == "BLOCKED_POSITION_EXISTS"]
    ex_tracked = [s for s in tracked if s.get("final_state") in ("POSITION_OPENED", "POSITION_CLOSED")]

    print(f"\nSignals with forward returns: {len(tracked)}")
    print(f"  POSITION_EXISTS blocked: {len(pe_tracked)}")
    print(f"  Executed/closed: {len(ex_tracked)}")

    if pe_tracked:
        mean_h5 = np.mean([s["forward_return_H5"] for s in pe_tracked if s["forward_return_H5"] is not None])
        mean_h50 = np.mean([s["forward_return_H50"] for s in pe_tracked if s["forward_return_H50"] is not None])
        print(f"  Blocked avg H5 return: {mean_h5:.6f}")
        print(f"  Blocked avg H50 return: {mean_h50:.6f}")

    if ex_tracked:
        mean_h5 = np.mean([s["forward_return_H5"] for s in ex_tracked if s["forward_return_H5"] is not None])
        mean_h50 = np.mean([s["forward_return_H50"] for s in ex_tracked if s["forward_return_H50"] is not None])
        print(f"  Executed avg H5 return: {mean_h5:.6f}")
        print(f"  Executed avg H50 return: {mean_h50:.6f}")

    con.close()


if __name__ == "__main__":
    main()
