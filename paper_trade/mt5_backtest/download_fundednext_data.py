"""Download M1 data from FundedNext Server 3 for Challenge-Z validation."""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from datetime import datetime, timezone
import json

FN_TERMINAL = r"C:\Program Files\FundedNext MT5 Terminal\terminal64.exe"
PAIRS = ["AUDUSD", "EURAUD", "GBPAUD"]
FROM = datetime(2026, 6, 8)
TO = datetime(2026, 7, 26)
OUT = os.path.dirname(__file__)

def main():
    print(f"Connecting to FundedNext terminal: {FN_TERMINAL}")
    if not mt5.initialize(path=FN_TERMINAL):
        print(f"FAILED: {mt5.last_error()}")
        # Try without path
        print("Retrying without explicit path...")
        mt5.shutdown()
        if not mt5.initialize():
            print(f"FAILED AGAIN: {mt5.last_error()}")
            sys.exit(1)

    print(f"Terminal connected: {mt5.terminal_info().name}")
    print(f"Server: {mt5.account_info().server if mt5.account_info() else 'Not logged in'}")

    results = {}
    for pair in PAIRS:
        print(f"\n{'='*60}")
        print(f"Downloading {pair}...")
        rates = mt5.copy_rates_range(pair, mt5.TIMEFRAME_M1, FROM, TO)
        if rates is None or len(rates) == 0:
            print(f"  {pair}: NO DATA")
            results[pair] = {"error": "no data"}
            continue

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')

        # Spread stats
        spreads = df['spread'] * 1e5  # convert to pips
        zero_sprd = (spreads == 0).sum()
        stats = {
            "bars": len(df),
            "date_range": f"{df['time'].min()} to {df['time'].max()}",
            "zero_spread_bars": int(zero_sprd),
            "zero_spread_pct": round(float(zero_sprd / len(df) * 100), 2),
            "spread_min_pips": round(float(spreads.min()), 2),
            "spread_max_pips": round(float(spreads.max()), 2),
            "spread_mean_pips": round(float(spreads.mean()), 2),
            "spread_median_pips": round(float(spreads.median()), 2),
            "spread_p25_pips": round(float(spreads.quantile(0.25)), 2),
            "spread_p75_pips": round(float(spreads.quantile(0.75)), 2),
            "spread_p90_pips": round(float(spreads.quantile(0.90)), 2),
            "spread_p99_pips": round(float(spreads.quantile(0.99)), 2),
        }
        results[pair] = stats
        print(f"  Bars: {stats['bars']}  ({stats['date_range']})")
        print(f"  Zero-spread: {stats['zero_spread_bars']} ({stats['zero_spread_pct']}%)")
        print(f"  Spread (pips): min={stats['spread_min_pips']:.1f} "
              f"p25={stats['spread_p25_pips']:.1f} "
              f"med={stats['spread_median_pips']:.1f} "
              f"p75={stats['spread_p75_pips']:.1f} "
              f"p90={stats['spread_p90_pips']:.1f} "
              f"max={stats['spread_max_pips']:.1f}")

        # Save raw rates
        fname = f"fundednext_{pair.lower()}_m1.npy"
        fpath = os.path.join(OUT, fname)
        np.save(fpath, rates)
        print(f"  Saved to {fname}")

    mt5.shutdown()

    # Save summary
    summary_path = os.path.join(OUT, "fundednext_spread_report.json")
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSpread report saved to fundednext_spread_report.json")

    # Side-by-side comparison
    print(f"\n{'='*60}")
    print(f"COMPARISON: FTMO vs FundedNext Spreads")
    print(f"{'='*60}")
    print(f"{'Pair':<10} {'Source':<14} {'Med(pips)':<12} {'p90(pips)':<12} {'Zero%':<12}")
    print(f"{'-'*60}")
    # Load FTMO data if available
    for pair in PAIRS:
        fn_stats = results.get(pair, {})
        fn_med = fn_stats.get('spread_median_pips', 'N/A')
        fn_p90 = fn_stats.get('spread_p90_pips', 'N/A')
        fn_zero = fn_stats.get('zero_spread_pct', 'N/A')
        print(f"{pair:<10} {'FundedNext':<14} {fn_med:<12} {fn_p90:<12} {fn_zero:<12}")

    return results

if __name__ == "__main__":
    main()
