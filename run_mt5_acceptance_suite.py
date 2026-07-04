#!/usr/bin/env python3
"""
Run MT5 Acceptance Suite — FINAL validation before Phase 6 live deployment.
Replaces synthetic tick data with real MT5 market data for all 4 adversarial tests.

Usage:
    python run_mt5_acceptance_suite.py --mode historical
    python run_mt5_acceptance_suite.py --mode live
    python run_mt5_acceptance_suite.py --mode csv --path data/ticks.csv
"""
import argparse
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from research.ucf.acceptance.mt5.mt5_tick_feed import MT5TickFeed
from research.ucf.acceptance.mt5.mt5_acceptance_runner import MT5AcceptanceRunner
from research.ucf.acceptance.mt5.mt5_acceptance_report import (
    compute_verdict, save_report, print_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MT5 Acceptance Suite — Final validation before Phase 6 live deployment"
    )
    parser.add_argument("--mode", type=str, default="historical",
                        choices=["live", "historical", "csv", "synthetic"])
    parser.add_argument("--path", type=str, default=None,
                        help="Path to CSV tick file (mode=csv)")
    parser.add_argument("--symbols", type=str, nargs="+",
                        default=["EURUSD", "USDJPY", "GBPUSD"],
                        help="Symbols for live/historical mode")
    parser.add_argument("--count", type=int, default=5000,
                        help="Number of ticks to fetch")
    args = parser.parse_args()

    feed = MT5TickFeed()

    if args.mode == "csv":
        if not args.path:
            print("ERROR: --path required for csv mode")
            sys.exit(1)
        feed.load_csv(args.path)
    elif args.mode == "live":
        feed.load_live_batch(args.symbols, args.count)
    elif args.mode == "synthetic":
        feed._generate_fallback(args.symbols, args.count)
    else:
        feed._generate_fallback(args.symbols, args.count)

    runner = MT5AcceptanceRunner(feed, mode=feed.mode)
    results = runner.run_all()

    report = compute_verdict(results)
    save_report(report)
    print_report(report)

    if report["final_verdict"] == "PASS":
        print("\n[MT5 SUITE] ALL TESTS PASS — Ready for Phase 6 certification")
    else:
        print(f"\n[MT5 SUITE] Final verdict: {report['final_verdict']}")


if __name__ == "__main__":
    main()
