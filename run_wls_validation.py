"""WLS Walk-Forward Validation Entry Point.

Usage:
    python run_wls_validation.py                         # grid search (default 7 days)
    python run_wls_validation.py --days 14                # longer history
    python run_wls_validation.py --lam 0.1 --alpha 0.3    # single run, custom params
    python run_wls_validation.py --quick                  # quick minimal test (1 day)
"""

import argparse
import time
import sys
import os
import json
import numpy as np
import MetaTrader5 as mt5
from datetime import datetime, timedelta


def main():
    parser = argparse.ArgumentParser(description="WLS Walk-Forward Validation")
    parser.add_argument("--days", type=int, default=7, help="Days of history to load")
    parser.add_argument("--lam", type=float, default=None, help="WLS regularization")
    parser.add_argument("--alpha", type=float, default=None, help="Smoothing alpha")
    parser.add_argument("--shrink", type=float, default=None, help="Prior shrink")
    parser.add_argument("--window", type=int, default=24, help="Training window in M5 bars")
    parser.add_argument("--quick", action="store_true", help="Quick test: 1 day, single run")
    parser.add_argument("--no-grid", action="store_true", help="Skip grid search")
    parser.add_argument("--verbose", action="store_true", help="Verbose per-step output")
    parser.add_argument("--derivatives", action="store_true", help="Test WLS derivatives (velocity, acceleration, force) as predictors")
    args = parser.parse_args()

    if args.quick:
        args.days = 1
        args.no_grid = True

    single_mode = args.no_grid or args.lam is not None or args.alpha is not None or args.shrink is not None
    lam = args.lam if args.lam is not None else 0.01
    alpha = args.alpha if args.alpha is not None else 0.2
    shrink = args.shrink if args.shrink is not None else 0.0

    from research.wls_validation.data_loader import (
        discover_available_pairs, load_m5_bars, build_return_matrix, build_design_matrix,
    )
    from research.wls_validation.walk_forward import run_walk_forward, grid_search, run_derivative_validation

    print("Connecting to MT5...")
    if not mt5.initialize():
        print(f"ERROR: MT5 initialize failed: {mt5.last_error()}")
        sys.exit(1)
    print(f"MT5 version: {mt5.version()}")

    available_pairs = discover_available_pairs()
    print(f"Discovered {len(available_pairs)} available pairs: {', '.join(available_pairs)}")
    if len(available_pairs) < 8:
        print("ERROR: Need at least 8 pairs for a meaningful decomposition")
        mt5.shutdown()
        sys.exit(1)

    date_to = datetime.now()
    date_from = date_to - timedelta(days=args.days)
    print(f"\nLoading M5 data: {date_from.date()} to {date_to.date()} ({args.days} days)")

    rates_map = {}
    for i, sym in enumerate(available_pairs):
        t0 = time.time()
        rates = load_m5_bars(sym, date_from, date_to)
        if rates is None or len(rates) == 0:
            print(f"  [{i+1}/{len(available_pairs)}] {sym} -> NO DATA")
            continue
        rates_map[sym] = rates
        print(f"  [{i+1}/{len(available_pairs)}] {sym} -> {len(rates)} bars ({time.time()-t0:.1f}s)")

    if len(rates_map) < 8:
        print(f"ERROR: Only got data for {len(rates_map)} symbols, need at least 8")
        mt5.shutdown()
        sys.exit(1)

    valid_symbols = [s for s in available_pairs if s in rates_map]
    print(f"\nBuilding return matrix for {len(valid_symbols)} symbols...")
    returns, timestamps, final_symbols = build_return_matrix(rates_map, valid_symbols)
    print(f"Return matrix shape: {returns.shape}")
    print(f"Temporal range: {datetime.fromtimestamp(timestamps[0]).strftime('%Y-%m-%d %H:%M')} "
          f"to {datetime.fromtimestamp(timestamps[-1]).strftime('%Y-%m-%d %H:%M')}")
    nonzero_bars = np.sum(np.any(np.abs(returns) > 1e-12, axis=1))
    print(f"Bars with activity: {nonzero_bars}/{returns.shape[0]}")

    A, pair_labels = build_design_matrix(final_symbols)
    print(f"Design matrix: {A.shape[0]} pairs x {A.shape[1]} currencies")
    print(f"Rank of A: {np.linalg.matrix_rank(A):.0f} (max possible={A.shape[1]-1})")

    mt5.shutdown()

    if args.derivatives:
        from research.wls_validation.walk_forward import run_derivative_validation
        print(f"\n{'='*60}")
        print(f"DERIVATIVE TEST: level vs change vs accel vs force")
        print(f"{'='*60}")
        t0 = time.time()
        deriv_result = run_derivative_validation(
            returns, timestamps, pair_labels, A,
            lam=lam, smoothing_alpha=alpha, prior_shrink=shrink,
            verbose=args.verbose,
        )
        elapsed = time.time() - t0
        print(f"\nDerivative validation (elapsed={elapsed:.1f}s):")
        print(f"{'Predictor':<12} {'Horizon':<10} {'N':<8} {'MSE Skill':<12} {'Dir Acc':<10} {'IC':<10} {'Spread':<12}")
        print("-" * 74)
        for method in ["level", "change", "accel", "force"]:
            key = f"derivative_{method}"
            if key not in deriv_result:
                continue
            for h in [1, 3, 6, 12]:
                hkey = f"{h}_bar"
                if hkey not in deriv_result[key]:
                    continue
                r = deriv_result[key][hkey]
                hl = {1: "5m", 3: "15m", 6: "30m", 12: "60m"}[h]
                print(f"{method:<12} {hl:<10} {r.get('n',0):<8} "
                      f"{r.get('mse_skill',0):<12.4f} {r.get('dir_acc',0):<10.3f} "
                      f"{r.get('ic',0):<10.4f} {r.get('spread_return',0):<12.6f}")
        meta = deriv_result.get("_meta", {})
        print(f"\nMeta: lam={meta.get('lam')} alpha={meta.get('smoothing_alpha')} shrink={meta.get('prior_shrink')}")
        return

    horizons = [1, 3, 6, 12]
    horizon_labels = {1: "5min", 3: "15min", 6: "30min", 12: "60min"}

    if single_mode:
        print(f"\n{'='*60}")
        print(f"Single run: lam={lam}, alpha={alpha}, shrink={shrink}, window={args.window}")
        print(f"{'='*60}")
        t0 = time.time()
        result = run_walk_forward(
            returns, timestamps, pair_labels, A,
            horizons=horizons, lam=lam, window=args.window,
            smoothing_alpha=alpha, prior_shrink=shrink,
            verbose=args.verbose,
        )
        elapsed = time.time() - t0
        print(f"\nResults (elapsed={elapsed:.1f}s):")
        meta = result.get("_meta", {})
        print(f"Walk-forward steps: {meta.get('n_walk_forward_steps', '?')}")
        print(f"{'Horizon':<12} {'N_holdout':<12} {'Hold MSE Skill':<16} {'All MSE Skill':<16} {'Dir Acc':<10} {'IC':<10} {'Spread':<10}")
        print("-" * 86)
        for h in horizons:
            key = f"{h}_bar"
            if key in result:
                r = result[key]
                print(f"{horizon_labels[h]:<12} {r.get('n_holdout',0):<12} "
                      f"{r.get('holdout_mse_skill',0):<16.4f} {r.get('all_mse_skill',0):<16.4f} "
                      f"{r.get('holdout_dir_acc',0):<10.3f} {r.get('information_coefficient',0):<10.4f} "
                      f"{r.get('spread_return',0):<10.6f}")
            else:
                print(f"{horizon_labels[h]:<12} ERROR: {result.get(key, {}).get('error', 'unknown')}")
        print(f"\nMeta: {json.dumps(meta, indent=2)}")
    else:
        print(f"\n{'='*60}")
        print(f"Grid search: {args.days} days, window={args.window}")
        print(f"Parameters: lam × alpha × shrink = "
              f"{len([0.001,0.01,0.05,0.1,0.5,1.0,5.0])} × "
              f"{len([0.05,0.1,0.2,0.5,1.0])} × "
              f"{len([0.0,0.3,0.5,0.7,0.9])} = "
              f"{7*5*5} combinations")
        print(f"{'='*60}")
        t0 = time.time()
        grid_result = grid_search(
            returns, timestamps, pair_labels, A,
            horizons=horizons, window=args.window,
        )
        elapsed = time.time() - t0
        print(f"\n{'='*60}")
        print(f"GRID SEARCH COMPLETE ({elapsed:.1f}s)")
        print(f"{'='*60}")
        print(f"Best params: lam={grid_result['best_params']['lam']:.3f}, "
              f"alpha={grid_result['best_params']['smoothing_alpha']:.1f}, "
              f"shrink={grid_result['best_params']['prior_shrink']:.1f}")
        print(f"Best holdout MSE skill ({horizon_labels[horizons[0]]}): {grid_result['best_skill']:.4f}")
        print()
        print(f"Top 15 configurations by 5m holdout MSE skill:")
        sorted_results = sorted(
            [r for r in grid_result["results"] if r.get("1_holdout_skill") is not None],
            key=lambda x: x["1_holdout_skill"],
            reverse=True,
        )
        print(f"{'lam':<8} {'alpha':<8} {'shrink':<8} {'5m_skill':<12} {'5m_all':<12} {'5m_dir':<10} {'5m_IC':<10} {'15m_skill':<12}")
        print("-" * 72)
        for r in sorted_results[:15]:
            print(f"{r['lam']:<8.3f} {r['smoothing_alpha']:<8.2f} {r['prior_shrink']:<8.2f} "
                  f"{r.get('1_holdout_skill',0):<12.4f} {r.get('1_all_skill',0):<12.4f} "
                  f"{r.get('1_dir_acc',0):<10.3f} {r.get('1_ic',0):<10.4f} "
                  f"{r.get('3_holdout_skill',0):<12.4f}")

        print(f"\nWorst 5 (to see negative skill range):")
        for r in sorted_results[-5:]:
            print(f"  lam={r['lam']:.3f} α={r['smoothing_alpha']:.1f} s={r['prior_shrink']:.1f} "
                  f"→ 5m_skill={r.get('1_holdout_skill',0):.4f}")

        summary = {
            "best_params": grid_result["best_params"],
            "best_skill": grid_result["best_skill"],
            "total_combinations": len(grid_result["results"]),
            "elapsed_s": round(elapsed, 1),
            "n_pairs": len(final_symbols),
            "n_bars": returns.shape[0],
        }
        out_path = os.path.join("research", "wls_validation", "grid_search_results.json")
        with open(out_path, "w") as f:
            json.dump({"summary": summary, "results": grid_result["results"]}, f, indent=2, default=str)
        print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
