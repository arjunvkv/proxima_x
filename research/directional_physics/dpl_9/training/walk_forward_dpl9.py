"""
DPL-9 Walk-Forward: Edge Activation Engine
  - Filter TCMA-aligned z(t) space for high-confidence edge conditions
  - Conditions: low entropy, high TCMA stability, volatility compression
  - Metric: baseline DA vs gated DA improvement
"""

import sys
import os
import json
import numpy as np
from datetime import datetime

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
sys.path.insert(0, PROJECT_ROOT)

import polars as pl

from research.directional_physics.dpl_9.core.edge_activation_engine import EdgeActivationEngine
from research.directional_physics.dpl_9.inference.trade_gate import TradeGate
from research.directional_physics.dpl7.features.feature_bridge import FeatureBridge
from research.directional_physics.dpl7.core.state_encoder import StateEncoder
from research.directional_physics.dpl7.core.market_manifold import MarketManifold
from research.directional_physics.dpl7.config import dpl7_config
from research.mechanism_discovery.energy_dynamics import EnergyDynamics
from research.mechanism_discovery.temporal_topology import TemporalTopology

DATA_DIR = dpl7_config.DATA_DIR
if DATA_DIR is None:
    DATA_DIR = os.path.join(PROJECT_ROOT, "data", "market")
SYMBOLS = dpl7_config.SYMBOLS
HORIZONS = dpl7_config.HORIZONS
WARMUP = dpl7_config.WARMUP_BARS


def load_symbol(symbol):
    path = os.path.join(DATA_DIR, f"{symbol}.parquet")
    if not os.path.exists(path):
        return None
    df = pl.read_parquet(path)
    price = df["close"].to_numpy().astype(np.float64)
    returns = np.diff(np.log(price), prepend=np.log(price[0]))
    volume = df["volume"].to_numpy().astype(np.float64) if "volume" in df.columns else np.ones(len(price))
    high = df["high"].to_numpy().astype(np.float64) if "high" in df.columns else price.copy()
    low = df["low"].to_numpy().astype(np.float64) if "low" in df.columns else price.copy()
    timestamps = df["timestamp"].to_numpy() if "timestamp" in df.columns else np.arange(len(price))
    return {"symbol": symbol, "price": price, "returns": returns,
            "volume": volume, "high": high, "low": low,
            "timestamps": timestamps, "n": len(price)}


def process_symbol(symbol, ed, tt):
    data = load_symbol(symbol)
    if data is None:
        return None, None, None
    n = data["n"]
    price = data["price"]
    max_h = max(HORIZONS)
    bridge = FeatureBridge(ed, tt)
    encoder = StateEncoder(dim=dpl7_config.LATENT_DIM)
    manifold = MarketManifold(alpha=dpl7_config.MANIFOLD_ALPHA)
    records = []
    z_list = []
    years_list = []
    for idx in range(WARMUP, n - max_h):
        features = bridge.extract(data, idx)
        z = encoder.encode(features)
        zs = manifold.update(z)
        z_list.append(zs.copy())
        ts = data["timestamps"][idx]
        yr = None
        if isinstance(ts, (int, float, np.integer)):
            yr = int(ts)
        elif isinstance(ts, datetime):
            yr = ts.year
        elif isinstance(ts, np.datetime64):
            yr = ts.astype("datetime64[Y]").astype(int) + 1970
        else:
            yr = int(str(ts)[:4])
        years_list.append(yr)
        rec = {"idx": idx, "year": yr, "symbol": symbol}
        for h in HORIZONS:
            fwd_ret = (price[idx + h] - price[idx]) / price[idx]
            rec[f"return_h{h}"] = float(fwd_ret)
            rec[f"direction_h{h}"] = 1.0 if fwd_ret > 0 else 0.0
        records.append(rec)
    z_seq = np.array(z_list, dtype=np.float64)
    return records, z_seq, np.array(years_list)


def main():
    print("=" * 80)
    print("  DPL-9 WALK-FORWARD: Edge Activation Engine")
    print("=" * 80)
    print(f"  Symbols: {', '.join(SYMBOLS)}")
    print(f"  Horizons: {HORIZONS}")
    print(f"  Warmup: {WARMUP}")

    ed = EnergyDynamics()
    tt = TemporalTopology()

    symbol_data = {}
    for sym in SYMBOLS:
        print(f"\n  Processing {sym}...", end=" ")
        sys.stdout.flush()
        records, z_seq, years = process_symbol(sym, ed, tt)
        if records is not None:
            print(f"{len(records)} records")
            symbol_data[sym] = {"records": records, "z_seq": z_seq, "years": years}

    results = {}

    for horizon in HORIZONS:
        print(f"\n{'=' * 60}")
        print(f"  H{horizon} EDGE ACTIVATION WALK-FORWARD")
        print(f"{'=' * 60}")

        all_symbol_results = []

        for sym, sd in symbol_data.items():
            records = sd["records"]
            z_seq = sd["z_seq"]
            years_arr = sd["years"]

            unique_years = sorted(set(int(y) for y in years_arr if y is not None))
            if len(unique_years) < 4:
                continue

            sym_results = []
            for split_idx in range(len(unique_years) - 3):
                train_years = unique_years[split_idx:split_idx + 2]
                test_year = unique_years[split_idx + 2]

                train_mask = np.array([y in train_years for y in years_arr])
                test_mask = np.array([y == test_year for y in years_arr])

                if np.sum(train_mask) < 100 or np.sum(test_mask) < 20:
                    continue

                engine = EdgeActivationEngine()
                engine.fit_thresholds(z_seq[train_mask])

                test_gate = engine.compute_gate(z_seq[test_mask])

                test_ret = np.array([records[i][f"return_h{horizon}"] for i in range(len(records)) if test_mask[i]])
                test_dir = np.sign(np.sum(z_seq[test_mask], axis=1))
                test_true_dir = np.sign(test_ret)

                baseline_da = float(np.mean(test_true_dir == test_dir))
                gated_idx = test_gate == 1.0
                n_gated = int(np.sum(gated_idx))

                if n_gated < 3:
                    gated_da = 0.0
                    coverage = 0.0
                    pnl_base = 0.0
                    pnl_gated = 0.0
                else:
                    gated_da = float(np.mean(test_true_dir[gated_idx] == test_dir[gated_idx]))
                    coverage = float(n_gated / len(test_dir))

                    pnl_base = float(np.sum(test_dir * test_ret))
                    gated_signals = test_dir[gated_idx] * (0.5 + 0.5 * np.abs(np.sum(z_seq[test_mask][gated_idx], axis=1))) ** 2
                    pnl_gated = float(np.sum(gated_signals * test_ret[gated_idx]))

                improvement = gated_da - baseline_da

                split_label = f"{sym}_{train_years[0]}-{train_years[-1]}->{test_year}"
                sr = {
                    "split": split_label, "symbol": sym,
                    "n_train": int(np.sum(train_mask)), "n_test": int(np.sum(test_mask)),
                    "n_gated": n_gated, "coverage": round(coverage, 4),
                    "baseline_da": round(baseline_da, 4),
                    "gated_da": round(gated_da, 4),
                    "improvement": round(improvement, 4),
                    "pnl_baseline": round(pnl_base, 4),
                    "pnl_gated": round(pnl_gated, 4),
                }
                sym_results.append(sr)
                all_symbol_results.append(sr)

                impr_str = f"+{improvement:.4f}" if improvement > 0 else f"{improvement:.4f}"
                print(f"\n  {split_label}:")
                print(f"    DA: {baseline_da:.4f} -> {gated_da:.4f} ({impr_str}) | Coverage: {coverage:.1%} | Gated: {n_gated}/{len(test_dir)}")

            if sym_results:
                sym_baseline = np.mean([s["baseline_da"] for s in sym_results])
                sym_gated = np.mean([s["gated_da"] for s in sym_results])
                sym_improv = np.mean([s["improvement"] for s in sym_results])
                print(f"\n  {sym} avg: baseline DA={sym_baseline:.4f} -> gated DA={sym_gated:.4f} (improvement={sym_improv:.4f})")

        if all_symbol_results:
            baseline_das = [s["baseline_da"] for s in all_symbol_results]
            gated_das = [s["gated_da"] for s in all_symbol_results]
            improvements = [s["improvement"] for s in all_symbol_results]
            coverages = [s["coverage"] for s in all_symbol_results]
            ns = len(all_symbol_results)

            n_improved = sum(1 for v in improvements if v > 0)
            n_gated_above = sum(1 for s in all_symbol_results if s["gated_da"] > 0.65)
            n_baseline_above = sum(1 for s in all_symbol_results if s["baseline_da"] > 0.65)

            print(f"\n  --- H{horizon} Aggregate Summary ---")
            print(f"    Total symbol-splits: {ns}")
            print(f"    Mean baseline DA: {np.mean(baseline_das):.4f}")
            print(f"    Mean gated DA:    {np.mean(gated_das):.4f}")
            print(f"    Mean improvement: {np.mean(improvements):.4f}")
            print(f"    Splits improved:  {n_improved}/{ns} = {round(n_improved/ns*100,1)}%")
            print(f"    Gated DA >0.65:   {n_gated_above}/{ns} = {round(n_gated_above/ns*100,1)}%")
            print(f"    Baseline DA >0.65:{n_baseline_above}/{ns} = {round(n_baseline_above/ns*100,1)}%")
            print(f"    Mean coverage:    {np.mean(coverages):.3f}")

            results[f"H{horizon}"] = {
                "n_splits": ns,
                "mean_baseline_da": round(float(np.mean(baseline_das)), 4),
                "mean_gated_da": round(float(np.mean(gated_das)), 4),
                "mean_improvement": round(float(np.mean(improvements)), 4),
                "improvement_rate": round(n_improved / ns * 100, 1),
                "gated_da_above_65_pct": round(n_gated_above / ns * 100, 1),
                "mean_coverage": round(float(np.mean(coverages)), 4),
                "splits": all_symbol_results
            }

    print(f"\n{'=' * 80}")
    print("  DPL-9 AGGREGATE FINDINGS")
    print("=" * 80)

    weighted_baseline = 0.0
    weighted_gated = 0.0
    weighted_improv = 0.0
    total_splits = 0

    for h_label, h_res in results.items():
        print(f"\n  {h_label}:")
        print(f"    Baseline DA: {h_res['mean_baseline_da']:.4f}")
        print(f"    Gated DA:    {h_res['mean_gated_da']:.4f}")
        print(f"    Improvement: {h_res['mean_improvement']:+.4f}")
        print(f"    Improved:    {h_res['improvement_rate']}% of splits")
        print(f"    Gated >0.65: {h_res['gated_da_above_65_pct']}% of splits")
        print(f"    Coverage:    {h_res['mean_coverage']:.1%}")
        weighted_baseline += h_res["mean_baseline_da"] * h_res["n_splits"]
        weighted_gated += h_res["mean_gated_da"] * h_res["n_splits"]
        weighted_improv += h_res["mean_improvement"] * h_res["n_splits"]
        total_splits += h_res["n_splits"]

    if total_splits > 0:
        overall_baseline = weighted_baseline / total_splits
        overall_gated = weighted_gated / total_splits
        overall_improv = weighted_improv / total_splits

        print(f"\n  Overall (weighted by splits):")
        print(f"    Baseline DA: {overall_baseline:.4f}")
        print(f"    Gated DA:    {overall_gated:.4f}")
        print(f"    Improvement: {overall_improv:+.4f}")

        positive_splits = sum(1 for h_res in results.values() for s in h_res["splits"] if s["improvement"] > 0)
        total_all = sum(h_res["n_splits"] for h_res in results.values())
        gated_above_65 = sum(1 for h_res in results.values() for s in h_res["splits"] if s["gated_da"] > 0.65)

        if overall_improv > 0.03 and gated_above_65 > total_all * 0.3:
            print("\n  VERDICT: POSITIVE EDGE ACTIVATION CONFIRMED (gating improves DA)")
        elif overall_improv > 0:
            print("\n  VERDICT: MARGINAL (weak gating effect, needs better threshold calibration)")
        else:
            print("\n  VERDICT: NO EDGE (edge activation filter does not improve DA)")

    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "reports")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "dpl9_walkforward_results.json")
    serializable = {}
    for k, v in results.items():
        serializable[k] = v
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\n  Results saved to: {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
