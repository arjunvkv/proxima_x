"""
DPL-8 Walk-Forward: Persistence Forecasting
  - Predict tau(t): how long latent directional coherence survives
  - Per-symbol processing (no cross-asset concatenation)
  - Ridge regression on trajectory + state features
  - Edge metric: persistence-weighted directional return
"""

import sys
import os
import json
import numpy as np
from datetime import datetime

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
sys.path.insert(0, PROJECT_ROOT)

import polars as pl

from research.directional_physics.dpl_8.core.persistence_labels import PersistenceLabeler, cosine
from research.directional_physics.dpl_8.core.trajectory_builder import TrajectoryBuilder
from research.directional_physics.dpl_8.features.persistence_features import PersistenceFeatureBuilder
from research.directional_physics.dpl_8.model.persistence_model import PersistenceModel
from research.directional_physics.dpl_8.model.survival_head import SurvivalHead
from research.directional_physics.dpl_8.inference.persistence_predictor import PersistenceSignalEngine
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
    timestamps = []
    for idx in range(WARMUP, n - max_h):
        features = bridge.extract(data, idx)
        z = encoder.encode(features)
        zs = manifold.update(z)
        z_list.append(zs.copy())
        rec = {"idx": idx, "timestamp": data["timestamps"][idx], "symbol": symbol, "year": None}
        ts = data["timestamps"][idx]
        if isinstance(ts, (int, float, np.integer)):
            rec["year"] = int(ts)
        elif isinstance(ts, datetime):
            rec["year"] = ts.year
        elif isinstance(ts, np.datetime64):
            rec["year"] = ts.astype("datetime64[Y]").astype(int) + 1970
        else:
            rec["year"] = int(str(ts)[:4])
        for h in HORIZONS:
            fwd_ret = (price[idx + h] - price[idx]) / price[idx]
            rec[f"return_h{h}"] = float(fwd_ret)
            rec[f"direction_h{h}"] = 1.0 if fwd_ret > 0 else 0.0
        for k, v in features.items():
            rec[k] = v
        for i in range(len(zs)):
            rec[f"z{i}"] = float(zs[i])
        records.append(rec)
        timestamps.append(rec["year"])
    z_seq = np.array(z_list, dtype=np.float64)
    return records, z_seq, np.array(timestamps)


def compute_c_index(pred, true):
    n = len(pred)
    if n < 3:
        return 0.5
    concordant = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            if abs(true[i] - true[j]) < 1e-6:
                continue
            total += 1
            if (true[i] > true[j] and pred[i] > pred[j]) or (true[i] < true[j] and pred[i] < pred[j]):
                concordant += 1
    return concordant / total if total > 0 else 0.5


def compute_pce(pred, true):
    return float(np.mean(np.abs(pred - true))) if len(pred) > 0 else 100.0


def edge_score(pred_tau, returns, directions):
    if len(pred_tau) < 2:
        return 0.0
    correct = np.sign(returns) == directions
    weight = np.clip(pred_tau / 50.0, 0, 1)
    return float(np.mean(correct * weight))


def compute_da(signals, returns):
    if len(signals) < 2:
        return 0.0
    correct = sum(1 for s, r in zip(signals, returns) if (s > 0) == (r > 0))
    return correct / len(signals)


def main():
    print("=" * 80)
    print("  DPL-8 WALK-FORWARD: Persistence Forecasting (per-symbol)")
    print("=" * 80)
    print(f"  Symbols: {', '.join(SYMBOLS)}")
    print(f"  Warmup: {WARMUP}, Horizons: {HORIZONS}")

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

    TRAJ_WINDOW = 20
    results = {}

    for horizon in HORIZONS:
        print(f"\n{'=' * 60}")
        print(f"  H{horizon} PERSISTENCE WALK-FORWARD")
        print(f"{'=' * 60}")

        all_split_results = []

        for sym, sd in symbol_data.items():
            records = sd["records"]
            z_seq = sd["z_seq"]
            years_arr = sd["years"]

            labeler = PersistenceLabeler(max_horizon=100, threshold_quantile=0.2)
            tau_true = labeler.compute_tau(z_seq)

            traj_builder = TrajectoryBuilder(window=TRAJ_WINDOW)
            traj_feats = traj_builder.build(z_seq)

            feat_builder = PersistenceFeatureBuilder()
            X_full = feat_builder.build(z_seq, traj_feats)

            unique_years = sorted(set(int(y) for y in years_arr if not np.isnan(y)))
            n_dim = dpl7_config.LATENT_DIM

            if len(unique_years) < 4:
                continue

            for split_idx in range(len(unique_years) - 3):
                train_years = unique_years[split_idx:split_idx + 2]
                test_year = unique_years[split_idx + 2]

                train_mask = np.isin(years_arr, train_years)
                test_mask = years_arr == test_year

                if np.sum(train_mask) < 100 or np.sum(test_mask) < 20:
                    continue

                X_train = X_full[train_mask].copy()
                y_train = tau_true[train_mask].copy()
                X_test = X_full[test_mask].copy()
                y_test = tau_true[test_mask].copy()

                train_ret = np.array([records[i][f"return_h{horizon}"] for i in range(len(records)) if train_mask[i]])
                test_ret = np.array([records[i][f"return_h{horizon}"] for i in range(len(records)) if test_mask[i]])

                model = PersistenceModel(alpha=1e-3)
                model.fit(X_train, y_train)
                pred_train = model.predict(X_train)
                pred_test = model.predict(X_test)

                train_ci = compute_c_index(pred_train, y_train)
                test_ci = compute_c_index(pred_test, y_test)
                train_pce = compute_pce(pred_train, y_train)
                test_pce = compute_pce(pred_test, y_test)

                train_dir = np.sign(np.sum(X_train[:, :n_dim], axis=1))
                test_dir = np.sign(np.sum(X_test[:, :n_dim], axis=1))

                train_es = edge_score(pred_train, train_ret, train_dir)
                test_es = edge_score(pred_test, test_ret, test_dir)
                test_da = compute_da(test_dir.tolist(), test_ret.tolist())

                test_sized_ret = np.sum([d * max(0.0, min(1.0, p / 50.0)) * r
                                         for d, p, r in zip(test_dir, pred_test, test_ret)])
                test_baseline = float(np.sum(test_ret))

                split_label = f"{sym}_{train_years[0]}-{train_years[-1]}->{test_year}"
                sr = {
                    "split": split_label, "symbol": sym,
                    "n_train": int(np.sum(train_mask)), "n_test": int(np.sum(test_mask)),
                    "train_c_index": round(train_ci, 4), "test_c_index": round(test_ci, 4),
                    "train_pce": round(train_pce, 2), "test_pce": round(test_pce, 2),
                    "train_edge_score": round(train_es, 4), "test_edge_score": round(test_es, 4),
                    "test_da": round(test_da, 4),
                    "test_pnl_baseline": round(test_baseline, 4),
                    "test_pnl_sized": round(test_sized_ret, 4),
                }
                all_split_results.append(sr)

                c_str = "YES" if test_ci > 0.60 else "MARGINAL" if test_ci > 0.55 else "NO"
                es_str = "YES" if test_es > 0.55 else "MARGINAL" if test_es > 0.50 else "NO"
                print(f"\n  {sym} {train_years[0]}-{train_years[-1]}->{test_year}:")
                print(f"    C-index: {test_ci:.4f} ({c_str}) | PCE: {test_pce:.1f} | Edge: {test_es:.4f} ({es_str}) | DA: {test_da:.4f}")
                print(f"    PnL: base={test_baseline:.4f} sized={test_sized_ret:.4f}")

        if all_split_results:
            test_cis = [s["test_c_index"] for s in all_split_results]
            test_ess = [s["test_edge_score"] for s in all_split_results]
            test_pces = [s["test_pce"] for s in all_split_results]
            test_das = [s["test_da"] for s in all_split_results]
            ns = len(all_split_results)

            n_ci = sum(1 for v in test_cis if v > 0.55)
            n_es = sum(1 for v in test_ess if v > 0.50)
            n_pce = sum(1 for v in test_pces if v < 20)
            n_da = sum(1 for v in test_das if v > 0.52)

            print(f"\n  --- H{horizon} Aggregate Summary ---")
            print(f"    Total splits (per-symbol): {ns}")
            print(f"    C-index >0.55:  {n_ci}/{ns} = {round(n_ci/ns*100,1)}%  (mean={np.mean(test_cis):.4f})")
            print(f"    Edge >0.50:     {n_es}/{ns} = {round(n_es/ns*100,1)}%  (mean={np.mean(test_ess):.4f})")
            print(f"    PCE <20 bars:   {n_pce}/{ns} = {round(n_pce/ns*100,1)}%  (mean={np.mean(test_pces):.1f})")
            print(f"    DA >52%:        {n_da}/{ns} = {round(n_da/ns*100,1)}%  (mean={np.mean(test_das):.4f})")

            results[f"H{horizon}"] = {
                "n_splits": ns,
                "c_index_survival_pct": round(n_ci / ns * 100, 1),
                "edge_survival_pct": round(n_es / ns * 100, 1),
                "pce_ok_pct": round(n_pce / ns * 100, 1),
                "da_survival_pct": round(n_da / ns * 100, 1),
                "mean_c_index": round(float(np.mean(test_cis)), 4),
                "mean_edge_score": round(float(np.mean(test_ess)), 4),
                "mean_pce": round(float(np.mean(test_pces)), 1),
                "splits": all_split_results
            }

    print(f"\n{'=' * 80}")
    print("  DPL-8 AGGREGATE FINDINGS")
    print("=" * 80)

    for h_label, h_res in results.items():
        print(f"\n  {h_label}:")
        print(f"    C-index >0.55: {h_res['c_index_survival_pct']}%  (mean={h_res['mean_c_index']})")
        print(f"    Edge >0.50:    {h_res['edge_survival_pct']}%  (mean={h_res['mean_edge_score']})")
        print(f"    PCE <20 bars:  {h_res['pce_ok_pct']}%  (mean={h_res['mean_pce']:.1f})")
        print(f"    DA >52%:       {h_res['da_survival_pct']}%")

    n_h = len(results)
    ci_mean = np.mean([h["mean_c_index"] for h in results.values()]) if n_h > 0 else 0
    es_mean = np.mean([h["mean_edge_score"] for h in results.values()]) if n_h > 0 else 0
    pce_mean = np.mean([h["mean_pce"] for h in results.values()]) if n_h > 0 else 0

    print(f"\n  Overall: C-index={ci_mean:.4f}  Edge={es_mean:.4f}  PCE={pce_mean:.1f}")

    if ci_mean > 0.55 and es_mean > 0.50 and pce_mean < 20:
        print("  VERDICT: POSITIVE PERSISTENCE EDGE (tau predictable from latent trajectory)")
    elif ci_mean > 0.52 or es_mean > 0.48:
        print("  VERDICT: MARGINAL (weak persistence structure)")
    else:
        print("  VERDICT: NO EDGE (persistence not predictable from latent state)")

    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "reports")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "dpl8_walkforward_results.json")
    serializable = {}
    for k, v in results.items():
        serializable[k] = v
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\n  Results saved to: {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
