import sys
import os
import json
import numpy as np
from datetime import datetime

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
sys.path.insert(0, PROJECT_ROOT)

import polars as pl

from research.directional_physics.dpl7.core.state_encoder import StateEncoder
from research.directional_physics.dpl7.core.market_manifold import MarketManifold
from research.directional_physics.dpl7.core.temporal_contrastive_aligner import TemporalContrastiveAligner
from research.directional_physics.dpl7.features.feature_bridge import FeatureBridge
from research.directional_physics.dpl7.config import dpl7_config
from research.directional_physics.dpl_10.core.transition_field import TransitionField
from research.directional_physics.dpl_10.core.flow_curvature import divergence_from_field, flow_coherence
from research.directional_physics.dpl_10.core.signal_from_flow import FlowSignalHead
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
    volume = df["volume"].to_numpy().astype(np.float64) if "volume" in df.columns else np.ones(len(price))
    high = df["high"].to_numpy().astype(np.float64) if "high" in df.columns else price.copy()
    low = df["low"].to_numpy().astype(np.float64) if "low" in df.columns else price.copy()
    timestamps = df["timestamp"].to_numpy() if "timestamp" in df.columns else np.arange(len(price))
    return {"symbol": symbol, "price": price, "volume": volume, "high": high, "low": low,
            "timestamps": timestamps, "n": len(price)}


def year_from_ts(ts):
    if isinstance(ts, (int, float, np.integer)):
        return int(ts)
    if isinstance(ts, datetime):
        return ts.year
    if isinstance(ts, np.datetime64):
        return ts.astype("datetime64[Y]").astype(int) + 1970
    return int(str(ts)[:4])


def run_symbol(symbol, ed, tt):
    data = load_symbol(symbol)
    if data is None:
        return None, None, None, None
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
        yr = year_from_ts(data["timestamps"][idx])
        years_list.append(yr)
        rec = {"idx": idx, "year": yr, "symbol": symbol}
        for k, v in features.items():
            rec[k] = v
        for h in HORIZONS:
            fwd_ret = (price[idx + h] - price[idx]) / price[idx]
            rec[f"return_h{h}"] = float(fwd_ret)
            rec[f"direction_h{h}"] = 1.0 if fwd_ret > 0 else 0.0
        records.append(rec)
    z_seq = np.array(z_list, dtype=np.float64)
    return records, z_seq, np.array(years_list), price


def compute_ic(scores, returns):
    if len(scores) < 3:
        return 0.0
    ic = float(np.corrcoef(scores, returns)[0, 1])
    return ic if not np.isnan(ic) else 0.0


def compute_da(signals, returns):
    if len(signals) < 2:
        return 0.0
    correct = sum(1 for s, r in zip(signals, returns) if (s > 0) == (r > 0))
    return correct / len(signals)


def main():
    print("=" * 80)
    print("  DPL-10 WALK-FORWARD: Transition Field Learning")
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
        records, z_seq, years, price = run_symbol(sym, ed, tt)
        if records is not None:
            print(f"{len(records)} records")
            symbol_data[sym] = {"records": records, "z_seq": z_seq, "years": years, "price": price}

    results = {}

    for horizon in HORIZONS:
        print(f"\n{'=' * 60}")
        print(f"  H{horizon} TRANSITION FIELD WALK-FORWARD")
        print(f"{'=' * 60}")

        all_split_results = []

        for sym, sd in symbol_data.items():
            records = sd["records"]
            z_seq = sd["z_seq"]
            years_arr = sd["years"]

            unique_years = sorted(set(int(y) for y in years_arr if y is not None))
            if len(unique_years) < 4:
                continue

            for split_idx in range(len(unique_years) - 3):
                train_years = unique_years[split_idx:split_idx + 2]
                test_year = unique_years[split_idx + 2]

                train_mask = np.array([y in train_years for y in years_arr])
                test_mask = np.array([y == test_year for y in years_arr])

                if np.sum(train_mask) < 100 or np.sum(test_mask) < 20:
                    continue

                n_dim = dpl7_config.LATENT_DIM
                z_train_raw = z_seq[train_mask]
                z_test_raw = z_seq[test_mask]

                mu = np.mean(z_train_raw, axis=0)
                sd_v = np.std(z_train_raw, axis=0) + 1e-8
                z_train_s = (z_train_raw - mu) / sd_v
                z_test_s = (z_test_raw - mu) / sd_v

                z_train_n = np.array([z / (np.linalg.norm(z) + 1e-8) for z in z_train_s])
                z_test_n = np.array([z / (np.linalg.norm(z) + 1e-8) for z in z_test_s])

                train_records = [records[i] for i in range(len(records)) if train_mask[i]]
                test_records = [records[i] for i in range(len(records)) if test_mask[i]]
                ret_train = np.array([r[f"return_h{horizon}"] for r in train_records])
                ret_test = np.array([r[f"return_h{horizon}"] for r in test_records])

                tcma = TemporalContrastiveAligner(dim=n_dim, tau=0.5, lambda_drift=0.1)
                tcma.pretrain(z_train_n, train_records, n_epochs=3, lr=0.001)

                z_train_a = tcma.align_batch(z_train_n)
                z_test_a = tcma.align_batch(z_test_n)

                al_train = np.array([z / (np.linalg.norm(z) + 1e-8) for z in z_train_a])
                al_test = np.array([z / (np.linalg.norm(z) + 1e-8) for z in z_test_a])

                tfield = TransitionField(k=20, sim_threshold=0.5)
                tfield.fit(al_train)

                train_expected_v, train_v_valid = tfield.predict_batch(al_train)
                test_expected_v, test_v_valid = tfield.predict_batch(al_test)

                train_v_use = train_expected_v[train_v_valid]
                ret_train_use = ret_train[train_v_valid]

                if len(train_v_use) < 20:
                    continue

                head = FlowSignalHead(dim=n_dim, lr=0.001)
                train_ic = head.fit(train_v_use, ret_train_use, epochs=10)

                train_divs, train_div_valid = divergence_from_field(tfield, al_train)
                train_coherence = flow_coherence(al_train)

                if np.sum(train_v_valid) < 2:
                    continue

                train_preds = head.predict_batch(train_expected_v)
                test_preds = head.predict_batch(test_expected_v)

                train_scores = np.where(train_v_valid, np.tanh(train_preds), 0.0)
                test_scores = np.where(test_v_valid, np.tanh(test_preds), 0.0)

                al_train_n = al_train[train_v_valid]
                test_divs, test_div_valid = divergence_from_field(tfield, al_test)
                test_coherence = flow_coherence(al_test)

                train_da = compute_da(train_scores[train_v_valid].tolist(), ret_train[train_v_valid].tolist())
                test_da = compute_da(test_scores[test_v_valid].tolist(), ret_test[test_v_valid].tolist())
                test_ic = compute_ic(test_preds[test_v_valid], ret_test[test_v_valid])

                flow_gate_train = np.ones(len(al_train))
                flow_gate_test = np.ones(len(al_test))
                for i in range(len(al_train)):
                    if train_div_valid[i]:
                        flow_gate_train[i] = 1.0 if train_divs[i] < 0 else 0.5
                for i in range(len(al_test)):
                    if test_div_valid[i]:
                        flow_gate_test[i] = 1.0 if test_divs[i] < 0 else 0.5

                gated_scores_train = train_scores * flow_gate_train
                gated_scores_test = test_scores * flow_gate_test

                gated_train_da = compute_da(gated_scores_train[train_v_valid].tolist(), ret_train[train_v_valid].tolist())
                gated_test_da = compute_da(gated_scores_test[test_v_valid].tolist(), ret_test[test_v_valid].tolist())

                n_train_total = len(al_train)
                n_test_total = len(al_test)
                n_train_flow = int(np.sum(train_v_valid))
                n_test_flow = int(np.sum(test_v_valid))

                split_label = f"{sym}_{train_years[0]}-{train_years[-1]}->{test_year}"
                sr = {
                    "split": split_label, "symbol": sym,
                    "n_train": n_train_total, "n_test": n_test_total,
                    "n_train_valid_flow": n_train_flow, "n_test_valid_flow": n_test_flow,
                    "train_ic": round(train_ic, 4), "test_ic": round(test_ic, 4),
                    "train_da": round(train_da, 4), "test_da": round(test_da, 4),
                    "gated_train_da": round(gated_train_da, 4),
                    "gated_test_da": round(gated_test_da, 4),
                }
                all_split_results.append(sr)

                print(f"\n  {split_label}:")
                print(f"    Train: {n_train_total} | Test: {n_test_total} | Flow-valid: {n_test_flow}/{n_test_total}")
                print(f"    Train IC: {train_ic:.4f} | Test IC: {test_ic:.4f}")
                print(f"    Train DA: {train_da:.4f} | Test DA: {test_da:.4f} | Gated Test DA: {gated_test_da:.4f}")

        if all_split_results:
            test_ics = [s["test_ic"] for s in all_split_results]
            test_das = [s["test_da"] for s in all_split_results]
            gtest_das = [s["gated_test_da"] for s in all_split_results]
            ns = len(all_split_results)

            n_ic_pos = sum(1 for v in test_ics if v > 0.02)
            n_da_pos = sum(1 for v in test_das if v > 0.52)
            n_gda_pos = sum(1 for v in gtest_das if v > 0.52)

            print(f"\n  --- H{horizon} Summary ---")
            print(f"    Total symbol-splits: {ns}")
            print(f"    Mean test IC: {np.mean(test_ics):.4f}")
            print(f"    Mean test DA: {np.mean(test_das):.4f}")
            print(f"    Mean gated test DA: {np.mean(gtest_das):.4f}")
            print(f"    IC survival (>0.02): {n_ic_pos}/{ns} = {round(n_ic_pos/ns*100,1)}%")
            print(f"    DA survival (>52%):  {n_da_pos}/{ns} = {round(n_da_pos/ns*100,1)}%")
            print(f"    Gated DA survival:   {n_gda_pos}/{ns} = {round(n_gda_pos/ns*100,1)}%")

            results[f"H{horizon}"] = {
                "n_splits": ns,
                "mean_test_ic": round(float(np.mean(test_ics)), 4),
                "mean_test_da": round(float(np.mean(test_das)), 4),
                "mean_gated_test_da": round(float(np.mean(gtest_das)), 4),
                "ic_survival_pct": round(n_ic_pos / ns * 100, 1),
                "da_survival_pct": round(n_da_pos / ns * 100, 1),
                "gated_da_survival_pct": round(n_gda_pos / ns * 100, 1),
                "splits": all_split_results
            }

    print(f"\n{'=' * 80}")
    print("  DPL-10 AGGREGATE FINDINGS")
    print("=" * 80)

    avg_test_ic = 0.0
    avg_test_da = 0.0
    avg_gated_da = 0.0
    total_n = 0

    for h_label, h_res in results.items():
        print(f"\n  {h_label}:")
        print(f"    Test IC:            {h_res['mean_test_ic']:.4f}")
        print(f"    Test DA:            {h_res['mean_test_da']:.4f}")
        print(f"    Gated Test DA:      {h_res['mean_gated_test_da']:.4f}")
        print(f"    IC survival (>0.02): {h_res['ic_survival_pct']}% of splits")
        print(f"    DA survival (>52%):  {h_res['da_survival_pct']}% of splits")
        print(f"    Gated DA survival:   {h_res['gated_da_survival_pct']}% of splits")
        avg_test_ic += h_res["mean_test_ic"] * h_res["n_splits"]
        avg_test_da += h_res["mean_test_da"] * h_res["n_splits"]
        avg_gated_da += h_res["mean_gated_test_da"] * h_res["n_splits"]
        total_n += h_res["n_splits"]

    if total_n > 0:
        avg_test_ic /= total_n
        avg_test_da /= total_n
        avg_gated_da /= total_n

        print(f"\n  Overall:")
        print(f"    Avg test IC:  {avg_test_ic:.4f}")
        print(f"    Avg test DA:  {avg_test_da:.4f}")
        print(f"    Avg gated DA: {avg_gated_da:.4f}")

        total_ic_pos = sum(1 for h in results.values() for s in h["splits"] if s["test_ic"] > 0.02)
        total_da_pos = sum(1 for h in results.values() for s in h["splits"] if s["test_da"] > 0.52)
        total_gda_pos = sum(1 for h in results.values() for s in h["splits"] if s["gated_test_da"] > 0.52)
        total_all = sum(h["n_splits"] for h in results.values())

        if avg_test_da > 0.58 and total_da_pos > total_all * 0.5:
            print("\n  VERDICT: POSITIVE EDGE CONFIRMED (transition field improves directional signal)")
        elif avg_test_da > 0.53:
            print("\n  VERDICT: MARGINAL (transition field adds weak signal)")
        elif avg_gated_da > 0.55:
            print("\n  VERDICT: MARGINAL (flow gating improves but raw signal weak)")
        else:
            print("\n  VERDICT: NO EDGE (transition field does not improve upon static readout)")

    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "reports")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "dpl10_walkforward_results.json")
    serializable = {}
    for k, v in results.items():
        serializable[k] = v
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\n  Results saved to: {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
