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
from research.directional_physics.dpl_11.core.signal_projection import FrozenSignalProjection
from research.directional_physics.dpl_12.core.trade_activation_gate import TradeActivationGate
from research.directional_physics.dpl_12.core.cohort_engine import (
    MomentumCohort, ReversionCohort, CrossAssetCohort,
    TransitionCohort, TCMAPriorCohort, CohortEnsemble
)
from research.directional_physics.dpl_12.core.consensus_voter import ConsensusVoter
from research.mechanism_discovery.energy_dynamics import EnergyDynamics
from research.mechanism_discovery.temporal_topology import TemporalTopology

DATA_DIR = dpl7_config.DATA_DIR
if DATA_DIR is None:
    DATA_DIR = os.path.join(PROJECT_ROOT, "data", "market")
SYMBOLS = dpl7_config.SYMBOLS
HORIZONS = [20]
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


def compute_da(signals, returns):
    if len(signals) < 2:
        return 0.0
    correct = sum(1 for s, r in zip(signals, returns) if (s > 0) == (r > 0))
    return correct / len(signals)


def compute_ic(scores, returns):
    if len(scores) < 3:
        return 0.0
    ic = float(np.corrcoef(scores, returns)[0, 1])
    return ic if not np.isnan(ic) else 0.0


def run_symbol_full(symbol, ed, tt):
    data = load_symbol(symbol)
    if data is None:
        return None, None, None, None, None
    n = data["n"]
    price = data["price"]
    max_h = max(HORIZONS)
    bridge = FeatureBridge(ed, tt)
    encoder = StateEncoder(dim=dpl7_config.LATENT_DIM)
    manifold = MarketManifold(alpha=dpl7_config.MANIFOLD_ALPHA)
    records = []
    z_list = []
    years_list = []
    features_list = []
    for idx in range(WARMUP, n - max_h):
        features = bridge.extract(data, idx)
        z = encoder.encode(features)
        zs = manifold.update(z)
        z_list.append(zs.copy())
        yr = year_from_ts(data["timestamps"][idx])
        years_list.append(yr)
        features_list.append(features)
        rec = {"idx": idx, "year": yr, "symbol": symbol}
        for h in HORIZONS:
            fwd_ret = (price[idx + h] - price[idx]) / price[idx]
            rec[f"return_h{h}"] = float(fwd_ret)
            rec[f"direction_h{h}"] = 1.0 if fwd_ret > 0 else 0.0
        records.append(rec)
    z_seq = np.array(z_list, dtype=np.float64)
    return records, z_seq, np.array(years_list), features_list, price


def main():
    print("=" * 80)
    print("  DPL-12 WALK-FORWARD: Cohort Consensus System")
    print("=" * 80)
    print(f"  Symbols: {', '.join(SYMBOLS)}")
    print(f"  Horizon: H20, Warmup: {WARMUP}")
    print("  Cohorts: Momentum, Reversion, CrossAsset, Transition, TCMAPrior")
    print("  Goal: replace single direction prediction with cohort consensus")

    ed = EnergyDynamics()
    tt = TemporalTopology()

    symbol_data = {}
    for sym in SYMBOLS:
        print(f"\n  Processing {sym}...", end=" ")
        sys.stdout.flush()
        records, z_seq, years, features_list, price = run_symbol_full(sym, ed, tt)
        if records is not None:
            print(f"{len(records)} records")
            symbol_data[sym] = {
                "records": records, "z_seq": z_seq, "years": years,
                "features_list": features_list, "price": price
            }

    results = {}

    for horizon in HORIZONS:
        print(f"\n{'=' * 60}")
        print(f"  H{horizon} COHORT CONSENSUS WALK-FORWARD")
        print(f"{'=' * 60}")

        all_split_results = []

        for sym, sd in symbol_data.items():
            records = sd["records"]
            z_seq = sd["z_seq"]
            years_arr = sd["years"]
            features_list = sd["features_list"]

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

                train_records = [records[i] for i in range(len(records)) if train_mask[i]]
                test_records = [records[i] for i in range(len(records)) if test_mask[i]]
                z_train = z_seq[train_mask]
                z_test = z_seq[test_mask]
                f_train = [features_list[i] for i in range(len(features_list)) if train_mask[i]]
                f_test = [features_list[i] for i in range(len(features_list)) if test_mask[i]]
                ret_test = np.array([r[f"return_h{horizon}"] for r in test_records])

                n_dim = dpl7_config.LATENT_DIM

                projector = FrozenSignalProjection(dim=n_dim)
                projector.fit(z_train, train_records, n_tcma_epochs=3)

                tcma_w = projector.tcma_W.copy()
                if tcma_w is None:
                    continue

                z_train_n = z_train.copy()
                z_test_n = z_test.copy()
                if projector.mu is not None and projector.sd is not None:
                    z_train_n = (z_train - projector.mu) / projector.sd
                    z_test_n = (z_test - projector.mu) / projector.sd
                z_train_norm = np.array([z / (np.linalg.norm(z) + 1e-8) for z in z_train_n])
                z_test_norm = np.array([z / (np.linalg.norm(z) + 1e-8) for z in z_test_n])

                z_train_a = np.array([tcma_w @ z for z in z_train_norm])
                z_test_a = np.array([tcma_w @ z for z in z_test_norm])
                al_train = np.array([z / (np.linalg.norm(z) + 1e-8) for z in z_train_a])
                al_test = np.array([z / (np.linalg.norm(z) + 1e-8) for z in z_test_a])

                tfield = TransitionField(k=20, sim_threshold=0.5)
                tfield.fit(al_train)

                train_rets = np.array([r[f"return_h{horizon}"] for r in train_records])
                train_scores = projector.raw_signal_batch(z_train)
                flow_w = np.mean(train_rets[:, None] * np.array([tfield.predict(z) for z in al_train]), axis=0)
                flow_w = flow_w / (np.linalg.norm(flow_w) + 1e-8) if np.linalg.norm(flow_w) > 0 else np.ones(n_dim) / np.sqrt(n_dim)

                ensemble = CohortEnsemble()
                ensemble.add(MomentumCohort(window=15, weight=1.0))
                ensemble.add(ReversionCohort(window=10, weight=0.8))
                ensemble.add(CrossAssetCohort(weight=1.2))
                ensemble.add(TransitionCohort(tfield, weight=0.7))
                ensemble.add(TCMAPriorCohort(projector, weight=1.0))

                voter = ConsensusVoter(min_cohorts=2, agreement_threshold=0.3, min_confidence=0.2)
                gate_engine = TradeActivationGate(tcma_window=15, entropy_window=30)

                test_gate_active, test_gate_scores = gate_engine.gate_active(z_test, f_test, threshold=0.0)

                all_votes = []
                for t in range(len(z_test)):
                    extra = {"flow_w": flow_w}

                    votes = ensemble.votes(t, al_test, f_test, extra)
                    all_votes.append(votes)

                consensus_signals, agreements, n_active = voter.batch_resolve(all_votes)

                tcma_baseline = projector.raw_signal_batch(z_test)
                tcma_dir = np.tanh(tcma_baseline)

                for t in range(len(test_gate_active)):
                    if not test_gate_active[t]:
                        consensus_signals[t] = 0.0

                gated_consensus_signals = consensus_signals.copy()

                baseline_da = compute_da(tcma_dir.tolist(), ret_test.tolist())
                consensus_da = compute_da(consensus_signals.tolist(), ret_test.tolist())
                gated_da = compute_da(gated_consensus_signals.tolist(), ret_test.tolist())
                baseline_ic = compute_ic(tcma_baseline, ret_test)
                consensus_ic = compute_ic(consensus_signals, ret_test)

                n_consensus_trades = int(np.sum(np.abs(consensus_signals) > 0.01))
                n_gated_trades = int(np.sum(np.abs(gated_consensus_signals) > 0.01))
                mean_agreement = float(np.mean(agreements[agreements > 0])) if np.sum(agreements > 0) > 0 else 0.0
                mean_n_active = float(np.mean(n_active[n_active > 0])) if np.sum(n_active > 0) > 0 else 0.0

                baseline_pnl = float(np.sum(tcma_dir * ret_test))
                consensus_pnl = float(np.sum(consensus_signals * ret_test))
                gated_pnl = float(np.sum(gated_consensus_signals * ret_test))

                split_label = f"{sym}_{train_years[0]}-{train_years[-1]}->{test_year}"

                sr = {
                    "split": split_label, "symbol": sym,
                    "n_test": len(ret_test),
                    "baseline_da": round(baseline_da, 4),
                    "consensus_da": round(consensus_da, 4),
                    "gated_da": round(gated_da, 4),
                    "baseline_ic": round(baseline_ic, 4),
                    "consensus_ic": round(consensus_ic, 4),
                    "baseline_pnl": round(baseline_pnl, 6),
                    "consensus_pnl": round(consensus_pnl, 6),
                    "gated_pnl": round(gated_pnl, 6),
                    "n_consensus_trades": n_consensus_trades,
                    "n_gated_trades": n_gated_trades,
                    "mean_agreement": round(mean_agreement, 4),
                    "mean_n_active_cohorts": round(mean_n_active, 2),
                    "n_gate_active": int(np.sum(test_gate_active)),
                }
                all_split_results.append(sr)

                impr_cons = consensus_da - baseline_da
                impr_gate = gated_da - baseline_da

                print(f"\n  {split_label}:")
                print(f"    Baseline DA={baseline_da:.4f} IC={baseline_ic:.4f}  Consensus DA={consensus_da:.4f} ({impr_cons:+.4f})  Gated DA={gated_da:.4f} ({impr_gate:+.4f})")
                print(f"    Consensus trades: {n_consensus_trades}/{len(ret_test)}  Gated trades: {n_gated_trades}/{len(ret_test)}  Agrmt={mean_agreement:.3f}  Cohorts={mean_n_active:.1f}")

        if all_split_results:
            baseline_das = [s["baseline_da"] for s in all_split_results]
            consensus_das = [s["consensus_da"] for s in all_split_results]
            gated_das = [s["gated_da"] for s in all_split_results]
            ns = len(all_split_results)

            n_cons_improved = sum(1 for s in all_split_results if s["consensus_da"] > s["baseline_da"])
            n_gate_improved = sum(1 for s in all_split_results if s["gated_da"] > s["baseline_da"])
            n_cons_above = sum(1 for v in consensus_das if v > 0.55)
            n_gate_above = sum(1 for v in gated_das if v > 0.55)

            print(f"\n  --- H{horizon} Summary ---")
            print(f"    Total symbol-splits: {ns}")
            print(f"    Mean baseline DA:  {np.mean(baseline_das):.4f}")
            print(f"    Mean consensus DA: {np.mean(consensus_das):.4f}")
            print(f"    Mean gated DA:     {np.mean(gated_das):.4f}")
            print(f"    Consensus improves:  {n_cons_improved}/{ns} = {round(n_cons_improved/ns*100,1)}%")
            print(f"    Gate improves:       {n_gate_improved}/{ns} = {round(n_gate_improved/ns*100,1)}%")
            print(f"    Consensus DA>0.55:   {n_cons_above}/{ns} = {round(n_cons_above/ns*100,1)}%")
            print(f"    Gated DA>0.55:       {n_gate_above}/{ns} = {round(n_gate_above/ns*100,1)}%")

            results[f"H{horizon}"] = {
                "n_splits": ns,
                "mean_baseline_da": round(float(np.mean(baseline_das)), 4),
                "mean_consensus_da": round(float(np.mean(consensus_das)), 4),
                "mean_gated_da": round(float(np.mean(gated_das)), 4),
                "consensus_improvement_rate": round(n_cons_improved / ns * 100, 1),
                "gated_improvement_rate": round(n_gate_improved / ns * 100, 1),
                "consensus_above_55_pct": round(n_cons_above / ns * 100, 1),
                "splits": all_split_results
            }

    print(f"\n{'=' * 80}")
    print("  DPL-12 AGGREGATE FINDINGS")
    print("=" * 80)

    for h_label, h_res in results.items():
        print(f"\n  {h_label}:")
        print(f"    Baseline DA:  {h_res['mean_baseline_da']:.4f}")
        print(f"    Consensus DA: {h_res['mean_consensus_da']:.4f}")
        print(f"    Gated DA:     {h_res['mean_gated_da']:.4f}")
        print(f"    Consensus improves: {h_res['consensus_improvement_rate']}% of splits")
        print(f"    Gate improves:      {h_res['gated_improvement_rate']}% of splits")
        print(f"    Consensus >0.55:    {h_res['consensus_above_55_pct']}% of splits")

    baseline_das_all = [s["baseline_da"] for h in results.values() for s in h["splits"]]
    consensus_das_all = [s["consensus_da"] for h in results.values() for s in h["splits"]]
    gated_das_all = [s["gated_da"] for h in results.values() for s in h["splits"]]

    if consensus_das_all:
        mean_base = np.mean(baseline_das_all)
        mean_cons = np.mean(consensus_das_all)
        mean_gate = np.mean(gated_das_all)
        n_cons_good = sum(1 for v in consensus_das_all if v > mean_base)
        n_gate_good = sum(1 for v in gated_das_all if v > mean_base)
        total_all = len(baseline_das_all)

        print(f"\n  Overall ({total_all} splits):")
        print(f"    Baseline DA:  {mean_base:.4f}")
        print(f"    Consensus DA: {mean_cons:.4f} (improves {n_cons_good}/{total_all})")
        print(f"    Gated DA:     {mean_gate:.4f} (improves {n_gate_good}/{total_all})")

        if mean_cons > mean_base * 1.05 and n_cons_good > total_all * 0.5:
            print("\n  VERDICT: COHORT CONSENSUS CONFIRMED (voting ensemble improves direction prediction)")
        elif mean_cons > mean_base:
            print("\n  VERDICT: MARGINAL (consensus helps but not decisively)")
        else:
            print("\n  VERDICT: NO EDGE (cohort consensus does not improve upon baseline)")

    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "reports")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "dpl12_walkforward_results.json")
    serializable = {}
    for k, v in results.items():
        serializable[k] = v
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\n  Results saved to: {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
