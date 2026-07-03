"""
DPL-7.1: Temporal Contrastive Market Alignment (TCMA) Walk-Forward
  - Self-supervised latent alignment before directional evaluation
  - Positive pairs: same structure + nearby time
  - Negative pairs: different volatility regime + distant time
  - Metrics: latent stability, IC stability, directional accuracy, PnL
"""

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
from research.directional_physics.dpl7.policy.signal_head import SignalHead
from research.directional_physics.dpl7.policy.sizing_policy import SizingPolicy
from research.directional_physics.dpl7.features.feature_bridge import FeatureBridge
from research.directional_physics.dpl7.config import dpl7_config
from research.mechanism_discovery.energy_dynamics import EnergyDynamics
from research.mechanism_discovery.temporal_topology import TemporalTopology

DATA_DIR = dpl7_config.DATA_DIR
if DATA_DIR is None:
    DATA_DIR = os.path.join(PROJECT_ROOT, "data", "market")
SYMBOLS = dpl7_config.SYMBOLS
HORIZONS = dpl7_config.HORIZONS
WARMUP = dpl7_config.WARMUP_BARS


def load_symbol(symbol: str) -> dict:
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
    return {
        "symbol": symbol, "price": price, "returns": returns,
        "volume": volume, "high": high, "low": low,
        "timestamps": timestamps, "n": len(price)
    }


def run_symbol(symbol: str, ed: EnergyDynamics, tt: TemporalTopology, horizons: list = None):
    if horizons is None:
        horizons = HORIZONS
    data = load_symbol(symbol)
    if data is None:
        return []
    n = data["n"]
    price = data["price"]
    max_horizon = max(horizons)
    bridge = FeatureBridge(ed, tt)
    encoder = StateEncoder(dim=dpl7_config.LATENT_DIM)
    manifold = MarketManifold(alpha=dpl7_config.MANIFOLD_ALPHA)
    records = []
    for idx in range(WARMUP, n - max_horizon):
        features = bridge.extract(data, idx)
        z = encoder.encode(features)
        z_smooth = manifold.update(z)
        record = {
            "idx": idx,
            "timestamp": data["timestamps"][idx],
            "symbol": symbol,
        }
        for k, v in features.items():
            record[k] = v
        for i in range(len(z_smooth)):
            record[f"z{i}"] = float(z_smooth[i])
        for h in horizons:
            fwd_ret = (price[idx + h] - price[idx]) / price[idx]
            direction = 1.0 if fwd_ret > 0 else 0.0
            record[f"return_h{h}"] = float(fwd_ret)
            record[f"direction_h{h}"] = direction
        records.append(record)
    return records


def year_from_record(r):
    ts = r["timestamp"]
    if isinstance(ts, (int, float, np.integer)):
        return int(ts)
    if isinstance(ts, datetime):
        return ts.year
    if isinstance(ts, np.datetime64):
        return ts.astype("datetime64[Y]").astype(int) + 1970
    return int(str(ts)[:4])


def compute_ic(z_arr, ret_arr):
    if len(z_arr) < 3:
        return 0.0
    ic = float(np.corrcoef(z_arr, ret_arr)[0, 1])
    return ic if not np.isnan(ic) else 0.0


def compute_da(signals, returns):
    if len(signals) < 2:
        return 0.0
    correct = sum(1 for s, r in zip(signals, returns) if (s > 0) == (r > 0))
    return correct / len(signals)


def compute_latent_drift(z_train, z_test):
    if len(z_train) < 1 or len(z_test) < 1:
        return 1.0
    z_train_mean = np.mean(np.array(z_train), axis=0)
    z_test_mean = np.mean(np.array(z_test), axis=0)
    drift = float(np.linalg.norm(z_train_mean - z_test_mean))
    avg_norm = float(np.mean([np.linalg.norm(z) for z in z_train])) + 1e-8
    return drift / avg_norm


def compute_lss(z_series):
    if len(z_series) < 3:
        return 0.0
    sims = []
    z_arr = np.array(z_series)
    for i in range(len(z_arr) - 1):
        a = z_arr[i] / (np.linalg.norm(z_arr[i]) + 1e-8)
        b = z_arr[i + 1] / (np.linalg.norm(z_arr[i + 1]) + 1e-8)
        sims.append(float(np.dot(a, b)))
    return float(np.mean(sims))


def compute_tdi(z_series, ema_alpha=0.1):
    if len(z_series) < 2:
        return 0.0
    ema = None
    tdis = []
    for z in z_series:
        if ema is None:
            ema = z.copy()
        else:
            ema = ema_alpha * z + (1 - ema_alpha) * ema
        tdis.append(float(np.linalg.norm(z - ema)))
    return float(np.mean(tdis))


def simulate_pnl(signals, returns):
    if len(signals) < 1:
        return {"baseline": 0.0, "sized": 0.0, "improvement": 0.0, "n_trades": 0}
    baseline = float(np.sum(returns))
    sized = float(np.sum([s * r for s, r in zip(signals, returns)]))
    return {
        "baseline": round(baseline, 4),
        "sized": round(sized, 4),
        "improvement": round(sized - baseline, 4),
        "n_trades": sum(1 for s in signals if abs(s) > 0.01)
    }


def run(horizons=None, symbols=None):
    if horizons is None:
        horizons = HORIZONS
    if symbols is None:
        symbols = list(SYMBOLS)

    print("=" * 80)
    print("  DPL-7.1 WALK-FORWARD: Temporal Contrastive Market Alignment")
    print("=" * 80)
    print(f"  Symbols: {', '.join(symbols)}")
    print(f"  Warmup: {WARMUP}, Horizons: {horizons}")

    ed = EnergyDynamics()
    tt = TemporalTopology()

    all_records = []
    for sym in symbols:
        print(f"\n  Generating latent states for {sym}...", end=" ")
        sys.stdout.flush()
        records = run_symbol(sym, ed, tt, horizons)
        print(f"{len(records)} records")
        all_records.extend(records)

    print(f"\n  Total records: {len(all_records)}")

    yearly = {}
    for r in all_records:
        yr = year_from_record(r)
        yearly.setdefault(yr, []).append(r)
    years = sorted(yearly.keys())
    print(f"  Year range: {years[0]}-{years[-1]}")

    results = {}

    for horizon in horizons:
        print(f"\n{'=' * 60}")
        print(f"  H{horizon} WALK-FORWARD (with TCMA)")
        print(f"{'=' * 60}")

        if len(years) < 4:
            print(f"  WARNING: only {len(years)} years, need >=4")
            continue

        split_results = []
        for split_idx in range(len(years) - 3):
            train_years = years[split_idx:split_idx + 2]
            test_year = years[split_idx + 2]

            train_records = []
            for y in train_years:
                train_records.extend(yearly.get(y, []))
            test_records = yearly.get(test_year, [])

            if len(train_records) < 50 or len(test_records) < 10:
                continue

            z_train_raw = np.array([[r[f"z{i}"] for i in range(dpl7_config.LATENT_DIM)] for r in train_records])
            ret_train = np.array([r[f"return_h{horizon}"] for r in train_records])
            z_test_raw = np.array([[r[f"z{i}"] for i in range(dpl7_config.LATENT_DIM)] for r in test_records])
            ret_test = np.array([r[f"return_h{horizon}"] for r in test_records])

            z_train_mean = np.mean(z_train_raw, axis=0)
            z_train_std = np.std(z_train_raw, axis=0) + 1e-8
            z_train = (z_train_raw - z_train_mean) / z_train_std
            z_test = (z_test_raw - z_train_mean) / z_train_std

            z_train_normed = np.array([z / (np.linalg.norm(z) + 1e-8) for z in z_train])
            z_test_normed = np.array([z / (np.linalg.norm(z) + 1e-8) for z in z_test])

            tcma = TemporalContrastiveAligner(dim=dpl7_config.LATENT_DIM, tau=0.2, lambda_drift=0.2)
            losses = tcma.pretrain(z_train_normed, train_records, n_epochs=3, lr=0.005)

            z_train_aligned = tcma.align_batch(z_train_normed)
            z_test_aligned = tcma.align_batch(z_test_normed)

            al_train = np.array([z / (np.linalg.norm(z) + 1e-8) for z in z_train_aligned])
            al_test = np.array([z / (np.linalg.norm(z) + 1e-8) for z in z_test_aligned])

            head = SignalHead(dim=dpl7_config.LATENT_DIM)

            train_signals = [head.predict(z)["direction"] for z in al_train]
            test_signals = [head.predict(z)["direction"] for z in al_test]
            train_conf = [head.predict(z)["confidence"] for z in al_train]
            test_conf = [head.predict(z)["confidence"] for z in al_test]

            train_scores = [head.predict(z)["score"] for z in al_train]
            test_scores = [head.predict(z)["score"] for z in al_test]

            train_ic = compute_ic(np.array(train_scores), ret_train)
            test_ic = compute_ic(np.array(test_scores), ret_test)
            train_da = compute_da(train_signals, ret_train.tolist())
            test_da = compute_da(test_signals, ret_test.tolist())

            drift = compute_latent_drift(z_train_normed, z_test_normed)
            train_lss = compute_lss(z_train_normed)
            test_lss = compute_lss(z_test_normed)
            train_tdi = compute_tdi(z_train_normed)
            test_tdi = compute_tdi(z_test_normed)

            train_pnl = simulate_pnl(train_signals, ret_train.tolist())
            test_pnl = simulate_pnl(test_signals, ret_test.tolist())

            split_label = f"{train_years[0]}-{train_years[-1]}->{test_year}"
            split_result = {
                "split": split_label,
                "train_years": train_years,
                "test_year": test_year,
                "n_train": len(train_records),
                "n_test": len(test_records),
                "train_ic": round(train_ic, 4),
                "test_ic": round(test_ic, 4),
                "train_da": round(train_da, 4),
                "test_da": round(test_da, 4),
                "latent_drift": round(drift, 4),
                "train_lss": round(train_lss, 4),
                "test_lss": round(test_lss, 4),
                "train_tdi": round(train_tdi, 4),
                "test_tdi": round(test_tdi, 4),
                "tcma_loss": [round(float(l), 4) for l in losses],
                "train_pnl": train_pnl,
                "test_pnl": test_pnl,
            }
            split_results.append(split_result)

            ic_str = "YES" if test_ic > 0.05 else "MARGINAL" if test_ic > 0.02 else "NO"
            da_str = "YES" if test_da > 0.55 else "MARGINAL" if test_da > 0.52 else "NO"
            lss_str = "STABLE" if test_lss > 0.80 else "UNSTABLE"

            print(f"\n  Split: {split_label}")
            print(f"    Train: {len(train_records)} | Test: {len(test_records)}")
            print(f"    Train IC: {train_ic:.4f} | Test IC: {test_ic:.4f} ({ic_str})")
            print(f"    Train DA: {train_da:.4f} | Test DA: {test_da:.4f} ({da_str})")
            print(f"    LSS: train={train_lss:.4f} test={test_lss:.4f} ({lss_str})")
            print(f"    TDI: train={train_tdi:.4f} test={test_tdi:.4f}")
            print(f"    Latent drift: {drift:.4f}")
            print(f"    TCMA loss: {losses[0]:.4f} -> {losses[-1]:.4f}")
            print(f"    Test PnL: base={test_pnl['baseline']:.4f} sized={test_pnl['sized']:.4f}")

        if split_results:
            test_ics = [s["test_ic"] for s in split_results]
            test_das = [s["test_da"] for s in split_results]
            drifts = [s["latent_drift"] for s in split_results]
            improvements = [s["test_pnl"]["improvement"] for s in split_results]
            test_lsss = [s["test_lss"] for s in split_results]
            test_tdis = [s["test_tdi"] for s in split_results]

            n_positive_ic = sum(1 for v in test_ics if v > 0.02)
            n_positive_da = sum(1 for v in test_das if v > 0.52)
            n_positive_pnl = sum(1 for v in improvements if v > 0)
            n_stable_lss = sum(1 for v in test_lsss if v > 0.80)
            n_splits = len(split_results)

            ic_survival = round(n_positive_ic / n_splits * 100, 1)
            da_survival = round(n_positive_da / n_splits * 100, 1)
            pnl_survival = round(n_positive_pnl / n_splits * 100, 1)
            lss_survival = round(n_stable_lss / n_splits * 100, 1)

            print(f"\n  --- H{horizon} Summary ---")
            print(f"    Splits: {n_splits}")
            print(f"    IC survival (>0.02): {n_positive_ic}/{n_splits} = {ic_survival}%  (mean={np.mean(test_ics):.4f})")
            print(f"    DA survival (>52%):  {n_positive_da}/{n_splits} = {da_survival}%  (mean={np.mean(test_das):.4f})")
            print(f"    PnL improve (>0):    {n_positive_pnl}/{n_splits} = {pnl_survival}%")
            print(f"    LSS stable (>0.80):  {n_stable_lss}/{n_splits} = {lss_survival}%  (mean={np.mean(test_lsss):.4f})")
            print(f"    Mean TDI:            {np.mean(test_tdis):.4f}")
            print(f"    Mean drift:          {np.mean(drifts):.4f}")
            print(f"    Max test IC:         {max(test_ics):.4f}")

            results[f"H{horizon}"] = {
                "n_splits": n_splits,
                "ic_survival_pct": ic_survival,
                "da_survival_pct": da_survival,
                "pnl_survival_pct": pnl_survival,
                "lss_survival_pct": lss_survival,
                "mean_test_ic": round(float(np.mean(test_ics)), 4),
                "mean_test_da": round(float(np.mean(test_das)), 4),
                "mean_lss": round(float(np.mean(test_lsss)), 4),
                "mean_tdi": round(float(np.mean(test_tdis)), 4),
                "mean_latent_drift": round(float(np.mean(drifts)), 4),
                "max_test_ic": round(float(max(test_ics)), 4),
                "splits": split_results
            }

    return results, all_records


def main():
    results, records = run()

    print(f"\n{'=' * 80}")
    print("  DPL-7.1 AGGREGATE FINDINGS")
    print("=" * 80)

    for h_label, h_res in results.items():
        print(f"\n  {h_label}:")
        print(f"    IC survival:  {h_res['ic_survival_pct']}%  (mean={h_res['mean_test_ic']})")
        print(f"    DA survival:  {h_res['da_survival_pct']}%  (mean={h_res['mean_test_da']})")
        print(f"    PnL improve:  {h_res['pnl_survival_pct']}%")
        print(f"    LSS stable:   {h_res['lss_survival_pct']}%  (mean={h_res['mean_lss']})")
        print(f"    TDI:          {h_res['mean_tdi']}")
        print(f"    Drift:        {h_res['mean_latent_drift']}")

    ic_ok = sum(1 for h in results.values() if h["mean_test_ic"] > 0.02)
    da_ok = sum(1 for h in results.values() if h["mean_test_da"] > 0.52)
    lss_ok = sum(1 for h in results.values() if h["mean_lss"] > 0.80)
    n_h = len(results)

    print(f"\n  Summary: IC={ic_ok}/{n_h}  DA={da_ok}/{n_h}  LSS={lss_ok}/{n_h}")

    if ic_ok == n_h and da_ok == n_h and lss_ok >= n_h - 1:
        print("  VERDICT: POSITIVE EDGE DETECTED (stable + predictive)")
    elif ic_ok >= n_h // 2 or da_ok >= n_h // 2:
        print("  VERDICT: MARGINAL (partial stabilization, inconsistent direction)")
    else:
        print("  VERDICT: NO EDGE (TCMA did not stabilize latent space sufficiently)")

    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "reports")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "dpl7_1_walkforward_results.json")

    serializable = {}
    for k, v in results.items():
        serializable[k] = v
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)

    print(f"\n  Results saved to: {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
