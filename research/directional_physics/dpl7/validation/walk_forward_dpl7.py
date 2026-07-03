"""
DPL-7 Walk-Forward Validation:
  - Continuous latent state encoding (no regime bins)
  - SignalHead trained per split via IC maximization
  - Metrics: IC, directional accuracy, latent drift, PnL attribution
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
        print(f"  WARNING: {path} not found, skipping {symbol}")
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


def split_by_year(records: list, train_years: list, test_year: int):
    def year_from_ts(ts):
        if isinstance(ts, (int, float, np.integer)):
            return int(ts)
        if isinstance(ts, datetime):
            return ts.year
        if isinstance(ts, np.datetime64):
            return ts.astype("datetime64[Y]").astype(int) + 1970
        return int(str(ts)[:4])
    train = [r for r in records if year_from_ts(r["timestamp"]) in train_years]
    test = [r for r in records if year_from_ts(r["timestamp"]) == test_year]
    idx_map = {r["idx"]: r for r in records}
    return train, test, year_from_ts


def compute_ic(z_list: list, returns_list: list, weights: np.ndarray, bias: float) -> float:
    if len(z_list) < 3:
        return 0.0
    z_arr = np.array(z_list)
    scores = z_arr @ weights + bias
    pred_dir = np.tanh(scores)
    ret_arr = np.array(returns_list)
    ic = float(np.corrcoef(pred_dir, ret_arr)[0, 1])
    return ic if not np.isnan(ic) else 0.0


def compute_directional_accuracy(signals: list, returns: list) -> float:
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


def simulate_pnl(signals: list, returns: list, confidence: list) -> dict:
    baseline_ret = float(np.sum(returns))
    sizing = SizingPolicy()
    sized_returns = []
    for s, r, c in zip(signals, returns, confidence):
        size = sizing.compute({"direction": s, "confidence": c})
        sized_returns.append(s * size * r)
    sized_ret = float(np.sum(sized_returns))
    n_trades = sum(1 for s in signals if abs(s) > 0.01)
    return {
        "baseline_return": round(baseline_ret, 4),
        "sized_return": round(sized_ret, 4),
        "improvement": round(sized_ret - baseline_ret, 4),
        "n_trades": n_trades
    }


def run(horizons=None, symbols=None):
    if horizons is None:
        horizons = HORIZONS
    if symbols is None:
        symbols = list(SYMBOLS)

    print("=" * 80)
    print("  DPL-7 WALK-FORWARD VALIDATION: Continuous Market State Model")
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

    results = {}

    for horizon in horizons:
        print(f"\n{'=' * 60}")
        print(f"  H{horizon} WALK-FORWARD")
        print(f"{'=' * 60}")

        yearly = {}
        for r in all_records:
            yr = None
            ts = r["timestamp"]
            if isinstance(ts, (int, float, np.integer)):
                yr = int(ts)
            elif isinstance(ts, np.datetime64):
                yr = ts.astype("datetime64[Y]").astype(int) + 1970
            elif isinstance(ts, datetime):
                yr = ts.year
            else:
                yr = int(str(ts)[:4])
            if yr is not None:
                yearly.setdefault(yr, []).append(r)

        years = sorted(yearly.keys())
        print(f"  Years: {years[0]}-{years[-1]}")

        row_labels = [f"z{i}" for i in range(dpl7_config.LATENT_DIM)]
        feature_names = StateEncoder(dim=dpl7_config.LATENT_DIM).feature_keys
        all_labels = feature_names + row_labels

        if len(years) < 4:
            print(f"  WARNING: only {len(years)} years, need >=4 for walk-forward")
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

            z_train = np.array([[r[f"z{i}"] for i in range(dpl7_config.LATENT_DIM)] for r in train_records])
            ret_train = np.array([r[f"return_h{horizon}"] for r in train_records])
            z_test = np.array([[r[f"z{i}"] for i in range(dpl7_config.LATENT_DIM)] for r in test_records])
            ret_test = np.array([r[f"return_h{horizon}"] for r in test_records])

            z_train_mean = np.mean(z_train, axis=0)
            z_train_std = np.std(z_train, axis=0) + 1e-8
            z_train_norm = (z_train - z_train_mean) / z_train_std
            z_test_norm = (z_test - z_train_mean) / z_train_std

            head = SignalHead(dim=dpl7_config.LATENT_DIM)
            train_ic = head.fit(z_train_norm, ret_train, lr=dpl7_config.LEARNING_RATE, epochs=dpl7_config.N_EPOCHS)
            test_ic = head.ic(z_test_norm, ret_test)

            train_signals = [head.predict(z)["direction"] for z in z_train_norm]
            test_signals = [head.predict(z)["direction"] for z in z_test_norm]
            train_conf = [head.predict(z)["confidence"] for z in z_train_norm]
            test_conf = [head.predict(z)["confidence"] for z in z_test_norm]

            train_da = compute_directional_accuracy(train_signals, ret_train.tolist())
            test_da = compute_directional_accuracy(test_signals, ret_test.tolist())

            drift = compute_latent_drift(z_train_norm, z_test_norm)

            train_pnl = simulate_pnl(train_signals, ret_train.tolist(), train_conf)
            test_pnl = simulate_pnl(test_signals, ret_test.tolist(), test_conf)

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
                "train_pnl": train_pnl,
                "test_pnl": test_pnl
            }
            split_results.append(split_result)

            da_str = "YES" if test_da > 0.55 else "MARGINAL" if test_da > 0.52 else "NO"
            ic_str = "YES" if test_ic > 0.05 else "MARGINAL" if test_ic > 0.02 else "NO"

            print(f"\n  Split: {split_label}")
            print(f"    Train: {len(train_records)} | Test: {len(test_records)}")
            print(f"    Train IC: {train_ic:.4f} | Test IC: {test_ic:.4f} ({ic_str})")
            print(f"    Train DA: {train_da:.4f} | Test DA: {test_da:.4f} ({da_str})")
            print(f"    Latent drift: {drift:.4f}")
            print(f"    Train PnL base: {train_pnl['baseline_return']:.4f} sized: {train_pnl['sized_return']:.4f}")
            print(f"    Test PnL base: {test_pnl['baseline_return']:.4f} sized: {test_pnl['sized_return']:.4f}")

        if split_results:
            test_ics = [s["test_ic"] for s in split_results]
            test_das = [s["test_da"] for s in split_results]
            drifts = [s["latent_drift"] for s in split_results]
            improvements = [s["test_pnl"]["improvement"] for s in split_results]

            n_positive_ic = sum(1 for v in test_ics if v > 0.02)
            n_positive_da = sum(1 for v in test_das if v > 0.52)
            n_positive_pnl = sum(1 for v in improvements if v > 0)
            n_splits = len(split_results)

            ic_survival = round(n_positive_ic / n_splits * 100, 1) if n_splits > 0 else 0
            da_survival = round(n_positive_da / n_splits * 100, 1) if n_splits > 0 else 0
            pnl_survival = round(n_positive_pnl / n_splits * 100, 1) if n_splits > 0 else 0

            print(f"\n  --- H{horizon} Summary ---")
            print(f"    Splits: {n_splits}")
            print(f"    IC survival (>0.02): {n_positive_ic}/{n_splits} = {ic_survival}%  (mean={np.mean(test_ics):.4f})")
            print(f"    DA survival (>52%):  {n_positive_da}/{n_splits} = {da_survival}%  (mean={np.mean(test_das):.4f})")
            print(f"    PnL improve (>0):    {n_positive_pnl}/{n_splits} = {pnl_survival}%")
            print(f"    Mean latent drift:   {np.mean(drifts):.4f}")
            print(f"    Max IC: {max(test_ics):.4f} | Best split IC: {max(test_ics):.4f}")

            results[f"H{horizon}"] = {
                "n_splits": n_splits,
                "ic_survival_pct": ic_survival,
                "da_survival_pct": da_survival,
                "pnl_survival_pct": pnl_survival,
                "mean_test_ic": round(float(np.mean(test_ics)), 4),
                "mean_test_da": round(float(np.mean(test_das)), 4),
                "mean_latent_drift": round(float(np.mean(drifts)), 4),
                "max_test_ic": round(float(max(test_ics)), 4),
                "max_test_da": round(float(max(test_das)), 4),
                "splits": split_results
            }

    return results, all_records


def main():
    results, records = run()

    print(f"\n{'=' * 80}")
    print("  DPL-7 AGGREGATE FINDINGS")
    print("=" * 80)

    all_positive = 0
    all_marginal = 0
    all_negative = 0

    for h_label, h_res in results.items():
        print(f"\n  {h_label}:")
        print(f"    IC survival (>{'>'}0.02):  {h_res['ic_survival_pct']}%  (mean={h_res['mean_test_ic']})")
        print(f"    DA survival (>52%):     {h_res['da_survival_pct']}%  (mean={h_res['mean_test_da']})")
        print(f"    PnL improve (>0):       {h_res['pnl_survival_pct']}%")
        print(f"    Latent drift:           {h_res['mean_latent_drift']}")
        print(f"    Max IC:                 {h_res['max_test_ic']}")

        if h_res["mean_test_ic"] > 0.03 and h_res["mean_test_da"] > 0.52:
            all_positive += 1
        elif h_res["mean_test_ic"] > 0.01 or h_res["mean_test_da"] > 0.51:
            all_marginal += 1
        else:
            all_negative += 1

    positive = all_positive > 0
    marginal = all_marginal > 0 and not positive

    print(f"\n  VERDICT: ", end="")
    if positive:
        print("POSITIVE EDGE DETECTED (IC+DA hold across horizons)")
    elif marginal:
        print("MARGINAL (some signal, insufficient for deployment)")
    else:
        print("NO EDGE (continuous latent state does not improve on baseline)")

    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "reports")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "dpl7_walkforward_results.json")

    serializable = {}
    for k, v in results.items():
        serializable[k] = v

    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\n  Results saved to: {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
