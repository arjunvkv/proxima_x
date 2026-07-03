"""
DPL-7.3: TCMA + LinearICHead
  - TCMA: temporal contrastive alignment (stabilized latent space)
  - LinearICHead: single linear projection via IC-aligned gradient
  - Final step before causal filter
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
from research.directional_physics.dpl7.policy.signal_head import LinearICHead
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


def run_symbol(symbol, ed, tt, horizons=None):
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
        record = {"idx": idx, "timestamp": data["timestamps"][idx], "symbol": symbol}
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


def compute_latent_drift(z_train, z_test):
    if len(z_train) < 1 or len(z_test) < 1:
        return 1.0
    m1 = np.mean(np.array(z_train), axis=0)
    m2 = np.mean(np.array(z_test), axis=0)
    drift = float(np.linalg.norm(m1 - m2))
    avg = float(np.mean([np.linalg.norm(z) for z in z_train])) + 1e-8
    return drift / avg


def compute_lss(z_series):
    if len(z_series) < 3:
        return 0.0
    sims = []
    arr = np.array(z_series)
    for i in range(len(arr) - 1):
        a = arr[i] / (np.linalg.norm(arr[i]) + 1e-8)
        b = arr[i + 1] / (np.linalg.norm(arr[i + 1]) + 1e-8)
        sims.append(float(np.dot(a, b)))
    return float(np.mean(sims))


def compute_tdi(z_series, alpha=0.1):
    if len(z_series) < 2:
        return 0.0
    ema = None
    vals = []
    for z in z_series:
        if ema is None:
            ema = z.copy()
        else:
            ema = alpha * z + (1 - alpha) * ema
        vals.append(float(np.linalg.norm(z - ema)))
    return float(np.mean(vals))


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
    print("  DPL-7.3 WALK-FORWARD: TCMA + LinearICHead")
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
        print(f"  H{horizon} WALK-FORWARD (TCMA + LinearICHead)")
        print(f"{'=' * 60}")

        if len(years) < 4:
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

            n_dim = dpl7_config.LATENT_DIM
            z_train_raw = np.array([[r[f"z{i}"] for i in range(n_dim)] for r in train_records])
            ret_train = np.array([r[f"return_h{horizon}"] for r in train_records])
            z_test_raw = np.array([[r[f"z{i}"] for i in range(n_dim)] for r in test_records])
            ret_test = np.array([r[f"return_h{horizon}"] for r in test_records])

            mu = np.mean(z_train_raw, axis=0)
            sd = np.std(z_train_raw, axis=0) + 1e-8
            z_train_s = (z_train_raw - mu) / sd
            z_test_s = (z_test_raw - mu) / sd

            z_train_n = np.array([z / (np.linalg.norm(z) + 1e-8) for z in z_train_s])
            z_test_n = np.array([z / (np.linalg.norm(z) + 1e-8) for z in z_test_s])

            tcma = TemporalContrastiveAligner(dim=n_dim, tau=0.5, lambda_drift=0.1)
            losses = tcma.pretrain(z_train_n, train_records, n_epochs=3, lr=0.001)

            z_train_a = tcma.align_batch(z_train_n)
            z_test_a = tcma.align_batch(z_test_n)

            al_train = np.array([z / (np.linalg.norm(z) + 1e-8) for z in z_train_a])
            al_test = np.array([z / (np.linalg.norm(z) + 1e-8) for z in z_test_a])

            head = LinearICHead(dim=n_dim, lr=0.001)
            train_ic = head.fit(al_train, ret_train, epochs=10)

            train_scores = np.array([head.forward(z)["score"] for z in al_train])
            test_scores = np.array([head.forward(z)["score"] for z in al_test])
            train_dir = np.tanh(train_scores)
            test_dir = np.tanh(test_scores)

            test_ic = compute_ic(test_scores, ret_test)
            train_da = compute_da(train_dir.tolist(), ret_train.tolist())
            test_da = compute_da(test_dir.tolist(), ret_test.tolist())

            drift = compute_latent_drift(al_train, al_test)
            train_lss = compute_lss(al_train)
            test_lss = compute_lss(al_test)
            train_tdi = compute_tdi(al_train)
            test_tdi = compute_tdi(al_test)

            train_pnl = simulate_pnl(train_dir.tolist(), ret_train.tolist())
            test_pnl = simulate_pnl(test_dir.tolist(), ret_test.tolist())

            split_label = f"{train_years[0]}-{train_years[-1]}->{test_year}"
            split_result = {
                "split": split_label,
                "train_years": list(train_years),
                "test_year": test_year,
                "n_train": len(train_records), "n_test": len(test_records),
                "train_ic": round(train_ic, 4), "test_ic": round(test_ic, 4),
                "train_da": round(train_da, 4), "test_da": round(test_da, 4),
                "latent_drift": round(drift, 4),
                "train_lss": round(train_lss, 4), "test_lss": round(test_lss, 4),
                "train_tdi": round(train_tdi, 4), "test_tdi": round(test_tdi, 4),
                "tcma_loss": [round(float(l), 4) for l in losses],
                "train_pnl": train_pnl, "test_pnl": test_pnl
            }
            split_results.append(split_result)

            ic_s = "YES" if test_ic > 0.05 else "MARGINAL" if test_ic > 0.02 else "NO"
            da_s = "YES" if test_da > 0.55 else "MARGINAL" if test_da > 0.52 else "NO"
            lss_s = "STABLE" if test_lss > 0.80 else "UNSTABLE"

            print(f"\n  Split: {split_label}")
            print(f"    Train: {len(train_records)} | Test: {len(test_records)}")
            print(f"    Train IC: {train_ic:.4f} | Test IC: {test_ic:.4f} ({ic_s})")
            print(f"    Train DA: {train_da:.4f} | Test DA: {test_da:.4f} ({da_s})")
            print(f"    LSS: {test_lss:.4f} ({lss_s}) | TDI: {test_tdi:.4f} | Drift: {drift:.4f}")
            print(f"    TCMA loss: {losses[0]:.4f} -> {losses[-1]:.4f}")
            print(f"    Test PnL: base={test_pnl['baseline']:.4f} sized={test_pnl['sized']:.4f} (improve={test_pnl['improvement']:.4f})")

        if split_results:
            test_ics = [s["test_ic"] for s in split_results]
            test_das = [s["test_da"] for s in split_results]
            drifts = [s["latent_drift"] for s in split_results]
            improvements = [s["test_pnl"]["improvement"] for s in split_results]
            test_lsss = [s["test_lss"] for s in split_results]
            test_tdis = [s["test_tdi"] for s in split_results]

            n_ic = sum(1 for v in test_ics if v > 0.02)
            n_da = sum(1 for v in test_das if v > 0.52)
            n_pnl = sum(1 for v in improvements if v > 0)
            n_lss = sum(1 for v in test_lsss if v > 0.80)
            ns = len(split_results)

            print(f"\n  --- H{horizon} Summary ---")
            print(f"    Splits: {ns}")
            print(f"    IC survival (>0.02): {n_ic}/{ns} = {round(n_ic/ns*100,1)}%  (mean={np.mean(test_ics):.4f})")
            print(f"    DA survival (>52%):  {n_da}/{ns} = {round(n_da/ns*100,1)}%  (mean={np.mean(test_das):.4f})")
            print(f"    PnL improve (>0):    {n_pnl}/{ns} = {round(n_pnl/ns*100,1)}%")
            print(f"    LSS stable (>0.80):  {n_lss}/{ns} = {round(n_lss/ns*100,1)}%  (mean={np.mean(test_lsss):.4f})")
            print(f"    Mean TDI: {np.mean(test_tdis):.4f} | Mean drift: {np.mean(drifts):.4f} | Max IC: {max(test_ics):.4f}")
            print(f"    IC values: {[round(v,4) for v in test_ics]}")

            results[f"H{horizon}"] = {
                "n_splits": ns,
                "ic_survival_pct": round(n_ic / ns * 100, 1),
                "da_survival_pct": round(n_da / ns * 100, 1),
                "pnl_survival_pct": round(n_pnl / ns * 100, 1),
                "lss_survival_pct": round(n_lss / ns * 100, 1),
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
    print("  DPL-7.3 AGGREGATE FINDINGS")
    print("=" * 80)

    for h_label, h_res in results.items():
        print(f"\n  {h_label}:")
        print(f"    IC survival:  {h_res['ic_survival_pct']}%  (mean={h_res['mean_test_ic']})")
        print(f"    DA survival:  {h_res['da_survival_pct']}%  (mean={h_res['mean_test_da']})")
        print(f"    PnL improve:  {h_res['pnl_survival_pct']}%")
        print(f"    LSS stable:   {h_res['lss_survival_pct']}%  (mean={h_res['mean_lss']})")
        print(f"    Max IC:       {h_res['max_test_ic']}")

    ic_ok = sum(1 for h in results.values() if h["mean_test_ic"] > 0.02)
    da_ok = sum(1 for h in results.values() if h["mean_test_da"] > 0.53)
    lss_ok = sum(1 for h in results.values() if h["mean_lss"] > 0.80)
    pnl_ok = sum(1 for h in results.values() if h["pnl_survival_pct"] >= 50)
    n_h = len(results)

    print(f"\n  Summary: IC={ic_ok}/{n_h}  DA={da_ok}/{n_h}  LSS={lss_ok}/{n_h}  PnL={pnl_ok}/{n_h}")

    edge = False
    if n_h > 0:
        mean_ic = np.mean([h["mean_test_ic"] for h in results.values()])
        mean_da = np.mean([h["mean_test_da"] for h in results.values()])
        overall_pnl = sum(h["pnl_survival_pct"] for h in results.values()) / n_h
        if mean_ic > 0.02 and mean_da > 0.53 and overall_pnl >= 50:
            edge = True

    if edge:
        print("  VERDICT: POSITIVE EDGE CONFIRMED (stable TCMA + linear IC readout)")
    elif ic_ok >= n_h // 2 and da_ok >= n_h // 2:
        print("  VERDICT: NEAR EDGE (strong stabilization, needs causal filter)")
    elif ic_ok >= 1 or da_ok >= 1:
        print("  VERDICT: MARGINAL (partial signal in some splits)")
    else:
        print("  VERDICT: NO EDGE (TCMA + linear readout insufficient)")

    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "reports")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "dpl7_3_walkforward_results.json")
    serializable = {}
    for k, v in results.items():
        serializable[k] = v
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\n  Results saved to: {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
