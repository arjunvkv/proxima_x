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
from research.directional_physics.dpl7.features.feature_bridge import FeatureBridge
from research.directional_physics.dpl7.config import dpl7_config
from research.directional_physics.dpl_11.core.signal_projection import FrozenSignalProjection
from research.directional_physics.dpl_11.core.position_sizing import PositionSizer, rolling_da, rolling_sharpe
from research.directional_physics.dpl_11.core.portfolio_basket import RiskWeightedBasket
from research.directional_physics.dpl_11.core.drawdown_controller import DrawdownController
from research.directional_physics.dpl_11.core.execution_gate import ExecutionGate
from research.mechanism_discovery.energy_dynamics import EnergyDynamics
from research.mechanism_discovery.temporal_topology import TemporalTopology

DATA_DIR = dpl7_config.DATA_DIR
if DATA_DIR is None:
    DATA_DIR = os.path.join(PROJECT_ROOT, "data", "market")
SYMBOLS = dpl7_config.SYMBOLS
HORIZONS = [20]
WARMUP = dpl7_config.WARMUP_BARS
SIZING_WINDOW = 60


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
        for h in HORIZONS:
            fwd_ret = (price[idx + h] - price[idx]) / price[idx]
            rec[f"return_h{h}"] = float(fwd_ret)
            rec[f"direction_h{h}"] = 1.0 if fwd_ret > 0 else 0.0
        records.append(rec)
    z_seq = np.array(z_list, dtype=np.float64)
    return records, z_seq, np.array(years_list), price


def evaluate_split(sym, train_records, test_records, z_train, z_test, records_all):
    train_rets = np.array([r["return_h20"] for r in train_records])
    test_rets = np.array([r["return_h20"] for r in test_records])

    projector = FrozenSignalProjection(dim=dpl7_config.LATENT_DIM)
    projector.fit(z_train, train_records, n_tcma_epochs=3)

    train_raw = projector.raw_signal_batch(z_train)
    test_raw = projector.raw_signal_batch(z_test)
    train_dir = np.tanh(train_raw)
    test_dir = np.tanh(test_raw)

    train_da = float(np.mean((train_dir > 0) == (train_rets > 0)))
    test_da = float(np.mean((test_dir > 0) == (test_rets > 0)))
    train_ic = float(np.corrcoef(train_raw, train_rets)[0, 1]) if len(train_raw) > 2 else 0.0
    test_ic = float(np.corrcoef(test_raw, test_rets)[0, 1]) if len(test_raw) > 2 else 0.0

    raw_pnl_base = float(np.sum(train_rets))
    raw_pnl = float(np.sum(train_dir * train_rets))

    sizer = PositionSizer(base_risk=0.02, vol_target=0.15)

    test_vol_s, test_dd_s, test_da_s = sizer.calibrate(test_dir, test_rets, np.cumsum(test_dir * test_rets), SIZING_WINDOW)

    ex_gate = ExecutionGate(vol_percentile=40, min_da=0.55)
    ex_gate.fit(train_rets)
    train_vol_regime = np.zeros(len(train_rets))
    for i in range(20, len(train_rets)):
        train_vol_regime[i] = float(np.std(train_rets[i - 20:i]))
    test_vol_regime = np.zeros(len(test_rets))
    for i in range(20, len(test_rets)):
        test_vol_regime[i] = float(np.std(test_rets[i - 20:i]))

    test_rolling_da = rolling_da(test_dir, test_rets, SIZING_WINDOW)

    sized_signals = np.zeros(len(test_dir))
    for i in range(len(test_dir)):
        gate = ex_gate.gate(z_test[i], test_dir[i], test_vol_regime[i], test_rolling_da[i])
        size = test_vol_s[i] * test_dd_s[i] * test_da_s[i] * gate
        sized_signals[i] = test_dir[i] * float(np.clip(size, 0.0, 2.0))

    sized_pnl = float(np.sum(sized_signals * test_rets))
    raw_test_pnl = float(np.sum(test_dir * test_rets))

    return {
        "train_da": round(train_da, 4), "test_da": round(test_da, 4),
        "train_ic": round(train_ic, 4) if not np.isnan(train_ic) else 0.0,
        "test_ic": round(test_ic, 4) if not np.isnan(test_ic) else 0.0,
        "raw_test_pnl": round(raw_test_pnl, 6),
        "sized_test_pnl": round(sized_pnl, 6),
        "sized_signals": sized_signals,
        "test_rets": test_rets,
        "n_test": len(test_rets)
    }


def main():
    print("=" * 80)
    print("  DPL-11 WALK-FORWARD: Weak Edge Monetization System")
    print("=" * 80)
    print(f"  Symbols: {', '.join(SYMBOLS)}")
    print(f"  Horizon: H20, Sizing window: {SIZING_WINDOW}")
    print(f"  Base risk: 2%, Vol target: 15%")
    print("  Strategy: TCMA projection + vol/dd/DA sizing + execution gate")
    print("  Goal: convert 61-66% DA into expectancy-positive portfolio")

    ed = EnergyDynamics()
    tt = TemporalTopology()

    symbol_data = {}
    for sym in SYMBOLS:
        print(f"\n  Processing {sym}...", end=" ")
        sys.stdout.flush()
        records, z_seq, years, price = run_symbol(sym, ed, tt)
        if records is not None:
            print(f"{len(records)} records")
            symbol_data[sym] = {"records": records, "z_seq": z_seq, "years": years}

    all_test_years = set()
    for sd in symbol_data.values():
        for y in sd["years"]:
            if y is not None:
                all_test_years.add(int(y))
    all_test_years = sorted(all_test_years)

    portfolio_results = {}

    for test_year in all_test_years[3:]:
        train_years = [y for y in all_test_years if y < test_year and y >= test_year - 3]
        train_years = train_years[-2:] if len(train_years) >= 2 else train_years

        print(f"\n{'=' * 60}")
        print(f"  TEST YEAR: {test_year}  (train: {train_years})")
        print(f"{'=' * 60}")

        split_pnls = {"raw": {}, "sized": {}, "da": {}, "ic": {}}
        split_results = []

        for sym, sd in symbol_data.items():
            records = sd["records"]
            z_seq = sd["z_seq"]
            years_arr = sd["years"]

            train_mask = np.array([y in train_years for y in years_arr])
            test_mask = np.array([y == test_year for y in years_arr])

            if np.sum(train_mask) < 100 or np.sum(test_mask) < 20:
                continue

            train_records = [records[i] for i in range(len(records)) if train_mask[i]]
            test_records = [records[i] for i in range(len(records)) if test_mask[i]]
            z_train = z_seq[train_mask]
            z_test = z_seq[test_mask]

            result = evaluate_split(sym, train_records, test_records, z_train, z_test, records)
            result["symbol"] = sym
            split_results.append(result)

            split_pnls["raw"][sym] = result["raw_test_pnl"]
            split_pnls["sized"][sym] = result["sized_test_pnl"]
            split_pnls["da"][sym] = result["test_da"]
            split_pnls["ic"][sym] = result["test_ic"]

            print(f"  {sym:8s}: DA={result['test_da']:.4f} IC={result['test_ic']:.4f}  raw={result['raw_test_pnl']:.6f}  sized={result['sized_test_pnl']:.6f}")

        if len(split_results) < 2:
            continue

        basket = RiskWeightedBasket()
        raw_rets_dict = {}
        sized_rets_dict = {}
        for sr in split_results:
            raw_rets_dict[sr["symbol"]] = sr["test_rets"]
            sized_rets_dict[sr["symbol"]] = sr["sized_signals"] * sr["test_rets"]

        basket.calibrate_weights(raw_rets_dict)
        n_min = min(len(v) for v in raw_rets_dict.values())

        portfolio_raw_pnl = np.zeros(n_min)
        portfolio_sized_pnl = np.zeros(n_min)
        for i, sr in enumerate(split_results):
            sym = sr["symbol"]
            w = basket.weight(sym)
            portfolio_raw_pnl += w * sr["test_rets"][:n_min]
            portfolio_sized_pnl += w * (sr["sized_signals"][:n_min] * sr["test_rets"][:n_min])

        def compute_metrics(pnl_series):
            if len(pnl_series) < 5:
                return {"mean": 0.0, "std": 0.0, "sharpe": 0.0, "total": 0.0, "max_dd": 0.0}
            total = float(np.sum(pnl_series))
            mean = float(np.mean(pnl_series))
            std = float(np.std(pnl_series)) + 1e-8
            sharpe = float(mean / std * np.sqrt(252))
            eq = np.cumsum(pnl_series)
            pk = np.maximum.accumulate(eq)
            dd = (eq - pk) / (pk + 1e-8)
            max_dd = float(np.min(dd))
            return {"total": round(total, 6), "mean": round(mean, 6),
                    "std": round(std, 6), "sharpe": round(sharpe, 4),
                    "max_dd": round(max_dd, 4)}

        raw_metrics = compute_metrics(portfolio_raw_pnl)
        sized_metrics = compute_metrics(portfolio_sized_pnl)

        portfolio_results[test_year] = {
            "train_years": train_years,
            "n_symbols": len(split_results),
            "symbols": list(split_pnls["da"].keys()),
            "mean_da": round(float(np.mean(list(split_pnls["da"].values()))), 4),
            "mean_ic": round(float(np.mean(list(split_pnls["ic"].values()))), 4),
            "raw_portfolio": raw_metrics,
            "sized_portfolio": sized_metrics,
        }

        print(f"\n  Portfolio ({len(split_results)} assets):")
        print(f"    Mean DA: {portfolio_results[test_year]['mean_da']:.4f}  Mean IC: {portfolio_results[test_year]['mean_ic']:.4f}")
        print(f"    RAW:  total={raw_metrics['total']:.6f}  sharpe={raw_metrics['sharpe']:.4f}  DD={raw_metrics['max_dd']:.4f}")
        print(f"    SIZED: total={sized_metrics['total']:.6f}  sharpe={sized_metrics['sharpe']:.4f}  DD={sized_metrics['max_dd']:.4f}")

    print(f"\n{'=' * 80}")
    print("  DPL-11 AGGREGATE FINDINGS")
    print("=" * 80)

    raw_sharpes = [v["raw_portfolio"]["sharpe"] for v in portfolio_results.values()]
    sized_sharpes = [v["sized_portfolio"]["sharpe"] for v in portfolio_results.values()]
    raw_totals = [v["raw_portfolio"]["total"] for v in portfolio_results.values()]
    sized_totals = [v["sized_portfolio"]["total"] for v in portfolio_results.values()]
    raw_dds = [v["raw_portfolio"]["max_dd"] for v in portfolio_results.values()]
    sized_dds = [v["sized_portfolio"]["max_dd"] for v in portfolio_results.values()]
    mean_das = [v["mean_da"] for v in portfolio_results.values()]
    years_list = list(portfolio_results.keys())

    print(f"\n  Walk-forward years: {years_list[0]} - {years_list[-1]}")
    print(f"  {len(years_list)} test years, {len(SYMBOLS)} symbols")

    print(f"\n  Raw portfolio:")
    print(f"    Mean Sharpe: {np.mean(raw_sharpes):.4f}")
    print(f"    Mean total return: {np.mean(raw_totals):.6f}")
    print(f"    Mean max DD: {np.mean(raw_dds):.4f}")
    print(f"    Sharpe > 0: {sum(1 for v in raw_sharpes if v > 0)}/{len(raw_sharpes)}")

    print(f"\n  Sized+Gated portfolio:")
    print(f"    Mean Sharpe: {np.mean(sized_sharpes):.4f}")
    print(f"    Mean total return: {np.mean(sized_totals):.6f}")
    print(f"    Mean max DD: {np.mean(sized_dds):.4f}")
    print(f"    Sharpe > 0: {sum(1 for v in sized_sharpes if v > 0)}/{len(sized_sharpes)}")

    improvement = float(np.mean(sized_sharpes) - np.mean(raw_sharpes))
    print(f"\n    Sharpe improvement from sizing: {improvement:+.4f}")

    total_return_raw = float(np.mean(raw_totals))
    total_return_sized = float(np.mean(sized_totals))
    expectancy_improvement = total_return_sized - total_return_raw

    print(f"\n    Avg test DA: {np.mean(mean_das):.4f}")

    if expectancy_improvement > 0 and np.mean(sized_sharpes) > 0.5:
        print("\n  VERDICT: POSITIVE EDGE CONFIRMED (monetization adds value)")
    elif expectancy_improvement > 0:
        print("\n  VERDICT: MARGINAL (sizing helps but insufficient for stable edge)")
    elif np.mean(raw_sharpes) > 0.3:
        print("\n  VERDICT: RAW EDGE EXISTS (sizing/gating harms more than helps)")
    else:
        print("\n  VERDICT: NO EDGE (even weak directional bias not monetizable)")

    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "reports")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "dpl11_walkforward_results.json")
    serializable = {}
    for k, v in portfolio_results.items():
        serializable[str(k)] = v
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\n  Results saved to: {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
